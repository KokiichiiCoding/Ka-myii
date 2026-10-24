"""
Background generation jobs.

A real run takes ~11 minutes and needs the whole 8 GB GPU, so ``/generate`` only queues
the request: one worker thread runs jobs strictly one at a time, and the browser polls
``/api/generation/jobs/<id>`` for state and progress (so a reloaded tab can re-attach).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.models.vtuber_model import GenerationRequest
from src.utils.progress_tracker import get_progress_manager

logger = logging.getLogger(__name__)

# Share of a real run per stage, from measured timings (~2 min paint, ~10–12 min decompose,
# seconds to package) — turns per-stage trackers into one bar that never goes backwards.
STAGES = {
    "image": (0.0, 0.12, "Painting the character…"),
    "separate": (0.12, 0.86, "Separating layers…"),
    "expressions": (0.86, 0.97, "Painting expressions…"),
    "assemble": (0.97, 1.0, "Packaging PSD + Cubism files…"),
}
# Jobs that only run some stages spread the bar over those.
STAGES_BY_OPERATION = {
    "expressions": {"expressions": (0.0, 0.9, "Painting expressions…"),
                    "assemble": (0.9, 1.0, "Packaging PSD + Cubism files…")},
    "edit": {"image": (0.0, 1.0, "Repainting the selection…")},
    "extend": {"image": (0.0, 1.0, "Extending the canvas…")},
    "refine": {"image": (0.0, 1.0, "Refining details…")},
}
KEEP_FINISHED = 20
ACTIVE = ("queued", "running")


@dataclass
class Job:
    id: str
    request: GenerationRequest
    state: str = "queued"  # queued | running | completed | failed | cancelled
    model_id: Optional[str] = None
    error: Optional[str] = None
    cancel_requested: bool = False
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    restarts: int = 0  # times the server stopped while this job was running


class JobQueue:
    """FIFO of generation requests executed by a single daemon worker thread.

    With ``state_path``, queued and running jobs are saved there and restored (as queued, same
    ids) when the server starts again, so an overnight batch survives a restart; a job that was
    interrupted mid-run starts over.
    """

    def __init__(self, run: Callable[[GenerationRequest, str], str], state_path: Optional[Path] = None):
        # run(request, job_id) executes one job and returns the finished model id.
        self._run = run
        self._jobs: Dict[str, Job] = {}
        self._cv = threading.Condition()
        self._state_path = state_path
        for saved in self._load():
            self._jobs[saved.id] = saved
        if self._jobs:
            logger.info("Restored %d unfinished job(s) from %s", len(self._jobs), state_path)
        threading.Thread(target=self._loop, name="kamyii-jobs", daemon=True).start()

    def _load(self) -> List[Job]:
        if not self._state_path or not self._state_path.exists():
            return []
        try:
            known = {f.name for f in fields(GenerationRequest)}
            jobs = []
            for entry in json.loads(self._state_path.read_text(encoding="utf-8")):
                restarts = entry.get("restarts", 0) + (entry.get("state") == "running")
                if restarts > 2:  # it keeps dying with the server; don't crash-loop on it
                    logger.warning("Dropping job %s: interrupted %d times", entry["id"], restarts)
                    continue
                request = GenerationRequest(**{k: v for k, v in entry["request"].items() if k in known})
                jobs.append(Job(id=entry["id"], request=request, restarts=restarts))
            return jobs
        except Exception as exc:  # a damaged file must not stop the server
            logger.warning("Ignoring unreadable job queue %s: %s", self._state_path, exc)
            return []

    def _save(self):
        """Write the unfinished jobs (call with ``self._cv`` held)."""
        if not self._state_path:
            return
        pending = [{"id": j.id, "state": j.state, "restarts": j.restarts, "request": asdict(j.request)}
                   for j in self._jobs.values() if j.state in ACTIVE]
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(pending, indent=1), encoding="utf-8")
            os.replace(tmp, self._state_path)
        except OSError as exc:  # persistence is a convenience; the in-memory queue keeps working
            logger.warning("Could not save the job queue: %s", exc)

    def submit(self, request: GenerationRequest) -> Job:
        job = Job(id=f"gen_{uuid.uuid4().hex[:8]}", request=request)
        with self._cv:
            self._jobs[job.id] = job
            self._prune()
            self._save()
            self._cv.notify()
        logger.info("Queued job %s: %s", job.id, request.prompt[:60])
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def jobs(self) -> List[Job]:
        with self._cv:
            return list(self._jobs.values())

    def cancel(self, job_id: str) -> Optional[Job]:
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None or job.state not in ACTIVE:
                return job
            job.cancel_requested = True
            if job.state == "queued":
                job.state, job.finished = "cancelled", time.time()
                for staged in (job.request.reference_image, job.request.mask_image):
                    if staged:  # uploaded/staged files that will now never be used
                        Path(staged).unlink(missing_ok=True)
                self._save()
            else:
                # The running stage raises InterruptedError at its next check (diffusion step,
                # See-through output line, or stage boundary).
                get_progress_manager().cancel(job_id)
        logger.info("Cancel requested for job %s", job_id)
        return job

    def describe(self, job: Job) -> dict:
        data = {
            "id": job.id,
            "state": job.state,
            "prompt": job.request.prompt,
            "model_id": job.model_id,
            "error": job.error,
            "cancel_requested": job.cancel_requested,
            "created": job.created,
            "started": job.started,
            "finished": job.finished,
            "progress": 100.0 if job.state == "completed" else 0.0,
            "message": {"queued": "Waiting for the GPU…", "completed": "Done",
                        "failed": "Failed", "cancelled": "Cancelled"}.get(job.state, ""),
            "stage": None,
            "eta": None,
        }
        if job.state == "queued":
            with self._cv:  # dict order is submission order (timestamps tie on Windows' coarse clock)
                data["position"] = [j for j in self._jobs.values() if j.state == "queued"].index(job)
        elif job.state == "running":
            manager = get_progress_manager()
            stage = manager.stages.get(job.id, "image")
            stages = STAGES_BY_OPERATION.get(getattr(job.request, "operation", "generate"), STAGES)
            lo, hi, label = stages.get(stage, (0.0, 1.0, ""))
            tracker = manager.get_tracker(job.id)
            frac = min(1.0, tracker.current_step / tracker.total_steps) if tracker and tracker.total_steps else 0.0
            data.update(
                stage=stage,
                progress=round(100 * (lo + (hi - lo) * frac), 1),
                message="Cancelling…" if job.cancel_requested else (tracker.message if tracker and tracker.message else label),
                # Per-stage ETA only: stage durations differ too much to extrapolate a total.
                eta=round(tracker.eta) if tracker and tracker.current_step else None,
            )
        return data

    def _prune(self):
        finished = [j for j in self._jobs.values() if j.state not in ACTIVE]
        for job in finished[:-KEEP_FINISHED]:
            del self._jobs[job.id]

    def _next(self) -> Optional[Job]:
        return next((j for j in self._jobs.values() if j.state == "queued"), None)

    def _loop(self):
        while True:
            with self._cv:
                while (job := self._next()) is None:
                    self._cv.wait()
                job.state, job.started = "running", time.time()
                self._save()
            try:
                job.model_id = self._run(job.request, job.id)
                state = "completed"
            except InterruptedError:
                state = "cancelled"
            except Exception as exc:  # the worker must survive any one job's failure
                logger.error("Job %s failed: %s", job.id, exc, exc_info=True)
                job.error, state = str(exc) or type(exc).__name__, "failed"
            with self._cv:
                job.state, job.finished = state, time.time()
                self._save()
            get_progress_manager().forget(job.id)
            logger.info("Job %s %s in %.0fs", job.id, state, job.finished - job.started)
