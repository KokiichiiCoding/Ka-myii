"""
Background generation jobs.

A real run takes ~15–20 minutes and needs the whole 8 GB GPU, so ``/generate`` only queues
the request: one worker thread runs jobs strictly one at a time, and the browser polls
``/api/generation/jobs/<id>`` for state and progress (so a reloaded tab can re-attach).
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.models.vtuber_model import GenerationRequest
from src.utils.progress_tracker import get_progress_manager

logger = logging.getLogger(__name__)

# Share of a real run per stage, from measured timings (~2 min paint, ~10–12 min decompose,
# seconds to package) — turns per-stage trackers into one bar that never goes backwards.
STAGES = {
    "image": (0.0, 0.15, "Painting the character…"),
    "separate": (0.15, 0.97, "Separating layers…"),
    "assemble": (0.97, 1.0, "Packaging PSD + Cubism files…"),
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


class JobQueue:
    """FIFO of generation requests executed by a single daemon worker thread.

    shortcut: jobs live in memory — a server restart forgets queued/running jobs (finished
    models are still on disk); persist the queue if unattended overnight batches matter.
    """

    def __init__(self, run: Callable[[GenerationRequest, str], str]):
        # run(request, job_id) executes one job and returns the finished model id.
        self._run = run
        self._jobs: Dict[str, Job] = {}
        self._cv = threading.Condition()
        threading.Thread(target=self._loop, name="kamyii-jobs", daemon=True).start()

    def submit(self, request: GenerationRequest) -> Job:
        job = Job(id=f"gen_{uuid.uuid4().hex[:8]}", request=request)
        with self._cv:
            self._jobs[job.id] = job
            self._prune()
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
            lo, hi, label = STAGES.get(stage, (0.0, 1.0, ""))
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
            get_progress_manager().forget(job.id)
            logger.info("Job %s %s in %.0fs", job.id, state, job.finished - job.started)
