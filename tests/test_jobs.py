"""Background job queue, cancellation and monotonic See-through progress (GPU-free)."""
import io
import threading
import time

import pytest

from src.models.vtuber_model import GenerationRequest
from src.pipeline.jobs import JobQueue
from src.utils.progress_tracker import get_progress_manager


def _wait(job, states=("completed", "failed", "cancelled"), timeout=10):
    end = time.time() + timeout
    while job.state not in states:
        assert time.time() < end, f"job stuck in {job.state}"
        time.sleep(0.01)


def test_jobs_run_in_order_and_queued_jobs_cancel():
    gate, order = threading.Event(), []

    def run(request, job_id):
        gate.wait(5)
        order.append(request.prompt)
        if request.prompt == "boom":
            raise RuntimeError("exploded")
        return f"model-{request.prompt}"

    q = JobQueue(run)
    a, b, c, d = (q.submit(GenerationRequest(prompt=p)) for p in ("a", "b", "boom", "d"))
    _wait(a, ("running",))
    assert q.describe(c)["position"] == 1  # b is ahead of it
    q.cancel(b.id)
    assert b.state == "cancelled"
    gate.set()
    for job in (a, c, d):
        _wait(job)
    assert order == ["a", "boom", "d"]
    assert (a.state, a.model_id) == ("completed", "model-a")
    assert (c.state, c.error) == ("failed", "exploded")
    assert q.describe(d)["progress"] == 100.0


def test_cancel_running_job_stops_assembly_line_and_removes_its_folder(tmp_path):
    from src.pipeline.assembly_line import AssemblyLine

    line = AssemblyLine(tmp_path, use_dummy_generators=True)
    started = threading.Event()
    real_generate = line.image_generator.generate

    def slow_generate(request, path, task_id=None):
        out = real_generate(request, path, task_id=task_id)
        started.set()
        while not get_progress_manager().is_cancelled(task_id):
            time.sleep(0.01)
        return out

    line.image_generator.generate = slow_generate
    q = JobQueue(lambda request, job_id: line.generate_model(request, task_id=job_id).id)
    job = q.submit(GenerationRequest(prompt="1girl", width=256, height=256, steps=2))
    assert started.wait(10)
    running = q.describe(job)
    assert running["stage"] == "image" and 0 < running["progress"] <= 15
    q.cancel(job.id)
    _wait(job)
    assert job.state == "cancelled"
    assert list(tmp_path.iterdir()) == []  # the half-built model folder is gone
    assert not get_progress_manager().is_cancelled(job.id)  # forgotten once finished


def test_failed_job_keeps_its_artwork_listed_and_drops_empty_folders(tmp_path):
    import json
    from src.pipeline.assembly_line import AssemblyLine

    line = AssemblyLine(tmp_path, use_dummy_generators=True)

    def out_of_memory(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    line.asset_separator.separate = out_of_memory
    with pytest.raises(RuntimeError):
        line.generate_model(GenerationRequest(prompt="1girl", width=256, height=256, steps=2))
    (folder,) = tmp_path.iterdir()  # the artwork survives, with metadata the Gallery can list
    saved = json.loads((folder / "metadata.json").read_text())
    assert saved["status"] == "failed" and "out of memory" in saved["error_message"]
    assert (folder / "base_image.png").exists()

    line.image_generator.generate = out_of_memory  # nothing worth keeping yet
    with pytest.raises(RuntimeError):
        line.generate_model(GenerationRequest(prompt="1girl", width=256, height=256, steps=2))
    assert list(tmp_path.iterdir()) == [folder]


def test_seethrough_progress_is_monotonic_across_body_and_head_passes():
    from src.core import seethrough

    out = b"".join([
        b"Building LayerDiff3D...\n",
        b"Running LayerDiff3D (body + head)...\n",
        *(f"{i}/30 [00:01<00:10]\r".encode() for i in range(31)),  # body pass
        *(f"{i}/30 [00:01<00:10]\r".encode() for i in range(31)),  # head pass restarts at 0
        b"Building Marigold...\nRunning Marigold...\nRunning PSD assembly...\nDone\n",
    ])
    seen, tail = [], []
    seethrough._follow(io.BytesIO(out), tail, lambda frac, label: seen.append(frac))
    assert seen == sorted(seen), "progress went backwards"
    middle = seen[2 + 31]  # end of the body pass ≈ halfway through the LayerDiff3D stage
    assert abs(middle - (0.10 + 0.70 * 0.5)) < 0.01
    assert seen[-1] >= 0.95 and tail[-1] == "Done"


@pytest.mark.parametrize("chatty", [True, False], ids=["printing", "silent-while-loading"])
def test_cancel_kills_the_seethrough_subprocess(tmp_path, monkeypatch, chatty):
    from src.core import seethrough
    import subprocess
    import sys

    # A stand-in runner that runs until killed: printing progress, or silent like See-through
    # while it loads weights (a cancel must not wait for its next output line).
    script = tmp_path / "forever.py"
    body = "print('1/30 [', flush=True); time.sleep(0.05)" if chatty else "time.sleep(1)"
    script.write_text(f"import time\nprint('Building LayerDiff3D', flush=True)\nwhile True:\n    {body}\n")
    procs = []
    real_popen = subprocess.Popen

    def popen(cmd, **kw):
        p = real_popen([sys.executable, "-u", str(script)], stdout=kw["stdout"], stderr=kw["stderr"])
        procs.append(p)
        return p

    monkeypatch.setattr(seethrough.subprocess, "Popen", popen)
    calls = []

    def progress(frac, label):
        calls.append(frac)
        if len(calls) > 3:
            raise InterruptedError("cancelled")

    started = time.time()
    with pytest.raises(InterruptedError):
        seethrough.decompose(tmp_path / "x.png", tmp_path, progress)
    assert time.time() - started < 8
    assert procs and procs[0].poll() is not None, "runner still alive after cancel"
