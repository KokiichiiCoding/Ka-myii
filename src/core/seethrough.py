"""
See-through layer decomposition (Lin et al., SIGGRAPH 2026 — github.com/shitagaki-lab/see-through).

Runs the See-through repo in its own virtualenv (``engines/see-through/.venv``) as a
subprocess: its pinned torch/diffusers stack stays isolated from Ka-myii's, and the
process exiting guarantees every byte of VRAM is returned — which matters on 8 GB GPUs
where the SDXL generator and See-through cannot be resident at the same time.
"""
from __future__ import annotations

import atexit
import logging
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import config

logger = logging.getLogger(__name__)

# Rough share of total runtime per stage, and how many tqdm bars the stage prints
# (LayerDiff3D runs a body pass, then a head pass), used to turn the script's console
# output into one progress bar that never goes backwards.
_STAGES = [
    ("Building LayerDiff3D", 0.00, 1),
    ("Running LayerDiff3D", 0.10, 2),
    ("Building Marigold", 0.80, 1),
    ("Running Marigold", 0.85, 1),
    ("Running PSD assembly", 0.95, 1),
]
_TQDM = re.compile(r"(\d+)/(\d+) \[")
RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "seethrough_runner.py"
# Runners in flight: Windows doesn't kill children with their parent, and an orphaned runner
# would hold the GPU while the next job starts.
_live: set = set()
atexit.register(lambda: [p.kill() for p in list(_live)])


def engine_dir() -> Path:
    return Path(config.ASSET_SEPARATION["seethrough_dir"])


def engine_python() -> Path:
    venv = engine_dir() / ".venv"
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def available() -> bool:
    return engine_python().exists() and (engine_dir() / "inference" / "scripts").exists()


def decompose(
    image_path: Path,
    work_dir: Path,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """Decompose ``image_path`` into a layered PSD. Returns the PSD path.

    The PSD's layers are fully inpainted parts named with See-through's tags
    (``front hair``, ``face``, ``eyewhite``, ``irides``, ``mouth``, ``topwear`` …).
    """
    # shortcut: always the NF4 + block-swap runner (fits 8 GB); 12 GB+ cards could run bf16 faster, add a switch if needed.
    cmd = [
        str(engine_python()),
        "-u",
        str(RUNNER),
        "--srcp", str(Path(image_path).resolve()),
        "--save_dir", str(Path(work_dir).resolve()),
        "--resolution", str(config.ASSET_SEPARATION["seethrough_resolution"]),
        "--tblr_split",  # split eyes / brows / ears / hands into left + right
    ]

    logger.info("See-through: %s", " ".join(cmd))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=engine_dir(),  # the repo's scripts import modules relative to its root
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )

    tail: list[str] = []
    _live.add(proc)
    try:
        _follow(proc.stdout, tail, progress)
    except BaseException:
        # Cancelled (progress raised) or the app is going down: never orphan a GPU job.
        proc.kill()
        proc.wait()
        raise
    finally:
        _live.discard(proc)

    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"See-through failed (exit {code}):\n" + "\n".join(tail[-15:]))

    stem = Path(image_path).stem
    psd = Path(work_dir) / f"{stem}.psd"
    if not psd.exists():
        raise RuntimeError(f"See-through finished but {psd} is missing:\n" + "\n".join(tail[-15:]))
    return psd


def _follow(stream, tail: list, progress: Optional[Callable[[float, str], None]], poll: float = 1.0):
    """Read the runner's output until EOF, keeping the last lines and reporting progress.

    Reads on a helper thread so ``progress`` is still called every ``poll`` seconds while the
    runner is silent (loading weights can take minutes): a cancel raised from it lands promptly.
    """
    chunks: queue.Queue = queue.Queue()

    def pump():
        while True:
            chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
            chunks.put(chunk)
            if not chunk:
                return

    threading.Thread(target=pump, name="seethrough-output", daemon=True).start()
    base, span, label, bars = 0.0, 0.1, "Starting See-through", 1
    done, last, reported = 0, 0.0, 0.0
    buf = b""
    while True:
        try:
            chunk = chunks.get(timeout=poll)
        except queue.Empty:
            if progress:
                progress(min(1.0, reported), label)
            continue
        if not chunk:
            return
        buf += chunk
        # tqdm redraws with \r, normal prints end with \n — treat both as line breaks.
        *lines, buf = re.split(rb"[\r\n]", buf)
        for raw in lines:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            tail[:] = (tail + [line])[-40:]
            for i, (marker, start, n) in enumerate(_STAGES):
                if line.startswith(marker):
                    end = _STAGES[i + 1][1] if i + 1 < len(_STAGES) else 1.0
                    base, span, label, bars, done, last = start, end - start, marker, n, 0, 0.0
            m = _TQDM.search(line)
            if m:
                frac = int(m.group(1)) / max(1, int(m.group(2)))
                if frac < last:  # the stage's next tqdm bar started
                    done += 1
                last = frac
            reported = max(reported, base + span * min(1.0, (done + last) / bars))
            if progress:
                progress(min(1.0, reported), label)
