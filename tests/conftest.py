import pytest


@pytest.fixture(autouse=True)
def _colour_key_cutouts(monkeypatch):
    """Tests use synthetic images: keep cut-outs deterministic and independent of whether the
    176 MB anime-segmentation model happens to be in the local Hugging Face cache."""
    from src.core import asset_separator

    monkeypatch.setattr(asset_separator, "_anime_seg_mask", lambda image: None)
    # ...and never start the real See-through engine (GPU, ~10 min) when it is installed;
    # tests that exercise its post-processing monkeypatch `available` back on with a fake PSD.
    from src.core import seethrough

    monkeypatch.setattr(seethrough, "available", lambda: False)
    # Likewise the 3D engine (a private ComfyUI) and the DWPose weights in the local HF cache.
    from src.core import mesh3d, pose

    monkeypatch.setattr(mesh3d, "available", lambda: False)
    monkeypatch.setattr(pose, "model_path", lambda download=False: None)



@pytest.fixture(autouse=True)
def _private_outputs(tmp_path, monkeypatch):
    """Never touch the real outputs/ folder (the owner's characters, the persisted job queue)."""
    import sys
    import time
    import config

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    yield
    # A job a test queued on the app's shared queue can outlive the test; once the patch above is
    # undone it would write into the real outputs/ (seen twice), so let it finish first.
    generation = sys.modules.get("src.api.generation")
    queue = getattr(generation, "job_queue", None)
    deadline = time.time() + 60
    while queue is not None and time.time() < deadline and \
            any(j.state in ("queued", "running") for j in queue.jobs()):
        time.sleep(.1)
