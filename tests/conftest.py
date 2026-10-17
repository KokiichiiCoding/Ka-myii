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



@pytest.fixture(autouse=True)
def _private_outputs(tmp_path, monkeypatch):
    """Never touch the real outputs/ folder (the owner's characters, the persisted job queue)."""
    import config

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
