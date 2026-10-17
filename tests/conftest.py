import pytest


@pytest.fixture(autouse=True)
def _colour_key_cutouts(monkeypatch):
    """Tests use synthetic images: keep cut-outs deterministic and independent of whether the
    176 MB anime-segmentation model happens to be in the local Hugging Face cache."""
    from src.core import asset_separator

    monkeypatch.setattr(asset_separator, "_anime_seg_mask", lambda image: None)
