"""
Region edits: part and painted selections, the /edit API and the pipeline's repaint stage.
GPU-free (the demo generator's inpaint stands in for SDXL).
"""
import base64
import io
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

import config


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app as app_module
    from src.api import generation

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    model = tmp_path / "outputs" / "abc-123"
    (model / "assets").mkdir(parents=True)
    Image.new("RGB", (100, 140), "white").save(model / "base_image.png")
    hair = Image.new("RGBA", (100, 140), (0, 0, 0, 0)); ImageDraw.Draw(hair).rectangle((20, 10, 80, 40), fill=(90, 60, 40, 255))
    hair.save(model / "assets" / "05_front hair.png")
    Image.new("RGBA", (100, 140), (1, 1, 1, 255)).save(model / "assets" / "character.png")  # the cut-out helper
    (model / "metadata.json").write_text(json.dumps({"request": {"prompt": "1girl, brown hair", "seed": 5,
                                                                 "operation": "edit", "edit_prompt": "old"}}))
    flask_app = app_module.create_app(use_dummy_generators=True)
    submitted = []
    monkeypatch.setattr(generation.job_queue, "submit", lambda req: submitted.append(req) or type("J", (), {"id": "j"})())
    monkeypatch.setattr(generation.job_queue, "describe", lambda job: {"id": job.id})
    monkeypatch.setattr("src.core.asset_separator._anime_seg_mask", lambda image: None)
    c = flask_app.test_client()
    c.submitted = submitted
    return c


def _png_url(size, box):
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle(box, fill=(255, 255, 255, 255))
    buf = io.BytesIO(); im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_parts_and_edit_requests(client):
    parts = {p["id"] for p in client.get("/api/generation/parts/abc-123").get_json()["parts"]}
    assert parts == {"hair", "background"}

    r = client.post("/api/generation/edit/abc-123", json={"parts": ["hair"], "prompt": "short blue hair", "strength": 0.8})
    assert r.status_code == 202
    req = client.submitted[-1]
    assert req.operation == "edit" and req.sketch and req.edit_prompt == "short blue hair" and req.strength == 0.8
    assert req.prompt == "1girl, brown hair" and isinstance(req.seed, int) and req.seed != 5  # new seed per edit
    mask = np.asarray(Image.open(req.mask_image)) > 127
    assert mask[25, 50] and not mask[100, 50]  # the hair, grown slightly, nothing else

    # A painted selection (scaled from the canvas size the Studio painted at) also works.
    r = client.post("/api/generation/edit/abc-123", json={"mask": _png_url((50, 70), (5, 45, 20, 60))})
    assert r.status_code == 202
    req = client.submitted[-1]
    mask = np.asarray(Image.open(req.mask_image)) > 127
    assert mask[110, 25] and not mask[25, 50] and req.edit_prompt == "1girl, brown hair"  # empty prompt = retry


@pytest.mark.parametrize("body, word", [
    ({}, "Select a part"), ({"parts": ["wings"]}, "Unknown part"), ({"mask": "data:image/gif;base64,AAAA"}, "PNG"),
    ({"mask": "data:image/png;base64,bm90IGEgcG5n"}, "Could not read"), ({"parts": ["hair"], "strength": 3}, "strength"),
])
def test_edit_rejects_bad_selections(client, body, word):
    r = client.post("/api/generation/edit/abc-123", json=body)
    assert r.status_code == 400 and word in r.get_json()["error"]
    assert client.post("/api/generation/edit/..", json={"parts": ["hair"]}).status_code == 404


def test_edit_job_repaints_only_the_selection(tmp_path):
    from src.models.vtuber_model import GenerationRequest
    from src.pipeline.assembly_line import AssemblyLine

    src = Image.new("RGB", (120, 160), (200, 120, 90))
    ImageDraw.Draw(src).rectangle((0, 0, 119, 40), fill=(30, 30, 160))
    src.save(tmp_path / "source.png")
    mask = Image.new("L", (120, 160), 0); ImageDraw.Draw(mask).rectangle((0, 0, 119, 40), fill=255)
    mask.save(tmp_path / "mask.png")
    line = AssemblyLine(output_base_dir=tmp_path / "out", use_dummy_generators=True)
    model = line.generate_model(GenerationRequest(
        prompt="1girl", operation="edit", sketch=True, source="artwork", seed=3, edit_prompt="red hair",
        reference_image=str(tmp_path / "source.png"), mask_image=str(tmp_path / "mask.png")))
    out = np.asarray(Image.open(tmp_path / "out" / model.id / "base_image.png"), np.int16)
    before = np.asarray(src, np.int16)
    assert model.status.value == "sketch" and model.metadata["source"] == "edit"
    assert np.abs(out[100:] - before[100:]).max() == 0  # far from the selection: untouched
    assert np.abs(out[:30] - before[:30]).max() > 0     # inside: repainted
    assert not (tmp_path / "source.png").exists() and not (tmp_path / "mask.png").exists()
    assert model.request.reference_image is None and model.request.mask_image is None


def test_selections_use_visible_pixels_only():
    """Layers hold hidden, inpainted parts (back hair behind the body); repainting "hair" must
    not paint over the clothes in front of it."""
    from src.core.regions import visible_alphas

    back = np.zeros((10, 10), np.uint8); back[:, :] = 255           # hair behind everything
    body = np.zeros((10, 10), np.uint8); body[5:, 2:8] = 255         # body in front of it
    visible = visible_alphas({"back hair": back, "topwear": body})
    assert visible["back hair"][0, 0] and not visible["back hair"][7, 5] and visible["topwear"][7, 5]


def test_pixels_no_part_predicts_are_released_to_the_leftover():
    """See-through's witch hat was rejected; the hat's top went to the back hair, whose hidden
    region spanned it (black hat vs lavender hair). Such solid regions now belong to no part."""
    from src.core.asset_separator import AssetSeparator

    hair, hat = (200, 190, 220), (20, 20, 30)
    src = np.zeros((60, 60, 3), np.int16); src[:] = hair; src[:25] = hat   # a hat over the top
    back = Image.new("RGBA", (60, 60), hair + (255,))                       # hair predicted everywhere
    face = Image.new("RGBA", (60, 60), (0, 0, 0, 0)); face.paste((240, 200, 180, 255), (20, 40, 40, 58))
    src[40:58, 20:40] = (240, 200, 180)
    covered = AssetSeparator._resolve_ownership([("back hair", back), ("face", face)], src)
    assert not covered[10, 30] and np.asarray(back)[10, 30, 3] == 0      # the hat: nobody's
    assert covered[35, 5] and np.asarray(back)[35, 5, 3] == 255          # real hair stays hair
    assert covered[50, 30]                                                # the face is untouched


def test_outpaint_plan_full_body_and_uncrop():
    from src.core import outpaint

    image = Image.new("RGB", (200, 300), "white")
    fg = np.zeros((300, 200), bool); fg[100:, 40:160] = True     # a torso cut by the bottom border
    canvas, mask, info = outpaint.plan(image, fg, "full_body")
    assert canvas.size == (200, 300) and info["touched"] == ["bottom"]
    left, top, w, h = info["placed"]
    assert w == 110 and h == 165 and mask[top + h // 2, left + w // 2] == False  # the artwork kept...
    assert mask[top + h + 5, 100] and mask[top + h - 2, 100]                     # ...below it and the cut edge painted
    assert mask[5, 5] and not mask[top + 10, left + 10]

    canvas, mask, info = outpaint.plan(image, fg, "uncrop")
    left, top, w, h = info["placed"]
    assert canvas.size == (200, 300) and top == 0 and h < 300 and mask[-1, 100] and not mask[10, 100]


def test_extend_api_queues_an_outpaint(client):
    r = client.post("/api/generation/extend/abc-123", json={"mode": "full_body", "prompt": "red sneakers"})
    assert r.status_code == 202
    req = client.submitted[-1]
    assert req.operation == "extend" and req.sketch and req.strength == 1.0 and req.framing == "full_body"
    assert req.edit_prompt.startswith("red sneakers") and "full body" in req.edit_prompt
    assert "cropped" in req.negative_prompt
    with Image.open(req.reference_image) as canvas:
        assert canvas.size == (100, 140)
    # The test character (a white canvas) touches no border, so there is nothing to uncrop.
    assert client.post("/api/generation/extend/abc-123", json={"mode": "uncrop"}).status_code == 400
    assert client.post("/api/generation/extend/abc-123", json={"mode": "sideways"}).status_code == 400
