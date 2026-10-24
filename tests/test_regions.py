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
    assert w == int(200 * .55 * 5.5 / outpaint.HEADS) and h == int(300 * .55 * 5.5 / outpaint.HEADS) and mask[top + h // 2, left + w // 2] == False  # the artwork kept...
    assert mask[top + h + 5, 100] and mask[top + h - 2, 100]                     # ...below it and the cut edge painted
    assert mask[5, 5] and not mask[top + 10, left + 10]

    canvas, mask, info = outpaint.plan(image, fg, "uncrop")
    left, top, w, h = info["placed"]
    assert canvas.size == (200, 300) and top == 0 and h < 300 and mask[-1, 100] and not mask[10, 100]

    # A known head size leaves room for a ~5.5-head figure: a big bust-up head shrinks further.
    _, _, info = outpaint.plan(image, fg, "full_body", head=150)
    assert info["placed"][3] == int(300 * 0.9 * 300 / (150 * outpaint.HEADS))
    assert outpaint.full_body_prompt("1girl, Upper Body, centered, cowboy shot, red hoodie") == "1girl, centered, red hoodie"
    # A graded mask: the cut edges are redrawn fully at the edge, less further in, and not deep inside;
    # a brim cut at the left side reaches further in than the bottom join.
    fg[:, :3] = True
    _, mask, info = outpaint.plan(image, fg, "full_body")
    left, top, w, h = info["placed"]
    row, col = mask[top + h // 2], mask[:, 100]
    assert row[left + 4] == 255 and 0 < row[left + 12] < 255 and row[left + 25] == 0
    assert col[top + h - 3] == 255 and 0 < col[top + h - 15] < 255 and col[top + h - 35] == 0
    assert 0 < row[left + w - 2] < 255 and row[left + w - 5] == 0   # an uncut edge only blends a few px


def test_extend_api_queues_an_outpaint(client):
    r = client.post("/api/generation/extend/abc-123", json={"mode": "full_body", "prompt": "red sneakers"})
    assert r.status_code == 202
    req = client.submitted[-1]
    assert req.operation == "extend" and req.sketch and req.strength == 0.9 and req.framing == "full_body"
    assert "brown hair" in req.edit_prompt and "red sneakers" in req.edit_prompt and "full body" in req.edit_prompt
    assert "cropped" in req.negative_prompt
    with Image.open(req.reference_image) as canvas:
        assert canvas.size == (100, 140)
    # A separated character's face sets the scale, and its brief loses "upper body".
    model = config.OUTPUT_DIR / "abc-123"
    face = Image.new("RGBA", (100, 140)); ImageDraw.Draw(face).rectangle((30, 20, 70, 59), fill="tan")
    face.save(model / "assets" / "07_face.png")
    (model / "metadata.json").write_text(json.dumps({"request": {"prompt": "1girl, upper body, red hoodie"}}))
    r = client.post("/api/generation/extend/abc-123", json={"mode": "full_body"})
    req, info = client.submitted[-1], r.get_json()["info"]
    # Crown (hair from 10 px) to chin (60 px) as drawn, then 5.75 typical heads for this 41 px wide face.
    assert info["placed"][3] == int(140 * (140 * 0.9) / (50 + 5.75 * 1.33 * 41))
    assert req.prompt == "1girl, red hoodie" and "upper body" not in req.edit_prompt and "arms at sides" in req.edit_prompt
    # The test character (a white canvas) touches no border, so there is nothing to uncrop.
    assert client.post("/api/generation/extend/abc-123", json={"mode": "uncrop"}).status_code == 400
    assert client.post("/api/generation/extend/abc-123", json={"mode": "sideways"}).status_code == 400


def test_extend_can_continue_to_layers_and_rejects_ambiguous_flags(client):
    r = client.post("/api/generation/extend/abc-123", json={"mode": "full_body", "separate": True, "prompt": "black boots"})
    assert r.status_code == 202
    req = client.submitted[-1]
    assert req.operation == "extend" and not req.sketch
    assert "brown hair" in req.prompt and "black boots" in req.prompt
    count = len(client.submitted)
    assert client.post("/api/generation/extend/abc-123", json={"separate": "false"}).status_code == 400
    assert len(client.submitted) == count


def test_extension_crown_axis_and_face_protection(tmp_path):
    from src.core import outpaint

    assets = tmp_path / "assets"; assets.mkdir()
    face = Image.new("RGBA", (400, 600)); ImageDraw.Draw(face).rectangle((145, 70, 264, 209), fill="tan")
    face.save(assets / "03_face.png")
    hair = Image.new("RGBA", face.size)
    draw = ImageDraw.Draw(hair); draw.rectangle((125, 30, 280, 90), fill="silver")
    draw.line((200,0,200,30),fill="silver",width=2)  # sparse ahoge is not the skull crown
    hair.save(assets / "02_front hair.png")
    neck=Image.new("RGBA",face.size);ImageDraw.Draw(neck).rectangle((195,190,229,270),fill="tan")
    neck.save(assets / "04_neck.png")
    geometry=outpaint.anatomy(tmp_path)
    assert geometry["crown_y"] == 30 and geometry["head_height"] == 180
    assert geometry["axis_x"] == 212
    image=Image.new("RGB",face.size,"white");fg=np.zeros((600,400),bool);fg[25:,110:290]=True
    _, mask, info=outpaint.plan(image,fg,geometry=geometry)
    x,y,w,h=info["placed"]; result=info["result_geometry"]
    assert abs(result["axis_x"]-200)<1 and result["head_height"] < 90
    assert mask[round(y+140*h/600),round(x+205*w/400)] == 0, "Keep the inner face even beneath edge ramps"
    assert mask[y+h-1,x+w//2] == 255 and mask[y+h-20,x+w//2] > 0
    neck = lambda k: mask[round(y + (210 + k * 180) * h / 600), round(x + 212 * w / 400)]
    assert neck(0.17) == 255 and 0 < neck(0.06) < 255, "The neck is redrawn from the jaw down (no kept collar plate)"
    _, _, compact=outpaint.plan(image,fg,geometry=geometry,proportions="compact")
    _, _, tall=outpaint.plan(image,fg,geometry=geometry,proportions="tall")
    assert tall["placed"][2]<w<compact["placed"][2]
    _,_,hat=outpaint.plan(image,fg,geometry={**geometry,"large_headwear":True})
    assert hat["placed"][1]>y and hat["placed"][2]<w, "Leave crown room for a large hat without enlarging the head"
    assert hat["pose_points"][10][1] < hat["foot_y"]-hat["result_geometry"]["unit"]*.4, "Room for shoes below the ankles"
    # Precise geometry survives artwork-first output, without a second separation.
    (tmp_path/"metadata.json").write_text(json.dumps({"request":{"extension":info}}))
    (assets/"03_face.png").unlink()
    assert outpaint.anatomy(tmp_path) == result


def test_extension_plan_is_read_only_and_options_are_validated(client):
    model=config.OUTPUT_DIR/"abc-123"
    before=(model/"base_image.png").read_bytes()
    response=client.get("/api/generation/extend/abc-123/plan?proportions=tall")
    assert response.status_code==200 and response.get_json()["info"]["target_heads"]==7.5
    assert not (config.OUTPUT_DIR/"_uploads").exists() and client.submitted==[]
    # A project file must not smuggle an arbitrary local pose-image path into a region edit.
    data=json.loads((model/"metadata.json").read_text());data["request"]["pose_image"]="C:/private.png"
    (model/"metadata.json").write_text(json.dumps(data))
    assert client.post("/api/generation/edit/abc-123",json={"parts":["hair"]}).status_code==202
    assert client.submitted[-1].pose_image is None
    client.submitted.clear()
    for body in ({"proportions":"tiny"},{"proportions":[]},{"keep_face":"false"}):
        assert client.post("/api/generation/extend/abc-123",json=body).status_code==400
    assert client.get("/api/generation/extend/abc-123/plan?proportions=giant").status_code==400
    assert client.get("/api/generation/extend/missing/plan").status_code==404
    response=client.post("/api/generation/extend/abc-123",json={"proportions":"tall","keep_face":False})
    assert response.status_code==202
    req=client.submitted[-1]
    assert req.extension["proportions"]=="tall" and req.extension["source_model"]=="abc-123"
    assert not req.extension["keep_face"] and "oversized head" in req.negative_prompt
    assert (model/"base_image.png").read_bytes()==before


def test_dragged_pose_guides_move_mirrored_joints_and_are_validated(client):
    from src.core import outpaint

    info = {"canvas": [832, 1216], "target_heads": 6.75, "foot_y": 1143,
            "result_geometry": {"face": [360, 80, 468, 210], "head_height": 162, "axis_x": 416, "crown_y": 48}}
    moved = outpaint.pose_points({**info, "guides": {"shoulders": [120, 300], "hips": [70, 600], "ankles": [40, 1050]}})
    assert moved[2] == (296, 300) and moved[5] == (536, 300), "Shoulders mirror about the standing axis"
    assert moved[8][1] == moved[11][1] == 600 and moved[10] == (376, 1050) and moved[13] == (456, 1050)
    assert 600 < moved[9][1] < 1050 and moved[9][0] == 416 - 55, "Knees sit between the hips and ankles"
    assert moved[4][0] < moved[2][0], "Wrists hang outside wider shoulders"
    for bad in ({"neck": [10, 10]}, {"hips": [10]}, {"hips": [True, 5]}, {"hips": [900, 50]}, "x",
                {"shoulders": [80, 700], "hips": [60, 500]}):
        with pytest.raises(ValueError):
            outpaint.check_guides(bad, (832, 1216))
    assert outpaint.check_guides({}, (832, 1216)) is None

    r = client.get("/api/generation/extend/abc-123/plan", query_string={"guides": json.dumps({"hips": [20, 90]})})
    assert r.status_code == 200
    points = r.get_json()["info"]["pose_points"]
    assert points[8][1] == points[11][1] == 90 and abs((points[11][0] - points[8][0]) / 2 - 20) < 1e-6
    assert client.get("/api/generation/extend/abc-123/plan?guides={oops").status_code == 400
    assert client.post("/api/generation/extend/abc-123", json={"guides": {"hips": [20, 9999]}}).status_code == 400
    assert client.post("/api/generation/extend/abc-123", json={"guides": {"ankles": [20, 120]}}).status_code == 202
    assert client.submitted[-1].extension["guides"] == {"ankles": [20.0, 120.0]}


def test_full_body_extension_reviews_its_result(tmp_path, monkeypatch):
    from src.models.vtuber_model import GenerationRequest
    from src.pipeline.assembly_line import AssemblyLine

    monkeypatch.setattr("src.core.asset_separator._anime_seg_mask", lambda image: None)
    src = Image.new("RGB", (120, 160), "white")
    ImageDraw.Draw(src).rectangle((40, 20, 80, 159), fill=(60, 80, 160))  # a figure cut off at the bottom
    src.save(tmp_path / "source.png")
    Image.new("L", (120, 160), 0).save(tmp_path / "mask.png")
    line = AssemblyLine(output_base_dir=tmp_path / "out", use_dummy_generators=True)
    model = line.generate_model(GenerationRequest(
        prompt="1girl", operation="extend", sketch=True, source="artwork", seed=3, edit_prompt="full body",
        extension={"mode": "full_body"}, reference_image=str(tmp_path / "source.png"), mask_image=str(tmp_path / "mask.png")))
    assert [f["id"] for f in model.metadata["checks"]] == ["feet"]


def test_refine_finds_hands_and_face_and_redraws_only_there(client, tmp_path):
    from src.core import refine
    from src.models.vtuber_model import GenerationRequest
    from src.pipeline.assembly_line import AssemblyLine

    model = config.OUTPUT_DIR / "abc-123"
    assert refine.boxes(model, "hands") == [] and client.post(
        "/api/generation/refine/abc-123", json={"target": "hands"}).status_code == 400, "No face layer: nothing to place"
    face = Image.new("RGBA", (100, 140)); ImageDraw.Draw(face).ellipse((35, 15, 65, 50), fill=(240, 210, 190, 255))
    face.save(model / "assets" / "06_face.png")
    hand = Image.new("RGBA", (100, 140)); ImageDraw.Draw(hand).rectangle((10, 60, 18, 120), fill=(240, 210, 190, 255))
    hand.save(model / "assets" / "03_handwear-l.png")
    [box] = refine.boxes(model, "hands")
    assert box[0] <= 14 <= box[2] and box[1] < 112 < box[3] and box[3] - box[1] <= 30, "Around the end of the arm"
    [fbox] = refine.boxes(model, "face")
    assert fbox[0] < 35 and fbox[2] > 65
    for body in ({"target": "feet"}, {"target": "face", "strength": 0.95}):
        assert client.post("/api/generation/refine/abc-123", json=body).status_code == 400
    r = client.post("/api/generation/refine/abc-123", json={"target": "hands"})
    assert r.status_code == 202
    req = client.submitted[-1]
    assert req.operation == "refine" and req.regions == [box] and req.sketch and "five fingers" in req.edit_prompt
    assert "1girl" not in req.edit_prompt and req.strength == refine.STRENGTH["hands"]

    src = Image.new("RGB", (100, 140), (200, 120, 90)); src.save(tmp_path / "src.png")
    line = AssemblyLine(output_base_dir=tmp_path / "out", use_dummy_generators=True)
    done = line.generate_model(GenerationRequest(prompt="1girl", operation="refine", sketch=True, source="artwork",
                                                 seed=1, regions=[[5, 5, 25, 25], [70, 100, 90, 120]],
                                                 reference_image=str(tmp_path / "src.png")))
    out = np.asarray(Image.open(tmp_path / "out" / done.id / "base_image.png"), np.int16)
    assert done.status.value == "sketch" and not (tmp_path / "src.png").exists()
    assert (out[60:80, 40:60] == (200, 120, 90)).all(), "Between: untouched"
    assert (out[15, 15] != (200, 120, 90)).any() and (out[110, 80] != (200, 120, 90)).any(), "Both boxes redrawn"
