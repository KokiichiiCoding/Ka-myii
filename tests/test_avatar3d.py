"""
3D avatar path without a GPU: framing, the Pixal3D graph, the camera mapping, skinning, the
T-pose, expression decals, the VRM writer, and the API / demo job.
"""
import json
import struct

import numpy as np
import pytest
from PIL import Image

import config
from src.core import mesh3d, rig3d

CAM = {"fov": 40.0, "box": (10.0, 20.0, 500.0), "paint": {
    "score": 0.95, "azimuth": 0.0, "flipped": False,
    "mesh_moments": [0.5, 0.48, 0.45], "photo_moments": [0.51, 0.47, 0.46]}}


def test_frame_crops_a_padded_square_around_the_character():
    image = Image.new("RGB", (200, 300), (200, 30, 30))
    mask = np.zeros((300, 200), bool)
    mask[100:250, 50:150] = True
    rgb, alpha, (x0, y0, side) = mesh3d.frame(image, mask, size=256)
    assert side == pytest.approx(165) and (x0, y0) == pytest.approx((17.5, 92.5))
    assert rgb.size == alpha.size == (256, 256)
    assert rgb.getpixel((128, 128)) == (200, 30, 30)   # the character, where the box says
    assert rgb.getpixel((2, 128)) == (0, 0, 0)          # background is black, as Pixal3D expects
    with pytest.raises(ValueError):
        mesh3d.frame(image, np.zeros((300, 200), bool))


def test_frame_does_not_bleed_background_into_resampled_subject():
    mask = np.zeros((31, 25), bool)
    mask[3:28, 7:18] = True
    dark = np.zeros((31, 25, 3), np.uint8)
    light = np.full((31, 25, 3), 255, np.uint8)
    dark[mask] = light[mask] = (60, 30, 120)
    a, alpha_a, box_a = mesh3d.frame(Image.fromarray(dark), mask, size=43)
    b, alpha_b, box_b = mesh3d.frame(Image.fromarray(light), mask, size=43)
    assert box_a == box_b
    np.testing.assert_array_equal(alpha_a, alpha_b)
    np.testing.assert_array_equal(a, b)


def test_graph_is_wired_and_reports_its_outputs():
    nodes, ids = mesh3d.graph("kamyii/a.png", "kamyii/a_mask.png", seed=5, quality="draft", faces=50_000)
    assert sum(n["class_type"] == "KSampler" for n in nodes.values()) == 4
    assert nodes[ids["model"]]["class_type"] == "SaveGLB"
    assert nodes[ids["fov"]]["class_type"] == "PreviewAny"
    assert nodes[ids["paint"]]["class_type"] == "KamyiiProjectImage"
    for node in nodes.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in nodes   # every link points at a node in the graph
    assert any(n["inputs"].get("target_face_count") == 50_000 for n in nodes.values())
    with pytest.raises(ValueError):
        mesh3d.graph("a", "b", seed=1, quality="ultra")
    node_src = (config.BASE_DIR / "scripts" / "comfy3d_nodes" / "kamyii3d" / "__init__.py").read_text(encoding="utf-8")
    assert 'node_id="KamyiiProjectImage"' in node_src


def test_engine_errors_read_well():
    class R:
        status_code = 400

        def json(self):
            return {"error": {"message": "Prompt outputs failed validation"},
                    "node_errors": {"7": {"class_type": "UNETLoader", "errors": [{"message": "Value not in list"}]}}}
    assert "UNETLoader: Value not in list" in mesh3d.describe_rejection(R())
    oom = {"messages": [["execution_error", {"node_type": "KSampler", "exception_message": "CUDA out of memory."}]]}
    assert "ran out of memory" in mesh3d.execution_error(oom)


def test_camera_rays_and_projection_agree():
    cam = rig3d.Camera(CAM)
    for uv in [(260.0, 300.0), (40.0, 480.0), (500.0, 60.0)]:
        o, d = cam.ray(uv)
        assert cam.to_image((o + d * 1.3)[None])[0] == pytest.approx(uv, abs=1e-6)
    for bad in ({"score": 0.5}, {"flipped": True}, {"azimuth": 180.0}):
        with pytest.raises(ValueError):
            rig3d.Camera({**CAM, "paint": {**CAM["paint"], **bad}})


def _box(a, b, half):
    """8 corners and 12 outward triangles of a box around segment a-b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = (b - a) / np.linalg.norm(b - a)
    u = np.cross(axis, [0, 0, 1]) if abs(axis[2]) < .9 else np.cross(axis, [1, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    ring = [s * u * half + r * v * half for s, r in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    pts = np.array([a + o for o in ring] + [b + o for o in ring])
    quads = [[k, (k + 1) % 4, 4 + (k + 1) % 4, 4 + k] for k in range(4)] + [[3, 2, 1, 0], [4, 5, 6, 7]]
    tri = np.concatenate([np.array(q)[[[0, 1, 2], [0, 2, 3]]] for q in quads])
    return pts, tri


def _figure():
    """An A-pose stick figure in Pixal3D's frame (faces +Z, its left is +X) with its joints."""
    j = {"hips": (0, -.08, 0), "spine": (0, .02, 0), "chest": (0, .13, 0), "neck": (0, .24, 0),
         "head": (0, .27, 0), "headTop": (0, .42, 0)}
    for side, x in (("left", 1), ("right", -1)):
        j.update({f"{side}Shoulder": (x * .03, .235, 0), f"{side}UpperArm": (x * .1, .23, 0),
                  f"{side}LowerArm": (x * .13, .09, 0), f"{side}Hand": (x * .16, -.05, 0),
                  f"{side}HandTip": (x * .17, -.09, 0), f"{side}UpperLeg": (x * .05, -.1, 0),
                  f"{side}LowerLeg": (x * .05, -.27, 0), f"{side}Foot": (x * .05, -.44, 0),
                  f"{side}Toes": (x * .05, -.46, .04)})
    j = {k: np.array(v, float) for k, v in j.items()}
    parts = {"torso": (j["hips"], j["neck"], .06), "head": (j["head"] + [0, .01, 0], j["headTop"], .06)}
    for side in ("left", "right"):
        parts[f"{side}Arm1"] = (j[f"{side}UpperArm"], j[f"{side}LowerArm"], .014)
        parts[f"{side}Arm2"] = (j[f"{side}LowerArm"], j[f"{side}Hand"], .012)
        parts[f"{side}Leg"] = (j[f"{side}UpperLeg"] - [0, .03, 0], j[f"{side}Foot"], .02)
    pos, faces, which = [], [], []
    for name, (a, b, half) in parts.items():
        p, t = _box(a, b, half)
        faces.append(t + len(pos))
        pos += list(p)
        which += [name] * len(p)
    radii = dict(UpperArm=.022, LowerArm=.02, Hand=.015, HandTip=.02, UpperLeg=.03, LowerLeg=.025, Foot=.02, Toes=.02)
    return np.array(pos), np.concatenate(faces), np.array(which), j, radii


def test_skinning_and_t_pose_move_only_the_arms():
    pos, faces, which, j, radii = _figure()
    w, keep = rig3d.skin_weights(pos, faces, j, radii)
    assert np.allclose(w.sum(1), 1) and keep.all()
    arm_bones = {rig3d.INDEX[f"left{b}"] for b in ("UpperArm", "LowerArm", "Hand")}
    assert all(b in arm_bones for b in w[which == "leftArm2"].argmax(1))
    assert (w[which == "head", rig3d.INDEX["head"]] > .99).all()
    new, nrm, jt = rig3d.t_pose(pos, rig3d.vertex_normals(pos, faces), w, j)
    shoulder = j["leftUpperArm"]
    assert jt["leftHand"][1] == pytest.approx(shoulder[1]) and jt["leftHand"][0] > shoulder[0] + .2
    arm = new[which == "leftArm2"]
    assert np.abs(arm[:, 1] - shoulder[1]).max() < .03 and arm[:, 0].min() > shoulder[0]
    assert np.allclose(new[which == "torso"], pos[which == "torso"])
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1)


def test_parts_spread_over_the_surface_and_cut_fused_hands():
    pos, faces, which, j, radii = _figure()
    labels = np.full(len(pos), "", object)
    labels[np.nonzero(which == "leftArm2")[0][0]] = "handwear-l"
    labels[np.nonzero(which == "torso")[0][0]] = "topwear"
    parts = rig3d.spread_parts(faces, pos, labels)
    assert (parts[which == "leftArm2"] == "arm-l").all() and (parts[which == "torso"] == "body").all()
    assert (parts[which == "head"] == "").all()   # an unlabelled separate piece stays unknown
    # fuse the hand to the torso with one triangle: the T-pose must cut it, not stretch it
    hand = np.nonzero(which == "leftArm2")[0][-1]
    body = np.nonzero(which == "torso")[0][:2]
    fused = np.concatenate([faces, [[hand, body[0], body[1]]]])
    labels[np.nonzero(which == "leftArm1")[0][0]] = "handwear-l"
    _, keep = rig3d.skin_weights(pos, fused, j, radii, rig3d.spread_parts(fused, pos, labels))
    kept = fused[keep]
    assert not (kept == [hand, body[0], body[1]]).all(1).any()
    for piece in ("leftArm2", "torso"):   # the forearm and the body themselves stay whole
        assert ((which[kept] == piece).all(1)).sum() == ((which[faces] == piece).all(1)).sum()


def _model_dir(tmp_path, size=(512, 540)):
    d = tmp_path / "char"
    (d / "expressions").mkdir(parents=True)
    Image.new("RGB", size, (240, 200, 180)).save(d / "base_image.png")
    eyes = np.zeros((size[1], size[0], 4), np.uint8)
    eyes[180:200, 200:240] = eyes[180:200, 280:320] = (90, 40, 40, 255)
    mouth = np.zeros_like(eyes)
    mouth[250:265, 245:275] = (150, 30, 40, 255)
    blush = np.zeros_like(eyes)
    blush[215:230, 190:230] = (240, 120, 130, 160)
    above = np.zeros_like(eyes)
    above[20:40, 240:260] = (0, 0, 0, 255)   # a "?" floating over the head: no surface there
    Image.fromarray(eyes).save(d / "expressions" / "eye_close.png")
    Image.fromarray(mouth).save(d / "expressions" / "mouth_open.png")
    Image.fromarray(mouth).save(d / "expressions" / "mouth_o.png")
    Image.fromarray(blush).save(d / "expressions" / "toggle_blush.png")
    Image.fromarray(above).save(d / "expressions" / "toggle_question.png")
    (d / "expressions" / "expressions.json").write_text(json.dumps(
        {"eye_close": "expressions/eye_close.png", "mouth_open": "expressions/mouth_open.png",
         "visemes": {"o": "expressions/mouth_o.png"},
         "toggles": {"blush": "expressions/toggle_blush.png", "question": "expressions/toggle_question.png"}}))
    return d


def _face_sheet(cam, x1=150, x2=370, y1=120, y2=320, n=24):
    """A flat grid facing the camera covering an artwork rectangle."""
    xs, ys = np.meshgrid(np.linspace(x1, x2, n), np.linspace(y1, y2, n))
    pos = []
    for uv in zip(xs.ravel(), ys.ravel()):
        o, d = cam.ray(uv)
        pos.append(o + d * (o[2] - 0.1) / -d[2])   # the plane z = 0.1
    k = np.arange(n * n).reshape(n, n)
    a, b, c, e = k[:-1, :-1].ravel(), k[:-1, 1:].ravel(), k[1:, 1:].ravel(), k[1:, :-1].ravel()
    faces = np.concatenate([np.stack([a, e, c], 1), np.stack([a, c, b], 1)])
    return np.array(pos), faces


def test_decals_cover_the_painted_eyes_and_mouth(tmp_path):
    cam = rig3d.Camera(CAM)
    d = _model_dir(tmp_path)
    pos, faces = _face_sheet(cam)
    nrm = np.tile([0.0, 0.0, 1.0], (len(pos), 1))
    kp = np.zeros((133, 3))
    kp[1] = (300, 190, .9)   # the character's left eye is on the picture's right
    kp[2] = (220, 190, .9)
    out = {e["name"]: e for e in rig3d.decals(d, cam, pos, nrm, faces, kp)}
    assert set(out) == {"blink_l", "blink_r", "a", "o", "toggle:blush"} and not any(e["shown"] for e in out.values())
    assert out["toggle:blush"]["label"] == "Blush"
    left = np.asarray(out["blink_l"]["image"])[..., 3]
    assert left.any() and out["blink_l"]["image"].width < 200   # only the left eye's half
    for e in out.values():
        used = np.unique(e["faces"])
        assert len(used) and (e["uv"][used] >= -0.05).all() and (e["uv"][used] <= 1.05).all()


def test_every_avatar_gets_drawn_face_marks_unless_painted(tmp_path):
    cam = rig3d.Camera(CAM)
    d = _model_dir(tmp_path)
    (d / "assets").mkdir()
    layer = np.zeros((540, 512, 4), np.uint8)
    face = layer.copy()
    face[110:320, 160:360] = (240, 200, 180, 255)
    Image.fromarray(face).save(d / "assets" / "01_face.png")
    for side, x in (("l", 200), ("r", 280)):
        eye = layer.copy()
        eye[180:200, x:x + 40] = (255, 255, 255, 255)
        Image.fromarray(eye).save(d / "assets" / f"02_eyewhite-{side}.png")
    pos, faces = _face_sheet(cam, y1=60)
    nrm = np.tile([0.0, 0.0, 1.0], (len(pos), 1))
    kp = np.zeros((133, 3))
    out = {e["name"]: e for e in rig3d.decals(d, cam, pos, nrm, faces, kp)}
    assert {"toggle:blush", "toggle:gloom", "toggle:anger", "toggle:sweat"} <= set(out)
    painted = np.asarray(out["toggle:blush"]["image"])
    assert (painted[painted[..., 3] > 0][:, :3] == (240, 120, 130)).all()   # the painted blush stays
    made = json.loads((d / "expressions" / "expressions.json").read_text())
    del made["toggles"]["blush"]
    (d / "expressions" / "expressions.json").write_text(json.dumps(made))
    drawn = np.asarray({e["name"]: e for e in rig3d.decals(d, cam, pos, nrm, faces, kp)}["toggle:blush"]["image"])
    pink = drawn[drawn[..., 3] > 60]
    assert len(pink) and (pink[:, 0] > pink[:, 1] + 60).all()   # drawn pink cheeks instead


def test_a_strand_touching_one_corner_does_not_punch_a_hole_in_the_eye_decal(tmp_path):
    cam = rig3d.Camera(CAM)
    d = _model_dir(tmp_path)
    pos, faces = _face_sheet(cam)
    v = int(np.argmin(np.linalg.norm(cam.to_image(pos) - [220, 190], axis=1)))   # under the right eye
    strand = []
    for du in ((0, 0), (1.5, 0), (0, 1.5)):   # a tiny bang strand 2 cm in front of that one vertex
        o, ray = cam.ray(cam.to_image(pos[v:v + 1])[0] + du)
        strand.append(o + ray * (o[2] - 0.12) / -ray[2])
    pos = np.concatenate([pos, strand])
    faces = np.concatenate([faces, [[len(pos) - 3, len(pos) - 2, len(pos) - 1]]])
    nrm = np.tile([0.0, 0.0, 1.0], (len(pos), 1))
    kp = np.zeros((133, 3))
    kp[1], kp[2] = (300, 190, .9), (220, 190, .9)
    out = {e["name"]: e for e in rig3d.decals(d, cam, pos, nrm, faces, kp)}
    around = (faces == v).any(1).sum()
    assert (out["blink_r"]["faces"] == v).any(1).sum() == around   # all its triangles stay


def test_open_art_with_painted_vowels_keeps_closed_neutral_and_hides_it_on_speech(tmp_path):
    cam = rig3d.Camera(CAM)
    d = _model_dir(tmp_path)
    record = json.loads((d / "expressions" / "expressions.json").read_text())
    record.pop("mouth_open")
    record["mouth_close"] = "expressions/mouth_open.png"
    record["visemes"]["a"] = "expressions/mouth_open.png"
    record["toggles"]["catmouth"] = "expressions/mouth_open.png"
    (d / "expressions" / "expressions.json").write_text(json.dumps(record))
    pos, faces = _face_sheet(cam)
    nrm = np.tile([0.0, 0.0, 1.0], (len(pos), 1))
    kp = np.zeros((133, 3))
    expr = rig3d.decals(d, cam, pos, nrm, faces, kp)
    names = [e["name"] for e in expr]
    assert names.index("_mouth_neutral") < names.index("a")
    assert expr[names.index("_mouth_neutral")]["shown"]
    assert not expr[names.index("a")]["shown"]

    _, _, _, joints, _ = _figure()
    w = np.zeros((len(pos), len(rig3d.BONES)))
    w[:, rig3d.INDEX["head"]] = 1
    out = tmp_path / "open-mouth.vrm"
    rig3d.write_vrm(out, pos, nrm, np.zeros((len(pos), 2)), faces, w, joints, expr,
                    b"fake png", "image/png", "Test", Image.new("RGB", (8, 8)))
    gltf, _ = _read_vrm(out)
    groups = gltf["extensions"]["VRM"]["blendShapeMaster"]["blendShapeGroups"]
    assert not any(group["name"] == "_MOUTH_NEUTRAL" for group in groups)
    for group in groups:
        if group["presetName"] in ("a", "i", "u", "e", "o") or group["name"] == "Cat mouth":
            assert any(bind["index"] == names.index("_mouth_neutral") and bind["weight"] == 100
                       for bind in group["binds"])
    blush = next(group for group in groups if group["name"] == "Blush")
    assert all(bind["index"] != names.index("_mouth_neutral") for bind in blush["binds"])

    # Older open-mouth characters may have only a closed replacement and a few
    # vowels: A itself is then the neutral decal, not an additional target.
    del record["visemes"]["a"]
    (d / "expressions" / "expressions.json").write_text(json.dumps(record))
    expr = rig3d.decals(d, cam, pos, nrm, faces, kp)
    names = [e["name"] for e in expr]
    assert expr[names.index("a")]["shown"] and "_mouth_neutral" not in names
    rig3d.write_vrm(out, pos, nrm, np.zeros((len(pos), 2)), faces, w, joints, expr,
                    b"fake png", "image/png", "Test", Image.new("RGB", (8, 8)))
    gltf, _ = _read_vrm(out)
    groups = gltf["extensions"]["VRM"]["blendShapeMaster"]["blendShapeGroups"]
    for group in groups:
        if group["presetName"] in ("a", "i", "u", "e", "o") or group["name"] == "Cat mouth":
            binds = [b for b in group["binds"] if b["index"] == names.index("a")]
            assert len(binds) == 1 and binds[0]["weight"] == 100


def test_t_pose_rotation_handles_opposite_and_missing_directions():
    direction = np.array([1.0, 0.0, 0.0])
    for target in (-direction, np.array([-1.0, 1e-7, 0.0]), direction):
        rotation = rig3d._rotation(direction, target)
        assert rotation @ direction == pytest.approx(target / np.linalg.norm(target), abs=1e-7)
        assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-7)
        assert np.linalg.det(rotation) == pytest.approx(1)
    with pytest.raises(ValueError, match="arm direction"):
        rig3d._rotation(np.zeros(3), direction)


def _read_vrm(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", data)
    assert magic == 0x46546C67 and version == 2 and length == len(data)
    return rig3d.read_glb(path)


def test_vrm_has_a_humanoid_skin_and_expression_shapes(tmp_path):
    pos, faces, which, j, radii = _figure()
    w, keep = rig3d.skin_weights(pos, faces, j, radii)
    faces = faces[keep]
    cam = rig3d.Camera(CAM)
    sheet, sheet_faces = _face_sheet(cam, n=8)
    expr = [{"name": n, "label": n, "image": Image.new("RGBA", (16, 8), (1, 2, 3, 255)), "faces": faces[:4],
             "uv": np.zeros((len(pos), 2)), "shown": n == "a"} for n in ("blink_l", "blink_r", "a")]
    expr.append({**expr[0], "name": "toggle:blush", "label": "Blush"})
    out = tmp_path / "avatar.vrm"
    thumb = Image.new("RGB", (64, 64))
    rig3d.write_vrm(out, pos, rig3d.vertex_normals(pos, faces), np.zeros((len(pos), 2)), faces, w, j, expr,
                    b"\x89PNG fake", "image/png", "Test", thumb)
    g, binary = _read_vrm(out)
    vrm = g["extensions"]["VRM"]
    assert {b["bone"] for b in vrm["humanoid"]["humanBones"]} == {name for name, _ in rig3d.BONES}
    assert len(g["skins"][0]["joints"]) == len(rig3d.BONES)
    face = g["meshes"][1]["primitives"][0]
    assert len(face["targets"]) == 4
    groups = vrm["blendShapeMaster"]["blendShapeGroups"]
    assert {"blink", "blink_l", "blink_r", "a", "i", "u", "e", "o", "neutral"} <= {grp["presetName"] for grp in groups}
    assert {"name": "Blush", "presetName": "unknown"}.items() <= next(grp for grp in groups if grp["name"] == "Blush").items()
    # Emotion presets come from painted parts: closed eyes + blush make Joy; no tears, no Sorrow.
    joy = next(grp for grp in groups if grp["presetName"] == "joy")
    assert sorted(b["index"] for b in joy["binds"]) == [0, 1, 3]
    assert not {"angry", "sorrow", "fun"} & {grp["presetName"] for grp in groups}
    for a in g["accessors"]:
        assert a["bufferView"] < len(g["bufferViews"])
    assert all("min" in g["accessors"][t["POSITION"]] for t in face["targets"])
    body = rig3d.accessor(g, binary, g["meshes"][0]["primitives"][0]["attributes"]["WEIGHTS_0"])
    assert np.allclose(body.sum(1), 1, atol=1e-5)
    assert vrm["meta"]["title"] == "Test" and vrm["materialProperties"][0]["shader"] == "VRM/MToon"


@pytest.fixture
def api(tmp_path, monkeypatch):
    import app as app_module
    from src.api import generation

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    d = tmp_path / "outputs" / "abc-123"
    d.mkdir(parents=True)
    Image.new("RGB", (64, 96), (200, 150, 150)).save(d / "base_image.png")
    (d / "metadata.json").write_text(json.dumps({"id": "abc-123", "name": "model_abc", "status": "completed",
                                                  "request": {"prompt": "1girl"}, "metadata": {}}))
    flask_app = app_module.create_app(use_dummy_generators=True)
    submitted = []
    monkeypatch.setattr(generation.job_queue, "submit", lambda req: submitted.append(req) or type("J", (), {"id": "j"})())
    monkeypatch.setattr(generation.job_queue, "describe", lambda job: {"id": job.id, "state": "queued"})
    c = flask_app.test_client()
    c.submitted, c.dir = submitted, d
    return c


def test_avatar3d_api_queues_builds_and_serves_the_vrm(api, monkeypatch):
    from src.api import generation

    assert api.post("/api/generation/avatar3d/nope").status_code == 404
    assert api.post("/api/generation/avatar3d/abc-123", json={"quality": "ultra"}).status_code == 400
    r = api.post("/api/generation/avatar3d/abc-123", json={"quality": "draft"})
    assert r.status_code == 202
    req = api.submitted[-1]
    assert req.operation == "avatar3d" and req.target_model == "abc-123" and req.avatar3d == {"quality": "draft"}
    assert api.get("/api/generation/artifact/abc-123/vrm").status_code == 404

    generation.assembly_line.make_avatar3d("abc-123", options=req.avatar3d)   # the demo mannequin
    saved = json.loads((api.dir / "metadata.json").read_text())["metadata"]["avatar3d"]
    assert saved["vrm"] == "avatar3d/avatar.vrm" and saved["quality"] == "draft"
    assert not (api.dir / "avatar3d.new").exists()
    vrm = api.get("/api/generation/artifact/abc-123/vrm")
    assert vrm.status_code == 200 and vrm.data[:4] == b"glTF"
    assert api.get("/api/generation/options").get_json()["avatar3d"]["available"] is True

    ok = api.post("/api/generation/generate", json={"prompt": "1girl", "framing": "upper_body", "avatar3d": True})
    assert ok.status_code == 202
    assert api.submitted[-1].framing == "full_body" and api.submitted[-1].avatar3d == {"quality": "standard"}

    monkeypatch.setattr(generation.assembly_line, "use_dummy_generators", False)
    monkeypatch.setattr(mesh3d, "available", lambda: False)
    r = api.post("/api/generation/avatar3d/abc-123")
    assert r.status_code == 400 and "install_3d" in r.get_json()["error"]


def test_a_failed_rebuild_keeps_the_previous_avatar(api, monkeypatch):
    from src.api import generation

    line = generation.assembly_line
    line.make_avatar3d("abc-123")
    before = (api.dir / "avatar3d" / "avatar.vrm").read_bytes()

    def broken(model_dir, out_dir, name):
        out_dir.mkdir(parents=True)
        (out_dir / "avatar.vrm").write_bytes(b"half")
        raise RuntimeError("engine died")
    monkeypatch.setattr(rig3d, "demo", broken)
    with pytest.raises(RuntimeError):
        line.make_avatar3d("abc-123")
    assert (api.dir / "avatar3d" / "avatar.vrm").read_bytes() == before
    assert not (api.dir / "avatar3d.new").exists()


def test_long_hair_gets_spring_chains_in_the_vrm(tmp_path):
    pos, faces, which, j, radii = _figure()
    lock, lock_faces = [], []   # a side lock down to the chest, in pieces so it has a few dozen vertices
    for y in np.linspace(.3, .06, 6):
        b, t = _box((.07, y, .05), (.07, y - .04, .05), .012)
        lock_faces.append(t + len(pos) + len(lock))
        lock += list(b)
    lock, lock_faces = np.array(lock), np.concatenate(lock_faces)
    pos, faces = np.concatenate([pos, lock]), np.concatenate([faces, lock_faces])
    which = np.concatenate([which, ["lock"] * len(lock)])
    parts = np.where(which == "lock", "hair", np.where(which == "head", "hair", "body")).astype(object)
    w, keep = rig3d.skin_weights(pos, faces, j, radii, parts)
    faces = faces[keep]
    w2, springs = rig3d.hair_springs(pos, parts, w, j)
    assert len(springs["roots"]) == 1 and w2.shape[1] == len(rig3d.BONES) + 3
    assert np.allclose(w2.sum(1), 1)
    tip = np.nonzero(which == "lock")[0][pos[which == "lock", 1].argmin()]
    assert w2[tip, len(rig3d.BONES):].sum() > .99          # the end of the lock swings on the chain
    skull = (which == "head") & (pos[:, 1] > .3)   # above where the chain hangs from
    assert (w2[skull, len(rig3d.BONES):] == 0).all()
    joints = {**j, **springs.pop("joints")}
    springs["colliders"] = [("head", .1, .08), ("chest", 0, .1)]
    out = tmp_path / "hair.vrm"
    rig3d.write_vrm(out, pos, rig3d.vertex_normals(pos, faces), np.zeros((len(pos), 2)), faces, w2, joints, [],
                    b"png", "image/png", "Hair", Image.new("RGB", (8, 8)), springs)
    g, _ = _read_vrm(out)
    anim = g["extensions"]["VRM"]["secondaryAnimation"]
    root = anim["boneGroups"][0]["bones"][0]
    assert g["nodes"][root]["name"] == springs["roots"][0] and len(anim["colliderGroups"]) == 2
    assert len(g["skins"][0]["joints"]) == len(rig3d.BONES) + 3


def _with_chest_front(bottom=-.06):
    """_figure plus a dense sheet on the torso's front (the box alone has too few vertices)."""
    pos, faces, which, j, radii = _figure()
    xs, ys = np.meshgrid(np.linspace(-.055, .055, 24), np.linspace(bottom, .22, 48))
    sheet = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, .061)], 1)
    k = np.arange(xs.size).reshape(xs.shape)[:-1, :-1].ravel()
    tri = np.concatenate([np.stack([k, k + 1, k + 24], 1), np.stack([k + 1, k + 25, k + 24], 1)])
    which = np.concatenate([which, ["chest"] * len(sheet)])
    return np.concatenate([pos, sheet]), np.concatenate([faces, tri + len(pos)]), which, j, radii


def test_the_chest_front_gets_two_spring_bones_in_the_vrm(tmp_path):
    pos, faces, which, j, radii = _with_chest_front()
    parts = np.where(np.char.startswith(which.astype(str), "left") | np.char.startswith(which.astype(str), "right"),
                     "arm", "body").astype(object)
    w, _ = rig3d.skin_weights(pos, faces, j, radii)
    springs = {"bones": [], "roots": [], "colliders": [], "joints": {}}
    w2 = rig3d.bust_springs(pos, parts, w, j, springs)
    assert springs["bust_roots"] == ["bustL", "bustR"] and w2.shape[1] == len(rig3d.BONES) + 4
    assert np.allclose(w2.sum(1), 1)
    sheet = which == "chest"
    centre = np.nonzero(sheet)[0][np.argmin(np.abs(pos[sheet, 0] - .044) + np.abs(pos[sheet, 1] - .125))]
    assert w2[centre, len(rig3d.BONES)] > .5                    # the left side moves on bustL
    assert (w2[pos[:, 2] < 0, len(rig3d.BONES):] == 0).all()    # the back stays with the chest
    assert (w2[parts == "arm", len(rig3d.BONES):] == 0).all()
    assert springs["joints"]["bustL_tip"][2] > springs["joints"]["bustL"][2]   # the tip points forward
    out = tmp_path / "bust.vrm"
    rig3d.write_vrm(out, pos, rig3d.vertex_normals(pos, faces), np.zeros((len(pos), 2)), faces, w2,
                    {**j, **springs.pop("joints")}, [], b"png", "image/png", "Bust", Image.new("RGB", (8, 8)), springs)
    g, _ = _read_vrm(out)
    groups = g["extensions"]["VRM"]["secondaryAnimation"]["boneGroups"]
    assert [x["comment"] for x in groups] == ["bust"]
    assert [g["nodes"][b]["name"] for b in groups[0]["bones"]] == ["bustL", "bustR"]


def test_a_bust_cut_off_at_the_chest_gets_no_chest_springs():
    pos, faces, which, j, radii = _with_chest_front(bottom=.11)
    keep = (pos[:, 1] > .1) | (which != "torso")   # the torso box's lower corners go too
    keep &= ~np.isin(which, ["leftLeg", "rightLeg"])
    remap = np.cumsum(keep) - 1
    faces = remap[faces[keep[faces].all(1)]]
    pos, which = pos[keep], which[keep]
    w, _ = rig3d.skin_weights(pos, faces, j, radii)
    springs = {"bones": [], "roots": [], "colliders": [], "joints": {}}
    assert rig3d.bust_springs(pos, None, w, j, springs) is w and not springs["bones"]


def test_cut_contacts_are_capped_closed():
    pos, faces = _box((0, 0, 0), (0, 1, 0), .2)
    w = np.zeros((len(pos), len(rig3d.BONES)))
    w[:, rig3d.INDEX["leftHand"]] = 1
    kept = faces[2:]   # cut one side open (two triangles)
    p2, n2, uv2, w2, f2 = rig3d.cap_holes(pos, rig3d.vertex_normals(pos, faces), np.zeros((len(pos), 2)), w, kept, faces)
    assert len(p2) == len(pos) + 1 and len(f2) == len(kept) + 4
    edges = set(zip(f2.ravel().tolist(), f2[:, [1, 2, 0]].ravel().tolist()))
    assert all((b, a) in edges for a, b in edges)   # closed again, every edge used both ways
    assert np.allclose(w2[-1, rig3d.INDEX["leftHand"]], 1) and len(uv2) == len(p2) == len(n2)


def test_hair_that_starts_below_the_jaw_still_gets_a_chain():
    pos, faces, which, j, radii = _figure()
    lock = np.concatenate([_box((.07, y, .035), (.07, y - .04, .035), .012)[0] for y in np.linspace(.12, .0, 4)])
    pos = np.concatenate([pos, lock])   # nothing between the jaw line and y = .12: a link and more
    parts = np.array(["body"] * (len(pos) - len(lock)) + ["hair"] * len(lock), object)
    w = np.zeros((len(pos), len(rig3d.BONES)))
    w[:, rig3d.INDEX["head"]] = 1
    w2, springs = rig3d.hair_springs(pos, parts, w, j)
    assert len(springs["roots"]) == 1 and np.allclose(w2.sum(1), 1)


def test_weld_joins_uv_seams_so_caps_and_labels_cross_them():
    pos, faces = _box((0, 0, 0), (0, 1, 0), .2)
    # split the box along a seam: the top cap's corners get their own copies, as a UV seam does
    dup = np.concatenate([pos, pos[4:8]])
    faces = faces.copy()
    top = (faces >= 4).all(1)
    faces[top] += 4
    canon = rig3d.weld(dup)
    assert (canon[8:] == np.arange(4, 8)).all() and (canon[:8] == np.arange(8)).all()
    labels = np.full(len(dup), "", object)
    labels[0] = "topwear"
    assert (rig3d.spread_parts(canon[faces], dup, labels)[canon] == "body").all()   # across the seam
    w = np.zeros((len(dup), len(rig3d.BONES)))
    w[:, 0] = 1
    kept = faces[2:]
    _, _, _, _, f2 = rig3d.cap_holes(dup, rig3d.vertex_normals(dup, faces), np.zeros((len(dup), 2)), w, kept, faces, canon)
    assert len(f2) == len(kept) + 4


def test_non_hair_head_accessories_follow_head_without_spring_bones():
    pos, faces, which, j, radii = _figure()
    accessory, accessory_faces = [], []
    for y in np.linspace(.22, .02, 6):
        v, f = _box((.07, y, .05), (.07, y - .035, .05), .012)
        accessory_faces.append(f + len(pos) + len(accessory))
        accessory += list(v)
    pos = np.concatenate([pos, np.array(accessory)])
    faces = np.concatenate([faces, *accessory_faces])
    labels = np.full(len(pos), "", object)
    labels[len(pos) - len(accessory):] = "earwear"
    labels[:len(which)] = np.where(which == "head", "face", "topwear")
    parts = rig3d.spread_parts(faces, pos, labels)
    w, keep = rig3d.skin_weights(pos, faces, j, radii, parts)
    w2, springs = rig3d.hair_springs(pos, parts, w, j)
    assert (parts[len(which):] == "head_attachment").all()
    assert not springs["roots"] and w2.shape == w.shape
    assert (w[len(which):, rig3d.INDEX["head"]] > .99).all(), "Head accessories stay attached even when long"
    assert rig3d.PART_OF["front hair"] == "hair" and rig3d.PART_OF["headwear"] == "head_attachment"


def test_front_labels_ignore_outside_image_vertices_and_their_depth(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    Image.new("RGBA", (20, 20), (180, 60, 70, 255)).save(assets / "00_headwear.png")
    class Camera:
        side = 20
        @staticmethod
        def to_image(pos):
            return pos[:, :2]
    # Outside points at high depth must neither acquire edge labels nor occlude
    # the visible point just inside the same border cell.
    pos = np.array([[0.5, 10, 0], [-.5, 10, 9], [20, 10, 9], [5, -1, 9], [np.nan, 10, 9]], float)
    labels = rig3d.front_labels(tmp_path, Camera(), pos, tolerance=.01)
    assert labels.tolist() == ["headwear", "", "", "", ""]


def test_closed_eye_decal_hides_the_open_eye_and_keeps_the_bangs(tmp_path):
    assets = tmp_path / "assets"; assets.mkdir()
    def layer(name, box, colour):
        img = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
        img.paste(Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), colour), box[:2])
        img.save(assets / f"{name}.png")
    layer("05_face", (0, 0, 40, 20), (200, 150, 130, 255))      # inpainted skin under everything
    layer("08_eyewhite-l", (10, 5, 30, 15), (255, 255, 255, 255))
    layer("15_front hair", (10, 0, 15, 20), (40, 40, 90, 255))   # a strand across the eye's left edge
    Image.new("RGB", (40, 20), (7, 8, 9)).save(tmp_path / "base_image.png")   # the artwork
    closed = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
    closed.paste(Image.new("RGBA", (24, 14), (60, 30, 30, 128)), (8, 3))   # feathered painting
    out = np.asarray(rig3d.cover_open_eyes(tmp_path, closed))
    assert out[10, 20, 3] == 255 and tuple(out[10, 20, :3]) != (255, 255, 255)   # opaque, no eye white
    assert abs(int(out[10, 20, 0]) - (60 + 200) // 2) <= 2    # the painting over the skin
    assert tuple(out[10, 12]) == (7, 8, 9, 255)               # under the bangs: the artwork's hair
    assert out[1, 2, 3] == 0                                  # away from the eye: untouched
    assert np.array_equal(np.asarray(rig3d.cover_open_eyes(tmp_path / "none", closed)), np.asarray(closed))


def test_gallery_flags_avatars_rigged_before_the_current_rig(api):
    meta = json.loads((api.dir / "metadata.json").read_text())
    meta["metadata"]["avatar3d"] = {"quality": "standard"}
    (api.dir / "metadata.json").write_text(json.dumps(meta))
    mesh = api.dir / "avatar3d"; mesh.mkdir()
    for name in ("mesh.glb", "camera.json", "crop.png"):
        (mesh / name).write_bytes(b"x")
    (mesh / "rig.json").write_text(json.dumps({"triangles": 1}))   # rigged before versions existed
    avatar = lambda: api.get("/api/generation/list").get_json()["models"][0]["metadata"]["avatar3d"]
    assert avatar()["rig_outdated"] is True
    (mesh / "rig.json").write_text(json.dumps({"rig_version": rig3d.RIG_VERSION}))
    assert avatar()["rig_outdated"] is False
    (mesh / "crop.png").unlink()   # cannot refresh without the saved inputs: no offer
    assert "rig_outdated" not in avatar()
