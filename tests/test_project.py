"""
Saving and reopening characters: the Gallery's Export zip opens again as a new result with its
paths moved, unsafe archives are refused, and Studio adjustments are kept with the character.
"""
import io
import json
import zipfile

import pytest
from PIL import Image

import config


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app as app_module

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    old = r"D:\elsewhere\Ka-myii\outputs\abc-123"  # exported on another machine
    model = tmp_path / "outputs" / "abc-123"
    (model / "assets").mkdir(parents=True)
    (model / "model" / "model_x").mkdir(parents=True)
    Image.new("RGB", (64, 96), "white").save(model / "base_image.png")
    Image.new("RGBA", (64, 96), (9, 9, 9, 255)).save(model / "assets" / "00_hair.png")
    (model / "model" / "model_x" / "model_x.psd").write_bytes(b"8BPS")
    (model / "model" / "model_x" / "kamyii_manifest.json").write_text(json.dumps({"artifacts": {"psd": old + r"\model\model_x\model_x.psd"}}))
    (model / "metadata.json").write_text(json.dumps({
        "id": "abc-123", "name": "model_x", "display_name": "Mika", "status": "completed",
        "created_at": "2026-10-01T10:00:00", "base_image_path": old + r"\base_image.png",
        "assets": [{"layer_type": "hair", "file_path": old + r"\assets\00_hair.png", "metadata": {}}],
        "final_model_path": old + r"\model\model_x",
        "request": {"prompt": "1girl", "model_id": r"D:\elsewhere\Ka-myii\models\checkpoints\x.safetensors"},
        "metadata": {"artifacts": {"psd": old + r"\model\model_x\model_x.psd"}}}))
    return app_module.create_app(use_dummy_generators=True).test_client()


def _zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    buf.seek(0)
    return buf


def _open(client, archive, name="mika.zip"):
    return client.post("/api/generation/open-project", data={"project": (archive, name)}, content_type="multipart/form-data")


def test_export_reopens_as_a_new_character_with_moved_paths(client):
    exported = client.get("/api/generation/download/abc-123")
    assert exported.status_code == 200
    r = _open(client, io.BytesIO(exported.data))
    assert r.status_code == 200, r.get_json()
    new_id = r.get_json()["model_id"]
    assert new_id != "abc-123"  # the original is still here
    folder = config.OUTPUT_DIR / new_id
    meta = json.loads((folder / "metadata.json").read_text())
    assert meta["id"] == new_id and meta["display_name"] == "Mika (imported)"
    assert meta["base_image_path"] == str(folder / "base_image.png")
    assert all(__import__("pathlib").Path(p).exists() for p in
               (meta["base_image_path"], meta["assets"][0]["file_path"], meta["metadata"]["artifacts"]["psd"]))
    assert meta["request"]["model_id"].endswith("x.safetensors") and "elsewhere" in meta["request"]["model_id"]
    manifest = json.loads((folder / "model" / "model_x" / "kamyii_manifest.json").read_text())
    assert manifest["artifacts"]["psd"] == str(folder / "model" / "model_x" / "model_x.psd")
    ids = {m["id"] for m in client.get("/api/generation/list").get_json()["models"]}
    assert ids == {"abc-123", new_id}
    assert not any((config.OUTPUT_DIR / "_uploads").glob("import-*"))  # staging cleaned up


def test_a_rezipped_folder_keeps_its_id_when_free(client):
    meta = {"id": "fresh-1", "status": "sketch", "created_at": "2026-10-02T10:00:00"}
    r = _open(client, _zip({"fresh-1/metadata.json": json.dumps(meta), "fresh-1/base_image.png": b"png"}))
    assert r.get_json()["model_id"] == "fresh-1" and (config.OUTPUT_DIR / "fresh-1" / "base_image.png").exists()


@pytest.mark.parametrize("archive, word", [
    (io.BytesIO(b"not a zip"), "not a zip"),
    (_zip({"readme.txt": "hi"}), "no metadata.json"),
    (_zip({"metadata.json": "{oops"}), "not valid JSON"),
    (_zip({"metadata.json": json.dumps({"id": "e"}), "../escape.txt": "x"}), "Unsafe path"),
])
def test_bad_archives_are_refused(client, tmp_path, archive, word):
    r = _open(client, archive)
    assert r.status_code == 400 and word in r.get_json()["error"]
    assert not (tmp_path / "outputs" / "escape.txt").exists() and not (tmp_path / "escape.txt").exists()
    assert {p.name for p in config.OUTPUT_DIR.iterdir() if p.name != "_uploads"} == {"abc-123"}


def test_studio_adjustments_are_kept_with_the_character(client):
    settings = {"format": "anime25d-settings", "version": 2, "params": {"eyeOpen": 0.8}, "layers": []}
    assert client.put("/api/generation/studio/abc-123", json=settings).get_json()["success"]
    assert json.loads((config.OUTPUT_DIR / "abc-123" / "studio.json").read_text()) == settings
    page = client.get("/preview/abc-123").get_data(as_text=True)
    assert '"eyeOpen": 0.8' in page and '"id": "abc-123"' not in page  # settings injected, id passed separately
    assert client.put("/api/generation/studio/abc-123", json=["nope"]).status_code == 400
    assert client.put("/api/generation/studio/missing", json=settings).status_code == 404


def test_only_project_import_accepts_big_bodies(client):
    big = b"x" * (17 * 1024 * 1024)
    assert client.post("/api/generation/upload", data=big, content_type="application/octet-stream").status_code == 413
    assert client.post("/api/generation/open-project", data=big, content_type="application/octet-stream").status_code == 400


def test_pngtuber_pack_has_four_aligned_states(client):
    folder = config.OUTPUT_DIR / "abc-123"
    assert client.get("/api/generation/artifact/abc-123/pngtuber").status_code == 422  # no cut-out yet
    cut = Image.new("RGBA", (128, 160), (0, 0, 0, 0)); cut.paste((200, 180, 170, 255), (40, 40, 80, 120))
    cut.save(folder / "assets" / "character.png")
    assert "blink & talk" in client.get("/api/generation/artifact/abc-123/pngtuber").get_json()["error"]
    (folder / "expressions").mkdir()
    eyes = Image.new("RGBA", (128, 160), (0, 0, 0, 0)); eyes.paste((150, 100, 100, 255), (50, 50, 70, 55))
    mouth = Image.new("RGBA", (128, 160), (0, 0, 0, 0)); mouth.paste((90, 20, 30, 255), (55, 70, 65, 76))
    eyes.save(folder / "expressions" / "eye_close.png"); mouth.save(folder / "expressions" / "mouth_open.png")
    (folder / "expressions" / "expressions.json").write_text(json.dumps(
        {"eye_close": "expressions/eye_close.png", "mouth_open": "expressions/mouth_open.png", "mouth_drawn": "closed"}))

    r = client.get("/api/generation/artifact/abc-123/pngtuber")
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.data)) as z:
        assert sorted(z.namelist()) == ["README.txt", "blink.png", "idle.png", "talk.png", "talk_blink.png"]
        imgs = {n[:-4]: Image.open(io.BytesIO(z.read(n))).convert("RGBA") for n in z.namelist() if n.endswith(".png")}
    assert {im.size for im in imgs.values()} == {(72, 112)}  # the character + 16 px padding, all alike
    mouth_px, eye_px = (60 - 24, 73 - 24), (60 - 24, 52 - 24)  # canvas coordinates minus the crop origin
    assert imgs["idle"].getpixel(mouth_px)[:3] == (200, 180, 170) and imgs["talk"].getpixel(mouth_px)[:3] == (90, 20, 30)
    assert imgs["blink"].getpixel(eye_px)[:3] == (150, 100, 100) and imgs["idle"].getpixel(eye_px)[:3] == (200, 180, 170)
    assert imgs["talk_blink"].getpixel(mouth_px)[:3] == (90, 20, 30) and imgs["talk_blink"].getpixel(eye_px)[:3] == (150, 100, 100)
