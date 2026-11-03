"""
HTTP-boundary checks: model ids from URLs, /generate validation, cross-site access.
"""
import pytest

import config


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app as app_module
    from src.api import generation

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    (tmp_path / "outputs" / "abc-123").mkdir(parents=True)
    (tmp_path / "keep.txt").write_text("outside the outputs folder")
    flask_app = app_module.create_app(use_dummy_generators=True)
    submitted = []
    monkeypatch.setattr(generation.job_queue, "submit", lambda req: submitted.append(req) or _FakeJob())
    monkeypatch.setattr(generation.job_queue, "describe", lambda job: {"id": job.id, "state": "queued"})
    c = flask_app.test_client()
    c.submitted = submitted
    c.app = flask_app
    return c


class _FakeJob:
    id = "job-1"


def test_model_ids_cannot_escape_the_outputs_folder(client, tmp_path):
    for bad in ("..", "%2e%2e", "abc-123%2f..", "..%5c.."):
        assert client.delete(f"/api/generation/delete/{bad}").status_code == 404
        assert client.get(f"/api/generation/download/{bad}").status_code == 404
        assert client.get(f"/api/generation/layers/{bad}").status_code == 404
    assert (tmp_path / "keep.txt").exists() and (tmp_path / "outputs").exists()
    assert client.delete("/api/generation/delete/abc-123").status_code == 200
    assert not (tmp_path / "outputs" / "abc-123").exists()


def test_generate_validates_the_brief(client):
    ok = client.post("/api/generation/generate", json={"prompt": "1girl", "width": 833, "seed": "7"})
    assert ok.status_code == 202
    req = client.submitted[-1]
    assert req.width == 832 and req.seed == 7 and req.model_id is None

    for body, word in [
        (None, "JSON"),
        ({"prompt": ""}, "Prompt"),
        ({"prompt": "x", "width": 100000}, "width"),
        ({"prompt": "x", "steps": "lots"}, "steps"),
        ({"prompt": "x", "model_id": "C:/Windows/notepad.exe"}, "checkpoint"),
        ({"prompt": "x", "loras": [{"id": "../../evil.pt"}]}, "LoRA"),
        ({"prompt": "x", "sampler": "nope"}, "sampler"),
        ({"prompt": "x, " * 3000}, "characters"),
    ]:
        r = client.post("/api/generation/generate", json=body) if body is not None else \
            client.post("/api/generation/generate", data="x", content_type="text/plain")
        assert r.status_code == 400 and word in r.get_json()["error"], body
    assert len(client.submitted) == 1


def test_no_cross_site_access(client):
    r = client.get("/api/generation/options", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in r.headers
    client.app.config["LOCAL_ONLY"] = True  # what main() sets when bound to loopback
    assert client.get("/health", headers={"Host": "rebind.evil.example:5000"}).status_code == 403
    assert client.get("/health", headers={"Host": "127.0.0.1:5000"}).status_code == 200
    assert client.get("/health", headers={"Host": "localhost"}).status_code == 200


def test_upload_queues_the_users_artwork(client, tmp_path):
    import io
    from PIL import Image

    def png(size, mode="RGBA"):
        buf = io.BytesIO()
        Image.new(mode, size, (200, 100, 50, 0) if mode == "RGBA" else (200, 100, 50)).save(buf, "PNG")
        buf.seek(0)
        return buf

    r = client.post("/api/generation/upload", data={"image": (png((3000, 1500)), "my oc.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 202
    req = client.submitted[-1]
    assert req.prompt == "Uploaded artwork: my oc" and (req.width, req.height) == (2048, 1024)
    assert req.source == "upload" and req.sketch  # a prepared sketch first
    assert req.import_options == {"frame": True, "plain_background": True, "enhance": 0.0}
    with Image.open(req.reference_image) as saved:
        assert saved.mode == "RGB" and saved.getpixel((5, 5)) == (255, 255, 255)  # transparency → white

    # A box around one character of a sheet: fractions of the picture.
    r = client.post("/api/generation/upload", data={"image": (png((1200, 800), "RGB"), "sheet.png"),
                                                    "crop": "0.5,0.25,1,1"}, content_type="multipart/form-data")
    assert r.status_code == 202 and (client.submitted[-1].width, client.submitted[-1].height) == (600, 600)

    for data, word in [({}, "Choose"), ({"image": (io.BytesIO(b"not an image"), "x.png")}, "not an image"),
                       ({"image": (png((100, 100), "RGB"), "tiny.png")}, "too small"),
                       ({"image": (png((1200, 800), "RGB"), "a.png"), "crop": "0,0,0.1,0.1"}, "too small"),
                       ({"image": (png((1200, 800), "RGB"), "a.png"), "crop": "0.6,0,0.2,1"}, "crop"),
                       ({"image": (png((1200, 800), "RGB"), "a.png"), "crop": "left"}, "crop")]:
        r = client.post("/api/generation/upload", data=data, content_type="multipart/form-data")
        assert r.status_code == 400 and word in r.get_json()["error"]
    assert len(client.submitted) == 2


def test_uploaded_artwork_runs_the_pipeline_without_sdxl(tmp_path):
    from PIL import Image
    from src.models.vtuber_model import GenerationRequest
    from src.pipeline.assembly_line import AssemblyLine

    art = tmp_path / "upload.png"
    Image.new("RGB", (320, 480), "white").save(art)
    line = AssemblyLine(output_base_dir=tmp_path / "out", use_dummy_generators=True)
    line.image_generator.generate = None  # must not be called
    model = line.generate_model(GenerationRequest(prompt="Uploaded artwork: oc", reference_image=str(art), source="upload"))
    assert model.status.value == "completed" and not art.exists()
    assert (tmp_path / "out" / model.id / "base_image.png").exists()
    assert model.request.reference_image is None and model.request.seed is None
    assert model.metadata["source"] == "upload"


def test_cancelling_a_queued_upload_deletes_its_file(tmp_path):
    from src.models.vtuber_model import GenerationRequest
    from src.pipeline.jobs import JobQueue
    import threading

    gate = threading.Event()
    queue = JobQueue(lambda req, task_id: gate.wait(5) and "done")
    queue.submit(GenerationRequest(prompt="first"))  # occupies the worker
    art = tmp_path / "queued.png"; art.write_bytes(b"x")
    job = queue.submit(GenerationRequest(prompt="Uploaded artwork: x", reference_image=str(art)))
    queue.cancel(job.id)
    gate.set()
    assert job.state == "cancelled" and not art.exists()


def test_retry_requeues_a_saved_results_artwork(client, tmp_path):
    import json
    from PIL import Image

    folder = tmp_path / "outputs" / "abc-123"
    Image.new("RGB", (400, 600), "white").save(folder / "base_image.png")
    (folder / "metadata.json").write_text(json.dumps({"status": "sketch", "request": {
        "prompt": "1girl, witch", "seed": 42, "sampler": "Euler a", "sketch": True, "not_a_field": 1}}))
    r = client.post("/api/generation/retry/abc-123")
    assert r.status_code == 202
    req = client.submitted[-1]
    assert req.prompt == "1girl, witch" and req.reference_image and (req.width, req.height) == (400, 600)
    # The separated character keeps the sketch's brief, but is a full run now.
    assert req.seed == 42 and req.sampler == "Euler a" and req.source == "artwork" and not req.sketch
    assert client.post("/api/generation/retry/..").status_code == 404
    assert client.post("/api/generation/retry/missing").status_code == 404


def test_rename_sets_and_clears_a_display_name(client, tmp_path):
    import json
    meta = tmp_path / "outputs" / "abc-123" / "metadata.json"
    meta.write_text(json.dumps({"id": "abc-123", "name": "model_abcd1234"}))
    r = client.post("/api/generation/rename/abc-123", json={"name": "  Mira \n the\x00 witch  "})
    assert r.status_code == 200 and r.get_json()["display_name"] == "Mira the witch"
    assert json.loads(meta.read_text())["display_name"] == "Mira the witch"
    client.post("/api/generation/rename/abc-123", json={"name": ""})
    assert "display_name" not in json.loads(meta.read_text())
    assert client.post("/api/generation/rename/..", json={"name": "x"}).status_code == 404
    assert len(client.post("/api/generation/rename/abc-123", json={"name": "x" * 500}).get_json()["display_name"]) == 60


def test_sketch_stops_after_the_artwork(tmp_path):
    from src.models.vtuber_model import GenerationRequest
    from src.pipeline.assembly_line import AssemblyLine

    line = AssemblyLine(output_base_dir=tmp_path / "out", use_dummy_generators=True)
    line.asset_separator.separate = None  # must not be called
    model = line.generate_model(GenerationRequest(prompt="1girl", width=256, height=384, steps=2, sketch=True))
    folder = tmp_path / "out" / model.id
    assert model.status.value == "sketch" and isinstance(model.request.seed, int)
    assert (folder / "base_image.png").exists() and not (folder / "preview.png").exists() and not (folder / "assets").exists()


def test_preview_thumbnail_is_small_cached_and_follows_the_art(tmp_path, monkeypatch):
    import os
    import config
    from PIL import Image
    import app as app_module
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    d = tmp_path / "outputs" / "thumb-1"; d.mkdir(parents=True)
    Image.new("RGBA", (832, 1216), (200, 120, 90, 255)).save(d / "base_image.png")
    (d / "metadata.json").write_text('{"id": "thumb-1"}')
    client = app_module.create_app(use_dummy_generators=True).test_client()
    r = client.get("/api/generation/preview/thumb-1?size=thumb")
    assert r.status_code == 200 and r.mimetype == "image/webp"
    from io import BytesIO
    assert max(Image.open(BytesIO(r.data)).size) == 640
    first = (d / "thumb.webp").stat().st_mtime
    assert client.get("/api/generation/preview/thumb-1?size=thumb").status_code == 200
    assert (d / "thumb.webp").stat().st_mtime == first      # cached
    Image.new("RGBA", (832, 1216), (10, 200, 90, 255)).save(d / "base_image.png")
    os.utime(d / "base_image.png", (first + 5, first + 5))
    r = client.get("/api/generation/preview/thumb-1?size=thumb")
    assert Image.open(BytesIO(r.data)).convert("RGB").getpixel((10, 10))[1] > 150   # remade from the new art
    assert client.get("/api/generation/preview/thumb-1").mimetype == "image/png"   # full size unchanged
    assert not list(d.glob(".thumb-*"))


def test_other_websites_cannot_post_to_the_local_server(tmp_path, monkeypatch):
    import config
    import app as app_module
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    client = app_module.create_app(use_dummy_generators=True).test_client()
    form = {"prompt": "1girl"}
    for headers in ({"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"},   # another localhost port
                    {"Origin": "https://evil.example"}, {"Origin": "null"}):
        assert client.post("/api/generation/generate", json=form, headers=headers).status_code == 403, headers
    # Same-origin passes the guard (404: no such character; nothing queued that could outlive the test).
    ok = client.post("/api/generation/rename/no-such-id", json={"name": "x"},
                     headers={"Sec-Fetch-Site": "same-origin", "Origin": "http://localhost"})
    assert ok.status_code == 404
    assert client.get("/api/generation/list", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 200


def test_reveal_opens_only_a_real_character_folder_for_a_local_browser(tmp_path, monkeypatch):
    import os
    import sys
    import config
    import app as app_module
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    d = tmp_path / "outputs" / "abc"; d.mkdir(parents=True)
    (d / "metadata.json").write_text('{"id": "abc"}')
    opened = []
    monkeypatch.setattr(os, "startfile", lambda p: opened.append(p), raising=False)
    monkeypatch.setattr("subprocess.Popen", lambda args: opened.append(args[-1]))
    client = app_module.create_app(use_dummy_generators=True).test_client()
    assert client.post("/api/generation/reveal/abc").status_code == 200
    assert str(opened[-1]) == str(d)
    assert client.post("/api/generation/reveal/..").status_code == 404
    assert client.post("/api/generation/reveal/abc", environ_base={"REMOTE_ADDR": "192.168.1.9"}).status_code == 403
    assert len(opened) == 1


def test_favorite_stars_and_unstars_a_character(tmp_path, monkeypatch):
    import json
    import config
    import app as app_module
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    d = tmp_path / "outputs" / "fav"; d.mkdir(parents=True)
    (d / "metadata.json").write_text('{"id": "fav", "status": "completed"}')
    client = app_module.create_app(use_dummy_generators=True).test_client()
    assert client.post("/api/generation/favorite/fav", json={"favorite": True}).get_json()["favorite"] is True
    assert json.loads((d / "metadata.json").read_text())["favorite"] is True
    assert client.get("/api/generation/list").get_json()["models"][0]["favorite"] is True
    client.post("/api/generation/favorite/fav", json={"favorite": False})
    assert "favorite" not in json.loads((d / "metadata.json").read_text())
    assert client.post("/api/generation/favorite/fav", json={"favorite": "yes"}).status_code == 400
    assert client.post("/api/generation/favorite/nope", json={"favorite": True}).status_code == 404


def test_delete_moves_to_the_trash_and_can_be_undone(tmp_path, monkeypatch):
    import os
    import time
    import config
    import app as app_module
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    out = tmp_path / "outputs"
    for name in ("mika", "old-one"):
        (out / name).mkdir(parents=True)
        (out / name / "metadata.json").write_text('{"id": "%s"}' % name)
    (out / "_uploads").mkdir()   # staging for uploads: an app folder
    client = app_module.create_app(use_dummy_generators=True).test_client()
    assert client.delete("/api/generation/delete/mika").get_json()["restorable"] is True
    assert not (out / "mika").exists() and (out / "_trash" / "mika" / "metadata.json").exists()
    assert [m["id"] for m in client.get("/api/generation/list").get_json()["models"]] == ["old-one"]
    assert client.post("/api/generation/restore/mika").status_code == 200
    assert (out / "mika" / "metadata.json").exists() and not (out / "mika" / ".deleted").exists()
    assert client.post("/api/generation/restore/mika").status_code == 404      # nothing left to restore
    # Characters deleted more than a week ago are emptied from the trash by the next delete.
    client.delete("/api/generation/delete/old-one")
    week_ago = time.time() - 8 * 86400
    os.utime(out / "_trash" / "old-one" / ".deleted", (week_ago, week_ago))
    client.delete("/api/generation/delete/mika")
    assert not (out / "_trash" / "old-one").exists() and (out / "_trash" / "mika").exists()
    # App folders are not characters.
    assert client.delete("/api/generation/delete/_uploads").status_code == 404
    assert client.delete("/api/generation/delete/_trash").status_code == 404
    assert client.post("/api/generation/restore/..").status_code == 404
    assert (out / "_uploads").exists()


def test_system_status_and_empty_trash(tmp_path, monkeypatch):
    import config
    import app as app_module
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    out = tmp_path / "outputs"
    (out / "mika").mkdir(parents=True)
    (out / "mika" / "metadata.json").write_text('{"id": "mika"}')
    (out / "mika" / "base_image.png").write_bytes(b"x" * 1000)
    (out / "_uploads").mkdir()
    client = app_module.create_app(use_dummy_generators=True).test_client()
    s = client.get("/api/generation/system").get_json()
    assert s["demo"] is True and s["characters"] == 1 and s["characters_bytes"] >= 1000
    assert s["trash_items"] == 0 and s["free_bytes"] > 0 and s["layers_engine"] is False
    client.delete("/api/generation/delete/mika")
    s = client.get("/api/generation/system").get_json()
    assert s["characters"] == 0 and s["trash_items"] == 1 and s["trash_bytes"] >= 1000
    assert client.post("/api/generation/trash/empty").get_json()["removed"] == 1
    assert client.get("/api/generation/system").get_json()["trash_items"] == 0
    assert (out / "_uploads").exists()


def test_notes_are_kept_with_the_character(tmp_path, monkeypatch):
    import json
    import config
    import app as app_module
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    d = tmp_path / "outputs" / "n1"; d.mkdir(parents=True)
    (d / "metadata.json").write_text('{"id": "n1"}')
    client = app_module.create_app(use_dummy_generators=True).test_client()
    r = client.post("/api/generation/notes/n1", json={"notes": "  stream intro\noutfit\x07  " + "x" * 600})
    notes = r.get_json()["notes"]
    assert notes.startswith("stream intro\noutfit") and "\x07" not in notes and len(notes) == 500
    assert client.get("/api/generation/list").get_json()["models"][0]["notes"] == notes
    client.post("/api/generation/notes/n1", json={"notes": ""})
    assert "notes" not in json.loads((d / "metadata.json").read_text())
    assert client.post("/api/generation/notes/n1", json={"notes": 5}).status_code == 400
    assert client.post("/api/generation/notes/..", json={"notes": "x"}).status_code == 404
