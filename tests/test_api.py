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
    with Image.open(req.reference_image) as saved:
        assert saved.mode == "RGB" and saved.getpixel((5, 5)) == (255, 255, 255)  # transparency → white

    for data, word in [({}, "Choose"), ({"image": (io.BytesIO(b"not an image"), "x.png")}, "not an image"),
                       ({"image": (png((100, 100), "RGB"), "tiny.png")}, "too small")]:
        r = client.post("/api/generation/upload", data=data, content_type="multipart/form-data")
        assert r.status_code == 400 and word in r.get_json()["error"]
    assert len(client.submitted) == 1


def test_uploaded_artwork_runs_the_pipeline_without_sdxl(tmp_path):
    from PIL import Image
    from src.models.vtuber_model import GenerationRequest
    from src.pipeline.assembly_line import AssemblyLine

    art = tmp_path / "upload.png"
    Image.new("RGB", (320, 480), "white").save(art)
    line = AssemblyLine(output_base_dir=tmp_path / "out", use_dummy_generators=True)
    line.image_generator.generate = None  # must not be called
    model = line.generate_model(GenerationRequest(prompt="Uploaded artwork: oc", reference_image=str(art)))
    assert model.status.value == "completed" and not art.exists()
    assert (tmp_path / "out" / model.id / "base_image.png").exists()
    assert model.request.reference_image == "upload" and model.request.seed is None
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
    (folder / "metadata.json").write_text(json.dumps({"status": "failed", "request": {"prompt": "1girl, witch"}}))
    r = client.post("/api/generation/retry/abc-123")
    assert r.status_code == 202
    req = client.submitted[-1]
    assert req.prompt == "1girl, witch" and req.reference_image and (req.width, req.height) == (400, 600)
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
