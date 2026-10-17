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
