"""Gallery can refresh a 3D rig from its saved mesh without another mesh generation."""
import json
import os
import threading
import pytest
from PIL import Image

sync = pytest.importorskip("playwright.sync_api")


def test_gallery_refreshes_existing_mesh(tmp_path, monkeypatch):
    import config
    import app
    from src.core import rig3d
    from werkzeug.serving import make_server
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    model = config.OUTPUT_DIR / "refresh-model"; model.mkdir(parents=True)
    Image.new("RGB", (64, 96)).save(model / "base_image.png")
    meta = {"id":"refresh-model","name":"model_12345678","status":"completed", "created_at":"2026-10-28",
            "request":{"prompt":"a character"},"assets":[],"metadata":{"avatar3d":{"quality":"high","seed":3}}}
    (model / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    old = model / "avatar3d"; old.mkdir()
    for name in ("mesh.glb", "camera.json", "crop.png", "avatar.vrm"):
        (old / name).write_bytes(b"old:" + name.encode())
    def build(model_dir, work, name):
        assert (work / "mesh.glb").read_bytes() == b"old:mesh.glb"
        (work / "avatar.vrm").write_bytes(b"new vrm")
        return {"expressions":["blink"]}
    monkeypatch.setattr(rig3d, "build", build)
    server = make_server("127.0.0.1", 0, app.create_app(use_dummy_generators=True), threaded=True)
    thread=threading.Thread(target=server.serve_forever, daemon=True);thread.start()
    try:
        with sync.sync_playwright() as p:
            browser=p.chromium.launch(channel=os.environ.get("KAMYII_E2E_CHANNEL") or None,
                ignore_default_args=["--headless=old"], args=["--headless=new", "--enable-unsafe-swiftshader"])
            page=browser.new_page();errors=[];page.on("pageerror",lambda e:errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{server.server_port}/gallery")
            card=page.locator(".model-card").first
            card.locator("summary").click()
            refresh=card.get_by_role("button",name="Refresh rig & expressions")
            sync.expect(refresh).to_be_visible()
            with page.expect_request(lambda r: "/avatar3d/refresh-model" in r.url and r.method=="POST") as req:
                refresh.click()
            assert req.value.post_data_json == {"reuse_mesh":True}
            page.wait_for_url("**/generator?model=refresh-model")
            sync.expect(page.locator("#open3d")).to_be_visible(timeout=60000)
            assert (old / "avatar.vrm").read_bytes() == b"new vrm"
            assert (old / "mesh.glb").read_bytes() == b"old:mesh.glb"
            assert not errors
            browser.close()
    finally:
        server.shutdown()
