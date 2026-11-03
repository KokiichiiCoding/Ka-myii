"""Failures and cancellations must not destroy a user's finished 3D avatar."""
import json
from pathlib import Path
import pytest
from PIL import Image

from src.core import rig3d
from src.pipeline.assembly_line import AssemblyLine, _replace_avatar


def test_previous_avatar_restored_if_final_rename_fails(tmp_path, monkeypatch):
    final = tmp_path / "avatar3d"
    work = tmp_path / "avatar3d.new"
    final.mkdir(); work.mkdir()
    (final / "avatar.vrm").write_bytes(b"original")
    (work / "avatar.vrm").write_bytes(b"new")
    real_rename = Path.rename
    def rename(path, target):
        if path == work:
            raise PermissionError("locked destination")
        return real_rename(path, target)
    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(PermissionError):
        _replace_avatar(work, final)
    assert (final / "avatar.vrm").read_bytes() == b"original"
    assert (work / "avatar.vrm").read_bytes() == b"new"
    assert not list(tmp_path.glob("avatar3d.backup-*"))


def test_cancel_after_rigging_preserves_old_avatar(tmp_path, monkeypatch):
    model = tmp_path / "char"; model.mkdir()
    Image.new("RGB", (64, 96)).save(model / "base_image.png")
    (model / "metadata.json").write_text(json.dumps({"name":"Test","metadata":{}}), encoding="utf-8")
    final = model / "avatar3d"; final.mkdir()
    (final / "avatar.vrm").write_bytes(b"original")
    class Manager:
        cancelled = False
        def is_cancelled(self, task_id): return self.cancelled
        def set_stage(self, task_id, stage): pass
    manager = Manager()
    monkeypatch.setattr("src.utils.progress_tracker.get_progress_manager", lambda: manager)
    monkeypatch.setattr("src.pipeline.assembly_line.get_progress_manager", lambda: manager)
    def demo(model_dir, work, name):
        work.mkdir()
        (work / "avatar.vrm").write_bytes(b"new")
        manager.cancelled = True
        return {}
    monkeypatch.setattr(rig3d, "demo", demo)
    line = AssemblyLine(output_base_dir=tmp_path, use_dummy_generators=True)
    with pytest.raises(InterruptedError):
        line.make_avatar3d("char", task_id="job")
    assert (final / "avatar.vrm").read_bytes() == b"original"
    assert not (model / "avatar3d.new").exists()


def test_refresh_rig_reuses_mesh_without_gpu_or_changing_source(tmp_path, monkeypatch):
    model = tmp_path / "char"; model.mkdir()
    Image.new("RGB", (64, 96)).save(model / "base_image.png")
    (model / "metadata.json").write_text(json.dumps({"name":"Test","metadata":{
        "avatar3d":{"quality":"high","seed":42}}}), encoding="utf-8")
    original = model / "avatar3d"; original.mkdir()
    for name, value in {"mesh.glb":b"mesh", "camera.json":b"camera", "crop.png":b"crop", "avatar.vrm":b"old"}.items():
        (original / name).write_bytes(value)
    def build(model_dir, work, name):
        assert model_dir == model and (work / "mesh.glb").read_bytes() == b"mesh"
        assert (original / "avatar.vrm").read_bytes() == b"old"
        (work / "avatar.vrm").write_bytes(b"updated")
        return {"expressions":["blink"]}
    monkeypatch.setattr(rig3d, "build", build)
    line = AssemblyLine(output_base_dir=tmp_path, use_dummy_generators=True)
    line.make_avatar3d("char", options={"reuse_mesh":True})
    assert (original / "avatar.vrm").read_bytes() == b"updated"
    assert (original / "mesh.glb").read_bytes() == b"mesh"
    saved = json.loads((model / "metadata.json").read_text())["metadata"]["avatar3d"]
    assert saved["seed"] == 42 and saved["quality"] == "high" and saved["expressions"] == ["blink"]
    assert not list(model.glob("avatar3d.backup-*"))


@pytest.fixture
def api(tmp_path, monkeypatch):
    import app as app_module
    import config
    from src.api import generation
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    model = config.OUTPUT_DIR / "abc-123"; model.mkdir(parents=True)
    Image.new("RGB", (64, 96)).save(model / "base_image.png")
    (model / "metadata.json").write_text(json.dumps({"name":"Test","metadata":{}}), encoding="utf-8")
    application = app_module.create_app(use_dummy_generators=True)
    submitted=[]
    monkeypatch.setattr(generation.job_queue, "submit", lambda req: submitted.append(req) or type("J", (), {"id":"j"})())
    monkeypatch.setattr(generation.job_queue, "describe", lambda job: {"id":"j"})
    client=application.test_client();client.dir=model;client.submitted=submitted
    return client


def test_refresh_endpoint_validates_availability(api, monkeypatch):
    from src.api import generation
    assert api.post("/api/generation/avatar3d/abc-123", json={"reuse_mesh": True}).status_code == 400
    assert api.post("/api/generation/avatar3d/abc-123", json={"reuse_mesh": "yes"}).status_code == 400
    source=api.dir / "avatar3d"; source.mkdir()
    for name in ("mesh.glb", "camera.json", "crop.png"):
        (source / name).write_bytes(b"fixture")
    r = api.post("/api/generation/avatar3d/abc-123", json={"reuse_mesh": True})
    assert r.status_code == 202 and api.submitted[-1].avatar3d["reuse_mesh"] is True
    r = api.post("/api/generation/generate", json={"prompt":"1girl", "avatar3d":{"reuse_mesh":True}})
    assert r.status_code == 400


def test_failed_rollback_leaves_recoverable_backup(tmp_path, monkeypatch):
    final = tmp_path / "avatar3d"; work = tmp_path / "avatar3d.new"
    final.mkdir();work.mkdir()
    (final / "avatar.vrm").write_bytes(b"original")
    real_rename = Path.rename
    def rename(path, target):
        if path == work or path.name.startswith("avatar3d.backup-"):
            raise PermissionError("destination locked")
        return real_rename(path, target)
    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(RuntimeError, match="preserved at"):
        _replace_avatar(work, final)
    backups = list(tmp_path.glob("avatar3d.backup-*"))
    assert len(backups) == 1 and (backups[0] / "avatar.vrm").read_bytes() == b"original"
