"""GPU-free checks for private 3D engine ownership and recovery."""
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.core import mesh3d


def test_port_preflight_rejects_even_an_unrelated_listener():
    # An unrelated HTTP service could return a 404 at /system_stats: a bind, not an
    # endpoint-specific probe, must reject the port before any child is started.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        with pytest.raises(RuntimeError, match="already in use"):
            mesh3d._assert_port_free(listener.getsockname()[1])
    mesh3d._assert_port_free(0)


def test_wait_ready_never_queries_a_stranger_before_the_child_binds(monkeypatch):
    engine = mesh3d.Engine()
    engine.proc = SimpleNamespace(poll=lambda: None)
    engine.tail[:] = ["Starting server"]
    events = []
    def wait(_):
        events.append("waiting")
        engine.bound.set()
    def get(path, timeout):
        assert events == ["waiting"], "Do not query an unknown listener before our child binds"
        assert path == "/system_stats"
        return {"devices": [{"type": "cuda"}]}
    monkeypatch.setattr(mesh3d.time, "sleep", wait)
    monkeypatch.setattr(engine, "_get", get)
    engine._wait_ready(timeout=3)


def test_history_poll_remains_cancellable_while_engine_is_silent(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    engine = mesh3d.Engine(progress=lambda frac, label: (_ for _ in ()).throw(InterruptedError("cancelled")))
    engine.proc = SimpleNamespace(poll=lambda: None)
    def blocked(path, timeout):
        entered.set()
        assert release.wait(4), "The test must release the simulated slow request"
        return {"job": {"outputs": {}}}
    monkeypatch.setattr(engine, "_get", blocked)
    try:
        with pytest.raises(InterruptedError, match="cancelled"):
            engine._history("job", .4, "Sculpting")
        assert entered.is_set()
    finally:
        release.set()


def test_history_returns_response_and_propagates_errors(monkeypatch):
    engine = mesh3d.Engine()
    engine.proc = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(engine, "_get", lambda path, timeout: {"job": {"outputs": {"a": 1}}})
    assert engine._history("job", 0, "Starting")["job"]["outputs"] == {"a": 1}
    def broken(path, timeout):
        raise ValueError("invalid engine response")
    monkeypatch.setattr(engine, "_get", broken)
    with pytest.raises(ValueError, match="invalid engine response"):
        engine._history("job", 0, "Starting")


def test_output_location_cannot_escape_private_engine(monkeypatch, tmp_path):
    engine = tmp_path / "engine"
    root = engine / "data" / "output" / "kamyii"
    root.mkdir(parents=True)
    model = root / "mesh_00001_.glb"
    model.write_bytes(b"glTF")
    monkeypatch.setattr(mesh3d, "engine_dir", lambda: engine)
    assert mesh3d._model_output({"type": "output", "subfolder": "kamyii", "filename": model.name}) == model
    outside = tmp_path / "owner.glb"
    outside.write_bytes(b"owner data")
    for info in [
        {"type": "input", "subfolder": "kamyii", "filename": model.name},
        {"type": "output", "subfolder": "../../..", "filename": "owner.glb"},
        {"type": "output", "subfolder": "kamyii", "filename": "../owner.glb"},
        {"type": "output", "subfolder": "kamyii", "filename": "..\\owner.glb"},
        {"type": "output", "subfolder": "kamyii", "filename": "missing.glb"},
        {"type": "output", "subfolder": "kamyii", "filename": "not-model.txt"},
    ]:
        with pytest.raises(RuntimeError, match="invalid"):
            mesh3d._model_output(info)
    assert outside.read_bytes() == b"owner data" and model.exists()


def test_invalid_quality_removes_staged_inputs(monkeypatch, tmp_path):
    engine = tmp_path / "engine"
    monkeypatch.setattr(mesh3d, "engine_dir", lambda: engine)
    monkeypatch.setattr(mesh3d, "available", lambda: True)
    image = Image.new("RGB", (20, 20), "red")
    with pytest.raises(ValueError, match="quality"):
        mesh3d.generate(image, np.ones((20, 20), bool), tmp_path / "result", 7, quality="unknown")
    assert not list((engine / "data" / "input" / "kamyii").glob("*.png"))


def test_prompt_submission_remains_cancellable(monkeypatch):
    import requests
    entered, release = threading.Event(), threading.Event()
    engine = mesh3d.Engine(progress=lambda frac, label: (_ for _ in ()).throw(InterruptedError("cancelled")))
    engine.proc = SimpleNamespace(poll=lambda: None)
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(4)
        return SimpleNamespace(status_code=200, json=lambda: {"prompt_id": "p"})
    monkeypatch.setattr(requests, "post", blocked)
    try:
        with pytest.raises(InterruptedError, match="cancelled"):
            engine.run({})
        assert entered.is_set()
    finally:
        release.set()


def test_stopped_engine_is_reaped_even_if_it_exited_before_kill():
    class Process:
        alive = True
        waited = False
        def poll(self):
            return None if self.alive else 1
        def kill(self):
            self.alive = False
            raise OSError("already exited")
        def wait(self):
            self.waited = True
    engine = mesh3d.Engine()
    proc = engine.proc = Process()
    mesh3d._live.add(proc)
    engine.__exit__()
    assert proc.waited and proc not in mesh3d._live and engine.proc is None


def test_partial_input_copy_failure_removes_first_staged_file(monkeypatch, tmp_path):
    engine = tmp_path / "engine"
    monkeypatch.setattr(mesh3d, "engine_dir", lambda: engine)
    monkeypatch.setattr(mesh3d, "available", lambda: True)
    original = mesh3d.shutil.copyfile
    def copy(source, target):
        if str(target).endswith("_mask.png"):
            raise OSError("disk write failed")
        return original(source, target)
    monkeypatch.setattr(mesh3d.shutil, "copyfile", copy)
    with pytest.raises(OSError, match="disk write failed"):
        mesh3d.generate(Image.new("RGB", (20, 20), "red"), np.ones((20, 20), bool),
                        tmp_path / "result", 7)
    assert not list((engine / "data" / "input" / "kamyii").glob("*.png"))


def test_generate_accepts_own_saved_glb_and_cleans_inputs(monkeypatch, tmp_path):
    engine_root = tmp_path / "engine"
    monkeypatch.setattr(mesh3d, "engine_dir", lambda: engine_root)
    monkeypatch.setattr(mesh3d, "available", lambda: True)
    class FakeEngine:
        def __init__(self, progress, log_path):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def run(self, prompt):
            ids = {node["class_type"]: key for key, node in prompt.items()}
            inputs = prompt[ids["LoadImage"]]["inputs"]["image"]
            assert (engine_root / "data" / "input" / inputs).is_file()
            folder = engine_root / "data" / "output" / "kamyii"
            folder.mkdir(parents=True)
            (folder / "mesh_00001_.glb").write_bytes(b"glTF fixture")
            return {ids["SaveGLB"]: {"3d": [{"type": "output", "subfolder": "kamyii", "filename": "mesh_00001_.glb"}]},
                    ids["PreviewAny"]: {"text": ["40"]},
                    ids["KamyiiProjectImage"]: {"text": ['{"score": 0.95}']}}
    monkeypatch.setattr(mesh3d, "Engine", FakeEngine)
    result = tmp_path / "result"
    camera = mesh3d.generate(Image.new("RGB", (20, 20), "red"), np.ones((20, 20), bool), result, 7)
    assert (result / "mesh.glb").read_bytes() == b"glTF fixture"
    assert camera["fov"] == 40 and camera["paint"]["score"] == .95
    assert not list((engine_root / "data" / "input" / "kamyii").glob("*.png"))


def test_engine_remembers_bind_after_noisy_startup(tmp_path, monkeypatch):
    engine = mesh3d.Engine(log_path=tmp_path / "engine.log")
    class Output:
        def __init__(self):
            self.chunks = [("To see the GUI go to: " + engine.url + "\n" + "other startup line\n" * 100).encode(), b""]
        def read1(self, size):
            return self.chunks.pop(0)
    engine.proc = SimpleNamespace(stdout=Output(), poll=lambda: None)
    engine._pump()
    assert engine.bound.is_set() and not any("To see the GUI" in line for line in engine.tail)
    monkeypatch.setattr(engine, "_get", lambda path, timeout: {"devices":[{"type":"cuda"}]})
    engine._wait_ready(timeout=1)
