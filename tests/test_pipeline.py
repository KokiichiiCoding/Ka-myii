"""
End-to-end smoke tests for the Ka-myii pipeline.

These run entirely in demo mode (no GPU / model weights required) and validate
that generation, decomposition and the rig-ready Live2D/PSD packaging all work.
"""
import json
from pathlib import Path

import pytest

from src.models.vtuber_model import GenerationRequest


@pytest.fixture
def workdir(tmp_path):
    return tmp_path


def test_demo_generation_and_decomposition(workdir):
    from src.core.image_generator import DummyImageGenerator
    from src.core.asset_separator import AssetSeparator

    gen = DummyImageGenerator()
    req = GenerationRequest(prompt="1girl, blue hair, green eyes", width=384, height=512, steps=4)
    base = gen.generate(req, workdir / "base_image.png")
    assert base.exists()
    assert (workdir / "base_image.regions.png").exists(), "region-map sidecar should be written"

    assets = AssetSeparator().separate(base, workdir / "layers")
    layer_types = {a.layer_type for a in assets}
    # Core riggable parts must be present and non-empty.
    for required in ("hair_back", "head", "eye_L", "eye_R", "mouth", "hair_front"):
        assert required in layer_types, f"missing layer: {required}"
    for a in assets:
        if a.layer_type != "character":
            assert a.file_path.exists()


def test_full_pipeline_produces_rig_ready_package(workdir):
    from src.pipeline.assembly_line import AssemblyLine

    al = AssemblyLine(output_base_dir=workdir, use_dummy_generators=True)
    req = GenerationRequest(prompt="vtuber, purple hair, sailor uniform", width=448, height=640, steps=4)
    model = al.generate_model(req, task_id="test")

    assert model.status.value == "completed"
    assert len(model.metadata.get("layers", [])) >= 10

    model_dir = model.final_model_path
    name = model.name
    # Cubism-ready scaffold
    assert (model_dir / f"{name}.model3.json").exists()
    assert (model_dir / f"{name}.physics3.json").exists()
    assert (model_dir / f"{name}.cdi3.json").exists()
    assert (model_dir / "RIGGING_GUIDE.md").exists()

    # Layered artifacts
    artifacts = model.metadata.get("artifacts", {})
    assert "ora" in artifacts and Path(artifacts["ora"]).exists()
    # cdi3 should carry the standard parameter set and a part per layer
    cdi = json.loads((model_dir / f"{name}.cdi3.json").read_text())
    param_ids = {p["Id"] for p in cdi["Parameters"]}
    for required in ("ParamEyeLOpen", "ParamMouthOpenY", "ParamAngleX"):
        assert required in param_ids
    assert len(cdi["Parts"]) >= 10


def test_psd_export_roundtrip(workdir):
    from src.core.image_generator import DummyImageGenerator
    from src.core.asset_separator import AssetSeparator
    from src.core import psd_exporter

    gen = DummyImageGenerator()
    req = GenerationRequest(prompt="1girl, red hair", width=384, height=512, steps=2)
    base = gen.generate(req, workdir / "base_image.png")
    assets = AssetSeparator().separate(base, workdir / "layers")

    produced = psd_exporter.export_layered_documents(assets, workdir / "export", "unit_model")
    assert "ora" in produced
    # PSD is best-effort (pytoshop); if present it must be a non-trivial file.
    if "psd" in produced:
        assert Path(produced["psd"]).stat().st_size > 1000


def test_app_boots_in_demo_mode():
    import app as app_module

    flask_app = app_module.create_app(use_dummy_generators=True)
    client = flask_app.test_client()
    assert client.get("/health").status_code == 200
    assert client.get("/generator").status_code == 200
    opts = client.get("/api/generation/options").get_json()
    assert opts["success"] and len(opts["samplers"]) > 0
