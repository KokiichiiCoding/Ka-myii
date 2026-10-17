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


def test_fill_holes_only_fills_interior():
    import numpy as np
    from src.core.asset_separator import AssetSeparator

    ring = np.zeros((10, 10), bool)
    ring[2:8, 2:8] = True
    ring[4:6, 4:6] = False
    assert AssetSeparator._fill_holes(ring).sum() == 36  # 6x6 square, nothing outside it


def test_seethrough_layers_map_back_to_source_canvas(workdir, monkeypatch):
    """See-through pads the input to a centred square and resizes it; layers must land back on the source pixels."""
    import numpy as np
    from PIL import Image
    from psd_tools import PSDImage
    from src.core import seethrough
    from src.core.asset_separator import AssetSeparator

    src = workdir / "portrait.png"
    portrait = Image.new("RGB", (100, 200), "white")  # w=100, h=200 -> square side 200, x-pad 50
    portrait.paste((40, 40, 40), (5, 25, 30, 50))  # the "character": dark patch on white background
    portrait.save(src)

    # Fake See-through output at resolution 400 (scale 2): a 20x20 block standing for source px (10..19, 30..39).
    psd = PSDImage.new(mode="RGBA", size=(400, 400), depth=8)
    block = Image.new("RGBA", (20, 20), (40, 40, 40, 255))  # a real part reproduces the source pixels
    psd.create_pixel_layer(block, name="front hair", top=30 * 2, left=(10 + 50) * 2, opacity=255)
    # ...and a hallucinated part floating over empty background (source px 60..79, 150..169) -> must be dropped.
    psd.create_pixel_layer(Image.new("RGBA", (40, 40), (0, 0, 255, 255)), name="headwear", top=150 * 2, left=(60 + 50) * 2, opacity=255)
    psd_path = workdir / "fake.psd"
    psd.save(psd_path)

    monkeypatch.setattr(seethrough, "available", lambda: True)
    monkeypatch.setattr(seethrough, "decompose", lambda *a, **k: psd_path)
    assets = AssetSeparator().separate(src, workdir / "layers")

    hair = next(a for a in assets if a.layer_type == "front hair")
    alpha = np.asarray(Image.open(hair.file_path))[..., 3]
    assert alpha.shape == (200, 100)
    assert hair.metadata["bbox"] == [10, 30, 19, 39]
    assert "headwear" not in {a.layer_type for a in assets}


def test_full_pipeline_produces_rig_ready_package(workdir):
    from src.pipeline.assembly_line import AssemblyLine

    al = AssemblyLine(output_base_dir=workdir, use_dummy_generators=True)
    req = GenerationRequest(prompt="vtuber, purple hair, sailor uniform", width=448, height=640, steps=4)
    model = al.generate_model(req, task_id="test")

    assert model.status.value == "completed"
    assert isinstance(model.request.seed, int) and model.request.seed >= 0
    assert model.to_dict()["request"]["seed"] == model.request.seed
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


def test_long_prompts_are_chunked_at_commas_not_truncated():
    from src.core.image_generator import chunk_token_ids

    COMMA = 267
    ids = list(range(1000, 1070)) + [COMMA] + list(range(2000, 2030))  # comma at index 70, 101 tokens
    chunks = chunk_token_ids(ids, COMMA)
    assert [len(c) for c in chunks] == [71, 30] and chunks[0][-1] == COMMA  # cut after the comma
    assert sum(chunks, []) == ids  # nothing dropped
    no_comma = list(range(160))
    assert [len(c) for c in chunk_token_ids(no_comma, COMMA)] == [75, 75, 10]
    assert chunk_token_ids([], COMMA) == [[]]


def test_seethrough_order_priors_keep_real_parts_and_leftovers(workdir, monkeypatch):
    """Real run regressions: See-through stacked an iris under its eye white (the colour check then
    deleted the eye white), emitted few-pixel accessory specks and assigned cat ears to no layer."""
    import numpy as np
    from PIL import Image
    from psd_tools import PSDImage
    from src.core import seethrough
    from src.core.asset_separator import AssetSeparator, apply_order_priors

    assert apply_order_priors(["topwear", "neck", "irides-l", "eyewhite-r", "eyewhite-l", "irides-r"]) == \
        ["neck", "topwear", "eyewhite-r", "eyewhite-l", "irides-l", "irides-r"]
    # Seen on a real run: bangs and a hat under the face (the bangs got baked into the face layer).
    assert apply_order_priors(["front hair", "headwear", "face", "eyebrow-l", "eyelash-l"]) == \
        ["face", "eyebrow-l", "eyelash-l", "front hair", "headwear"]

    src = workdir / "portrait.png"
    portrait = Image.new("RGB", (100, 200), "white")  # square side 200, x-pad 50
    portrait.paste((40, 40, 40), (0, 20, 60, 100))      # character silhouette
    portrait.paste((255, 255, 255), (10, 30, 30, 40))   # eye white
    portrait.paste((200, 150, 0), (12, 30, 28, 40))     # iris on top of it
    portrait.save(src)

    psd = PSDImage.new(mode="RGBA", size=(200, 200), depth=8)
    layer = lambda size, colour, name, x, y: psd.create_pixel_layer(  # noqa: E731
        Image.new("RGBA", size, colour), name=name, top=y, left=x + 50, opacity=255)
    layer((16, 10), (200, 150, 0, 255), "irides-l", 12, 30)     # wrongly below...
    layer((20, 10), (255, 255, 255, 255), "eyewhite-l", 10, 30)  # ...its eye white
    layer((2, 2), (90, 90, 90, 255), "eyewear", 40, 60)          # noise speck
    layer((60, 25), (40, 40, 40, 255), "face", 0, 20)
    layer((50, 20), (40, 40, 40, 255), "nose", 5, 22)            # degenerate: a copy of the face
    psd_path = workdir / "fake.psd"
    psd.save(psd_path)

    monkeypatch.setattr(seethrough, "available", lambda: True)
    monkeypatch.setattr(seethrough, "decompose", lambda *a, **k: psd_path)
    assets = {a.layer_type: a for a in AssetSeparator().separate(src, workdir / "layers")}

    assert "eyewear" not in assets and "nose" not in assets and "face" in assets
    assert assets["eyewhite-l"].metadata["order"] < assets["irides-l"].metadata["order"]
    leftover = np.asarray(Image.open(assets["leftover"].file_path))
    assert assets["leftover"].metadata["order"] == 0
    assert leftover[80, 30, 3] > 128 and tuple(leftover[80, 30, :3]) == (40, 40, 40)  # uncovered body kept
    assert leftover[35, 20, 3] == 0  # the eye belongs to its own layers


def test_chroma_cutout_handles_upper_body_crops():
    """Upper-body framing puts clothing on the border; the median border colour then matched
    nothing and the whole image (white background included) counted as character."""
    import numpy as np
    from PIL import Image
    from src.core.asset_separator import AssetSeparator

    img = Image.new("RGB", (100, 120), "white")
    img.paste((200, 30, 30), (0, 70, 100, 120))   # hoodie across the whole bottom border
    img.paste((40, 30, 30), (30, 10, 70, 70))     # head
    img.paste((255, 255, 255), (45, 30, 50, 35))  # eye highlight, same white as the background
    alpha = np.asarray(AssetSeparator._chroma_cutout(img.convert("RGBA")).split()[-1])
    assert alpha[5, 5] == 0 and alpha[40, 90] == 0  # background removed
    assert alpha[100, 50] == 255 and alpha[20, 50] == 255  # hoodie and head kept
    assert alpha[32, 47] == 255  # enclosed white highlight stays opaque


def test_ownership_cuts_overextended_parts_and_background_spill():
    """See-through predicted bangs across the cheeks; with the hair on top, the cheeks became
    skin-coloured patches in the hair layer with hair-coloured outlines. Each pixel now goes to
    the part whose prediction matches the source; parts stacked above it are cut there."""
    import numpy as np
    from PIL import Image
    from src.core.asset_separator import AssetSeparator

    skin, hair = (240, 200, 180), (150, 150, 170)
    src = np.zeros((20, 20, 3), np.int16); src[:] = skin; src[:5] = hair  # bangs only on rows 0-4
    face = Image.new("RGBA", (20, 20), skin + (255,))
    bangs = Image.new("RGBA", (20, 20), (0, 0, 0, 0)); bangs.paste(hair + (255,), (0, 0, 20, 12))  # over-extended
    blob = Image.new("RGBA", (20, 20), (0, 0, 0, 0)); blob.paste((250, 250, 250, 255), (16, 16, 20, 20))
    kept = [("face", face), ("front hair", bangs), ("headwear", blob)]
    background = np.zeros((20, 20), bool); background[15:, 15:] = True

    covered = AssetSeparator._resolve_ownership(kept, src, background)
    hair_alpha = np.asarray(bangs)[..., 3]
    assert (hair_alpha[:5] == 255).all() and (hair_alpha[6:] == 0).all()  # cut back to the real bangs
    assert np.asarray(blob)[..., 3].max() == 0  # spill over trusted background cleared
    assert tuple(np.asarray(face)[10, 10, :3]) == skin and covered[10, 10] and not covered[18, 18]


def test_leftover_above_the_chin_becomes_headwear():
    """Real runs left a witch hat / cat ears in `leftover`; the rig moves a layer with either
    the head or the body, so ears plus hoodie scraps in one layer tore when the head turned."""
    import numpy as np
    from PIL import Image
    from src.core.asset_separator import AssetSeparator

    face = Image.new("RGBA", (100, 100), (0, 0, 0, 0)); face.paste((240, 200, 180, 255), (30, 20, 70, 60))
    leftover = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    leftover.paste((90, 60, 40, 255), (30, 0, 70, 15))   # cat ears above the face
    leftover.paste((200, 30, 30, 255), (10, 80, 90, 95))  # hoodie scrap below the chin

    pieces = dict(AssetSeparator._split_leftover(leftover, [("face", face)]))
    assert set(pieces) == {"headwear", "leftover"}
    ears, scrap = (np.asarray(pieces[n])[..., 3] for n in ("headwear", "leftover"))
    assert ears[5, 50] == 255 and ears[85, 50] == 0 and scrap[85, 50] == 255 and scrap[5, 50] == 0
    assert [n for n, _ in AssetSeparator._split_leftover(leftover, [])] == ["leftover"]  # no face: no chin

    # Placement: leftover above the back hair; above-chin pieces join a real headwear layer.
    hat = Image.new("RGBA", (100, 100), (0, 0, 0, 0)); hat.paste((20, 20, 120, 255), (0, 0, 20, 10))
    kept = [("back hair", Image.new("RGBA", (100, 100))), ("face", face), ("headwear", hat)]
    AssetSeparator._place_leftover(kept, leftover)
    assert [n for n, _ in kept] == ["back hair", "leftover", "face", "headwear"]
    merged = np.asarray(kept[-1][1])[..., 3]
    assert merged[5, 5] == 255 and merged[5, 50] == 255 and merged[85, 50] == 0  # stars + ears


def test_cubism_files_follow_the_specs_and_drive_declared_parameters(workdir):
    """physics3.json lacked Meta.EffectiveForces / PhysicsDictionary (required by Live2D's schema)
    and drove ids like "ParamFront Hair" that no file declared; human ears got accessory physics."""
    import re
    from PIL import Image
    from src.core.model_assembler import ModelAssembler
    from src.models.vtuber_model import Asset

    assets = []
    for i, name in enumerate(["back hair", "face", "ears-l", "front hair", "headwear"]):
        path = workdir / f"{i:02d}_{name}.png"
        Image.new("RGBA", (64, 64), (200, 100, 50, 255)).save(path)
        assets.append(Asset(layer_type=name, file_path=path, metadata={"order": i}))
    model_dir = ModelAssembler().assemble(assets, workdir / "model", "study")

    physics = json.loads((model_dir / "study.physics3.json").read_text())
    cdi = json.loads((model_dir / "study.cdi3.json").read_text())
    meta, settings = physics["Meta"], physics["PhysicsSettings"]
    for key in ("PhysicsSettingCount", "TotalInputCount", "TotalOutputCount", "VertexCount",
                "EffectiveForces", "PhysicsDictionary"):
        assert key in meta
    assert meta["PhysicsSettingCount"] == len(settings) == len(meta["PhysicsDictionary"]) == 3
    assert meta["TotalInputCount"] == sum(len(s["Input"]) for s in settings)
    assert meta["VertexCount"] == sum(len(s["Vertices"]) for s in settings)

    declared = {p["Id"] for p in cdi["Parameters"]}
    used = {i["Source"]["Id"] for s in settings for i in s["Input"]} | \
           {o["Destination"]["Id"] for s in settings for o in s["Output"]}
    assert used <= declared and all(re.fullmatch(r"[A-Za-z0-9_]+", p) for p in declared)
    assert {o["Destination"]["Id"] for s in settings for o in s["Output"]} == \
        {"ParamHairBack", "ParamHairFront", "ParamAccessorySway"}  # no physics for the real ears
