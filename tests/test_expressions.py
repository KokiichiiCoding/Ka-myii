"""
Painted expression layers (closed eyes, the other mouth) and where they land in Live Preview.
GPU-free: the demo generator's inpaint stands in for SDXL, a stub stands in for the tagger.
"""
import numpy as np
from PIL import Image, ImageDraw


def _character(tmp_path):
    """A face with two eyes (white, iris, lashes), a closed mouth and bangs over the left eye."""
    model = tmp_path / "model"
    assets = model / "assets"
    assets.mkdir(parents=True)
    size = (200, 260)
    base = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(base)
    draw.ellipse((40, 40, 160, 200), fill=(240, 200, 180))
    layers = {}

    def layer(name, box, colour, shape="rect"):
        im = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        (d.ellipse if shape == "ellipse" else d.rectangle)(box, fill=colour + (255,))
        layers[name] = im
        draw_base = ImageDraw.Draw(base)
        (draw_base.ellipse if shape == "ellipse" else draw_base.rectangle)(box, fill=colour)

    layer("face", (40, 40, 160, 200), (240, 200, 180), "ellipse")
    for side, x in (("r", 65), ("l", 115)):
        layer(f"eyewhite-{side}", (x, 100, x + 25, 115), (250, 250, 250), "ellipse")
        layer(f"irides-{side}", (x + 7, 100, x + 18, 115), (120, 60, 160), "ellipse")
        layer(f"eyelash-{side}", (x - 2, 97, x + 27, 100), (40, 30, 30))
    layer("mouth", (90, 160, 110, 163), (150, 60, 60))
    layer("front hair", (110, 60, 150, 105), (200, 200, 220))  # bangs over the top of the left eye
    for i, (name, im) in enumerate(layers.items()):
        im.save(assets / f"{i:02d}_{name}.png")
    base.save(model / "base_image.png")
    return model


def test_expressions_paint_closed_eyes_and_open_mouth(tmp_path):
    from src.core import expressions
    from src.core.image_generator import DummyImageGenerator

    model = _character(tmp_path)
    made = expressions.generate(DummyImageGenerator(), model, tag_image=lambda image: None)
    assert made["mouth_drawn"] == "closed" and set(made) >= {"eye_close", "mouth_open"}
    eye = np.asarray(Image.open(model / made["eye_close"]))[..., 3]
    mouth = np.asarray(Image.open(model / made["mouth_open"]))[..., 3]
    assert eye[107, 77] > 200 and eye[107, 127] < 255  # right eye fully covered; bangs cut into the left
    assert eye[96, 125] <= 2  # where the bangs are drawn on top, the closed eye stays out
    assert mouth[165, 100] > 200 and mouth[107, 77] == 0
    assert eye[30, 100] == 0 and mouth[230, 100] == 0  # nothing outside the face area


def test_preview_places_expressions_where_the_rig_expects_them(tmp_path):
    from src.core import expressions
    from src.core.image_generator import DummyImageGenerator
    from src.core.live_preview import preview_layers

    model = _character(tmp_path)
    expressions.generate(DummyImageGenerator(), model, tag_image=lambda image: None)
    names = [name for name, _ in preview_layers(model / "assets")]
    # Closed eyes above the eye parts but under the bangs; the drawn mouth becomes the closed one.
    assert names.index("eyelash") < names.index("eye_close") < names.index("front hair")
    assert names.index("mouth_close") < names.index("mouth_open") and "mouth" not in names


def test_expression_layers_are_hidden_in_the_artist_documents(tmp_path):
    from psd_tools import PSDImage
    from src.core import expressions, psd_exporter
    from src.core.image_generator import DummyImageGenerator
    from src.models.vtuber_model import Asset

    model = _character(tmp_path)
    made = expressions.generate(DummyImageGenerator(), model, tag_image=lambda image: None)
    assets = [Asset(layer_type=p.stem.split("_", 1)[1], file_path=p, metadata={"order": i, "backend": "see-through"})
              for i, p in enumerate(sorted((model / "assets").glob("*.png")))]
    assets += expressions.as_assets(model, made, assets)
    produced = psd_exporter.export_layered_documents(assets, tmp_path / "doc", "study")
    psd = PSDImage.open(produced["psd"])
    visible = {layer.name: layer.visible for layer in psd}
    assert visible["Eye Close"] is False and visible["Mouth Open"] is False and visible["Face"] is True
    flat = np.asarray(Image.open(produced["flat"]))
    assert tuple(flat[107, 77, :3]) == (120, 60, 160) or flat[107, 77, :3].tolist() != [0, 0, 0]  # eyes open


def test_toggles_are_painted_on_request_and_switch_in_the_preview(tmp_path):
    from src.core import expressions
    from src.core.image_generator import DummyImageGenerator
    from src.core.live_preview import preview_layers

    model = _character(tmp_path)
    layers = expressions.load_layers(model / "assets")
    cheeks = expressions.toggle_mask(layers, "cheeks")
    assert cheeks[125, 77] and cheeks[125, 127] and not cheeks[107, 77]   # under each eye, not on it
    temple = expressions.toggle_mask(layers, "temple")
    assert temple.any() and temple[:, 100:].sum() == temple.sum()         # one drop, image right
    made = expressions.paint_toggles(DummyImageGenerator(), model, ["blush", "squeeze", "anger"], seed=3)
    expressions.generate(DummyImageGenerator(), model, tag_image=lambda image: None)  # keeps the toggles
    assert set(expressions.read(model)["toggles"]) == {"blush", "squeeze", "anger"}
    anger = np.asarray(Image.open(model / made["toggles"]["anger"]))
    red = (anger[..., 3] > 200) & (anger[..., 0] > 200) & (anger[..., 1] < 90)
    assert red.sum() > 50 and np.nonzero(red)[1].max() < 100   # a drawn red mark, image-left forehead
    blush = np.asarray(Image.open(model / made["toggles"]["blush"]))[..., 3]
    assert blush[125, 77] > 200 and blush[30, 100] == 0 and blush[107, 77] < 30
    # Named for the preview's buttons, over the face and under the bangs.
    names = [name for name, _ in preview_layers(model / "assets")]
    assert names.index("eye_close") < names.index("toggle > < eyes") < names.index("front hair")
    assert names.index("toggle Blush") < names.index("front hair")


def test_toggle_requests_are_validated(tmp_path, monkeypatch):
    import json

    import app as app_module
    import config
    from src.api import generation

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    model = _character(tmp_path)
    (model / "metadata.json").write_text(json.dumps({"request": {"prompt": "1girl"}, "assets": [
        {"layer_type": "face", "file_path": "x", "metadata": {"backend": "see-through"}}]}))
    client = app_module.create_app(use_dummy_generators=True).test_client()
    submitted = []
    monkeypatch.setattr(generation.job_queue, "submit", lambda req: submitted.append(req) or type("J", (), {"id": "j"})())
    monkeypatch.setattr(generation.job_queue, "describe", lambda job: {"id": job.id})

    assert client.post("/api/generation/expressions/model", json={"toggles": ["blush", "blush", "hearts"]}).status_code == 202
    assert submitted[-1].operation == "expressions" and submitted[-1].toggles == ["blush", "hearts"]
    assert client.post("/api/generation/expressions/model", json={"toggles": ["wings"]}).status_code == 400
    assert client.post("/api/generation/generate", json={"prompt": "1girl", "toggles": "blush"}).status_code == 400
    assert client.post("/api/generation/generate", json={"prompt": "1girl", "toggles": ["tears"]}).status_code == 202
    assert submitted[-1].toggles == ["tears"]
    assert {"id": "blush", "label": "Blush"} in client.get("/api/generation/options").get_json()["toggles"]
    assert b"Paint toggles" in client.get("/preview/model").data


def test_drawn_emotes_preserve_art_and_need_no_diffusion(tmp_path):
    from src.core import expressions
    model=_character(tmp_path)
    before=(model/"base_image.png").read_bytes()
    class NoDiffusion:
        def inpaint(self,*args,**kwargs):
            raise AssertionError("Drawn manga emotes must not use the GPU")
    made=expressions.paint_toggles(NoDiffusion(),model,["gloom","sparkles"])
    layers=expressions.load_layers(model/"assets")
    gloom=np.asarray(Image.open(model/made["toggles"]["gloom"]))
    assert 20 < gloom[...,3].max() <=125 and gloom[...,3].sum()>1000
    assert not gloom[...,3][layers["face"]==0].any(),"Gloom stays on the face silhouette"
    sparkles=np.asarray(Image.open(model/made["toggles"]["sparkles"]))
    assert (sparkles[...,3]>150).sum()>30 and (sparkles[...,3]>0)[layers["face"]==0].any()
    assert (model/"base_image.png").read_bytes()==before


def test_emote_repackaging_preserves_a_concurrent_rename(tmp_path,monkeypatch):
    import json
    from src.pipeline.assembly_line import AssemblyLine
    from src.models.vtuber_model import Asset,VTuberModel,GenerationRequest
    from src.core import expressions
    model_dir=_character(tmp_path)
    assets=[Asset(layer_type=p.stem.split("_",1)[1],file_path=p,metadata={"backend":"see-through","order":i})
            for i,p in enumerate(sorted((model_dir/"assets").glob("*.png")))]
    model=VTuberModel(id=model_dir.name,name="study",request=GenerationRequest(prompt="1girl"),assets=assets)
    meta=model_dir/"metadata.json";meta.write_text(json.dumps(model.to_dict()))
    line=AssemblyLine(output_base_dir=tmp_path,use_dummy_generators=True)
    def repackage(*args):
        data=json.loads(meta.read_text());data["display_name"]="Owner's new name"
        meta.write_text(json.dumps(data));return model_dir/"model"
    monkeypatch.setattr(line,"_assemble_model",repackage)
    monkeypatch.setattr(line,"_ensure_generator",lambda *_:(_ for _ in ()).throw(AssertionError("Vector emotes must not load diffusion")))
    line.add_expressions(model_dir.name,toggles=["gloom"])
    data=json.loads(meta.read_text())
    assert data["display_name"]=="Owner's new name" and "gloom" in data["metadata"]["expressions"]["toggles"]


def test_vowel_mouths_are_painted_and_become_lip_sync_layers(tmp_path, monkeypatch):
    import json

    import app as app_module
    import config
    from src.api import generation
    from src.core import expressions
    from src.core.image_generator import DummyImageGenerator
    from src.core.live_preview import preview_layers

    model = _character(tmp_path)
    made = expressions.paint_visemes(DummyImageGenerator(), model, seed=2)
    assert set(made["visemes"]) == set("aiueo")
    mouth = expressions.mouth_mask(expressions.load_layers(model / "assets"))
    for vowel, path in made["visemes"].items():
        alpha = np.asarray(Image.open(model / path))[..., 3]
        assert alpha.any() and not alpha[~expressions._grow(mouth, 14)].any(), f"{vowel}: only around the mouth (feathered)"
    expressions.generate(DummyImageGenerator(), model, tag_image=lambda image: None)
    assert set(expressions.read(model)["visemes"]) == set("aiueo"), "Repainting blink/talk keeps the vowels"
    names = [name for name, _ in preview_layers(model / "assets")]
    assert all(f"mouth_{v}" in names for v in "aiueo") and names.index("mouth_a") > names.index("face")
    assets = expressions.as_assets(model, expressions.read(model), [])
    assert {a.layer_type for a in assets} >= {f"mouth_{v}" for v in "aiueo"} and all(a.metadata["expression"] for a in assets)

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    (model / "metadata.json").write_text(json.dumps({"request": {"prompt": "1girl"}, "assets": [
        {"layer_type": "face", "file_path": "x", "metadata": {"backend": "see-through"}}]}))
    client = app_module.create_app(use_dummy_generators=True).test_client()
    submitted = []
    monkeypatch.setattr(generation.job_queue, "submit", lambda req: submitted.append(req) or type("J", (), {"id": "j"})())
    monkeypatch.setattr(generation.job_queue, "describe", lambda job: {"id": job.id})
    assert client.post("/api/generation/expressions/model", json={"toggles": ["visemes"]}).status_code == 202
    assert submitted[-1].visemes and submitted[-1].toggles is None
    assert client.post("/api/generation/generate", json={"prompt": "1girl", "toggles": ["visemes", "blush"]}).status_code == 202
    assert submitted[-1].visemes and submitted[-1].toggles == ["blush"]
    assert any(t["id"] == "visemes" for t in client.get("/api/generation/options").get_json()["toggles"])
    assert b"A I U E O mouths" in client.get("/preview/model").data


def test_alternate_hair_fills_hidden_regions_and_replaces_the_part(tmp_path, monkeypatch):
    import json

    import app as app_module
    import config
    from src.api import generation
    from src.core import expressions, variants
    from src.core.image_generator import DummyImageGenerator
    from src.core.live_preview import preview_layers

    monkeypatch.setattr("src.core.asset_separator._anime_seg_mask", lambda image: None)
    model = _character(tmp_path)
    before = {p.name: p.read_bytes() for p in (model / "assets").iterdir()}
    assert variants.brief_without("1girl, long silver hair, blue eyes, hoodie", "hair") == "1girl, blue eyes, hoodie"
    class Bangs:  # paints black bangs across the top of the selection, leaves the rest as it was
        def inpaint(self, image, mask, prompt, **kw):
            out = np.array(image); m = np.asarray(mask) > 127; m[90:] = False
            out[m] = (20, 20, 30)
            self.prompt = prompt
            return Image.fromarray(out)
    painter = Bangs()
    made = variants.paint(painter, model, "hair", "short black bob", seed=1, brief="1girl, long hair, blue eyes")
    assert painter.prompt == "short black bob, 1girl, blue eyes", "The old hair tags are dropped from the brief"
    [(vid, v)] = made["variants"].items()
    assert vid == "hair-1" and v["label"] == "Hair: short black bob" and v["after"] == "front hair"
    alpha = np.asarray(Image.open(model / v["path"]))[..., 3]
    assert alpha[80, 130] > 200, "Drawn where the old bangs were"
    assert alpha[107, 127] == 0 and alpha[161, 100] == 0, "Never over the eyes or mouth (blink and lip sync stay live)"
    assert alpha[100, 140] == 0, "Below the new bangs the face shows through: nothing was drawn there"
    assert alpha[230, 10] == 0, "Background stays transparent"
    assert before == {p.name: p.read_bytes() for p in (model / "assets").iterdir()}, "The artist's layers are untouched"
    assert variants.paint(painter, model, "hair", "pink twintails")["variants"].keys() == {"hair-1", "hair-2"}
    expressions.generate(DummyImageGenerator(), model, tag_image=lambda image: None)
    assert set(expressions.read(model)["variants"]) == {"hair-1", "hair-2"}
    names = [name for name, _ in preview_layers(model / "assets")]
    assert names.index("front hair") < names.index("toggle Hair: short black bob")
    assets = expressions.as_assets(model, expressions.read(model), [])
    assert "variant hair-1" in {a.layer_type for a in assets}
    import pytest
    with pytest.raises(ValueError):
        variants.paint(DummyImageGenerator(), model, "top", "red hoodie")   # no topwear layer

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    (model / "metadata.json").write_text(json.dumps({"request": {"prompt": "1girl"}, "assets": [
        {"layer_type": "face", "file_path": "x", "metadata": {"backend": "see-through"}}]}))
    client = app_module.create_app(use_dummy_generators=True).test_client()
    submitted = []
    monkeypatch.setattr(generation.job_queue, "submit", lambda req: submitted.append(req) or type("J", (), {"id": "j"})())
    monkeypatch.setattr(generation.job_queue, "describe", lambda job: {"id": job.id})
    for body in ({"part": "wings", "prompt": "x"}, {"part": "hair", "prompt": " "}, {"part": "hair", "prompt": "x" * 201}):
        assert client.post("/api/generation/variant/model", json=body).status_code == 400
    assert client.post("/api/generation/variant/model", json={"part": "top", "prompt": " red  hoodie "}).status_code == 202
    assert submitted[-1].operation == "expressions" and submitted[-1].variant == {"part": "top", "prompt": "red hoodie"}
    assert b"Alternate look" in client.get("/preview/model").data


def test_drawn_head_symbols_and_mouth_emotes(tmp_path):
    from src.core import expressions
    from src.core.image_generator import DummyImageGenerator
    from src.core.live_preview import preview_layers

    model = _character(tmp_path)
    layers = expressions.load_layers(model / "assets")
    assert (expressions.toggle_mask(layers, "mouth") == expressions.mouth_mask(layers)).all()
    drawn = ["question", "exclaim", "zzz", "notes", "hearts_float"]
    made = expressions.paint_toggles(DummyImageGenerator(), model, drawn + ["tongue"], seed=1)
    shapes = {}
    for name in drawn:
        alpha = np.asarray(Image.open(model / made["toggles"][name]))[..., 3]
        assert alpha.sum() > 255 * 30, f"{name} is drawn"
        ys, xs = np.nonzero(alpha > 128)
        shapes[name] = (ys.min(), xs.mean())
    assert shapes["question"][0] < 100 and shapes["question"][1] > 100, "Above the head, image right"
    q = np.asarray(Image.open(model / made["toggles"]["question"]))
    assert not np.array_equal(q, np.asarray(Image.open(model / made["toggles"]["exclaim"]))), "Each symbol is its own drawing"
    names = [name for name, _ in preview_layers(model / "assets")]
    assert names.index("toggle Question mark") > names.index("front hair"), "Symbols sit over the hair"
    assert names.index("toggle Tongue out") < names.index("front hair")
