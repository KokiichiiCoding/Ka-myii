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
