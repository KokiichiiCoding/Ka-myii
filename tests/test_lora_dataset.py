"""The LoRA dataset builder's curation and captions (scripts/build_lora_dataset.py)."""
import importlib.util
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

spec = importlib.util.spec_from_file_location("build_lora_dataset", Path(__file__).parents[1] / "scripts" / "build_lora_dataset.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

GOOD = {"general": ["1girl", "solo", "looking at viewer", "virtual youtuber", "^_^"], "character": ["hatsune miku"],
        "rating": "general", "scores": {"solo": 0.95, "looking at viewer": 0.9, "simple background": 0.8}}


def _figure(box=(300, 100, 530, 1216), second=None):
    img = Image.new("RGB", (832, 1216), "white")
    alpha = Image.new("L", img.size, 0)
    for b in [box] + ([second] if second else []):
        ImageDraw.Draw(img).ellipse(b, fill=(200, 120, 160)); ImageDraw.Draw(alpha).ellipse(b, fill=255)
    return img, np.asarray(alpha)


def test_a_centred_solo_figure_on_white_is_kept():
    assert builder.verdict(*_figure(), GOOD) == []


def test_rig_breaking_pictures_are_rejected_with_reasons():
    assert "more than one figure" in builder.verdict(*_figure(second=(20, 900, 200, 1216)), GOOD)
    assert "cut off at the top" in builder.verdict(*_figure(box=(300, -50, 530, 1216)), GOOD)
    assert "cut off at the left" in builder.verdict(*_figure(box=(-60, 100, 400, 1216)), GOOD)
    shoulders = _figure(); ImageDraw.Draw(shoulders[0]).rectangle((0, 1000, 832, 1216), fill=(200, 120, 160))
    shoulders = (shoulders[0], np.maximum(shoulders[1], np.pad(np.full((216, 832), 255, np.uint8), ((1000, 0), (0, 0)))))
    assert not [r for r in builder.verdict(*shoulders, GOOD) if r.startswith("cut off")]  # low side cuts are fine
    assert "figure too small" in builder.verdict(*_figure(box=(350, 700, 480, 1216)), GOOD)
    img, alpha = _figure(); ImageDraw.Draw(img).rectangle((0, 0, 832, 600), fill=(20, 40, 90))
    assert "background not plain white" in builder.verdict(img, alpha, GOOD)
    img, alpha = _figure(); ImageDraw.Draw(img).rectangle((40, 40, 792, 1176), outline=(180, 150, 60), width=4)
    assert builder.verdict(img, alpha, GOOD) == ["decorations in the background"]  # a thin frame
    bad = {**GOOD, "rating": "explicit", "scores": {"solo": 0.9, "closed eyes": 0.8, "simple background": 0.9}}
    assert {"tag: closed eyes", "missing tag: looking at viewer", "rating: explicit"} <= set(builder.verdict(*_figure(), bad))


def test_captions_lead_with_the_trigger_and_skip_character_names():
    assert builder.caption(GOOD, "kmyvt") == "kmyvt, 1girl, solo, looking at viewer, ^_^"


def test_prompts_are_varied_and_always_rig_ready():
    rng = random.Random(1)
    prompts = [builder.compose(rng)[0] for _ in range(50)]
    assert len(set(prompts)) == 50 and all(builder.FIXED in p for p in prompts)
