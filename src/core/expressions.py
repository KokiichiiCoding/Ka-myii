"""
Character-specific expression art: closed eyes and the other mouth state, painted by SDXL.

The preview rig blinks by fading the open-eye parts out and an ``eye_close`` layer in, and
talks by fading ``mouth_close`` out and a vertically stretched ``mouth_open`` in. See-through
gives one drawn mouth and no closed eyes, so without this the rig used generic clip-art. Here
both are inpainted in the character's own style and saved as extra layers in
``<model>/expressions/`` (outside ``assets/``, which rebuild_layers.py replaces).

Recipe (tuned on a real result, see AGENTS.md): eyes are a geometric ellipse per eye around
everything eye-like, minus bangs drawn on top, flooded with the surrounding skin colour and
repainted at full strength as "closed eyes" — a looser mask left iris strokes the model then
drew around. The mouth gets room to open; the tagger says which state the drawing already is.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)

FOLDER = "expressions"
EYE_PARTS = ("eyewhite", "irides", "eyelash")
NEGATIVE_EYES = "open eyes, iris, pupils, eyeball, heart, sparkle, tears, glasses"
# Seen: with the whole character description, the model drew a black mask over the chin.
NEGATIVE_OPEN = "closed mouth, tongue out, fang, mask, choker, beard, facial hair, black"
NEGATIVE_CLOSED = "open mouth, teeth, tongue, mask, beard, facial hair"


def load_layers(assets_dir: Path) -> Dict[str, np.ndarray]:
    """Alpha (uint8) of each separated layer by name ("02_front hair.png" -> "front hair")."""
    out = {}
    for path in sorted(Path(assets_dir).glob("*.png")):
        name = path.stem.split("_", 1)[1] if "_" in path.stem and path.stem.split("_", 1)[0].isdigit() else path.stem
        with Image.open(path) as im:
            out[name] = np.asarray(im.convert("RGBA"))[..., 3]
    return out


def _grow(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(px * 2 + 1))) > 0


def _zeros(layers):
    return np.zeros(next(iter(layers.values())).shape, bool)


def eye_mask(layers: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """An ellipse per eye around its white, iris and lashes, minus bangs drawn on top."""
    if not layers:
        return None
    shape = next(iter(layers.values())).shape
    canvas = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    found = False
    for side in "lr":
        m = _zeros(layers)
        for part in EYE_PARTS:
            if f"{part}-{side}" in layers:
                m |= layers[f"{part}-{side}"] > 16
        ys, xs = np.nonzero(m)
        if len(xs) < 20:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        px, py = (x1 - x0) * 0.12 + 4, (y1 - y0) * 0.18 + 4
        draw.ellipse((x0 - px, y0 - py, x1 + px, y1 + py), fill=255)
        found = True
    if not found:
        return None
    mask = np.asarray(canvas) > 0
    if "front hair" in layers:
        mask &= ~(layers["front hair"] > 128)
    return mask


def mouth_mask(layers: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Room for the mouth to open: an ellipse around it, kept inside the face."""
    if "mouth" not in layers:
        return None
    ys, xs = np.nonzero(layers["mouth"] > 16)
    if len(xs) < 10:
        return None
    width = xs.max() - xs.min()
    shape = layers["mouth"].shape
    canvas = Image.new("L", (shape[1], shape[0]), 0)
    # Room around the lips and below them: the probe that opened convincingly used ~16 px.
    ImageDraw.Draw(canvas).ellipse((xs.min() - 16, ys.min() - 12, xs.max() + 16,
                                    ys.max() + max(20, width * 0.45)), fill=255)
    mask = np.asarray(canvas) > 0
    if "face" in layers:
        mask &= _grow(layers["face"] > 128, 2)
    if "front hair" in layers:
        mask &= ~(layers["front hair"] > 128)
    return mask


def skin_colour(base: Image.Image, region: np.ndarray, layers: Dict[str, np.ndarray]):
    """Median colour of the face just around ``region`` (no hair, brows or eyes)."""
    ring = _grow(region, 14) & ~region
    if "face" in layers:
        ring &= layers["face"] > 128
    for name in ("front hair", "eyebrow-l", "eyebrow-r"):
        if name in layers:
            ring &= ~(layers[name] > 128)
    pixels = np.asarray(base.convert("RGB"))[ring]
    return tuple(int(v) for v in np.median(pixels, axis=0)) if len(pixels) > 20 else None


def _soft(mask: np.ndarray, feather: int = 4) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(feather))


def _changed(painted: Image.Image, base: Image.Image, mask: np.ndarray, grow: int = 5) -> np.ndarray:
    """The part of ``mask`` the painting really changed (plus a margin): a mouth layer that is
    just the mouth, not a plate of skin, so the rig's mouth anchor and stretch fit it."""
    diff = np.abs(np.asarray(painted.convert("RGB"), np.int16) - np.asarray(base.convert("RGB"), np.int16)).max(-1)
    changed = _grow(mask & (diff > 32), grow) & mask
    return changed if changed.sum() > 30 else mask


def _save_layer(painted: Image.Image, mask: np.ndarray, path: Path):
    layer = painted.convert("RGBA")
    layer.putalpha(_soft(mask))
    path.parent.mkdir(parents=True, exist_ok=True)
    layer.save(path)


def generate(generator, model_dir: Path, seed: Optional[int] = None,
             task_id: Optional[str] = None, tag_image=None) -> Dict:
    """Paint the expression layers for a separated character; returns what was made
    (also written to ``expressions/expressions.json``)."""
    from src.core import tagger

    model_dir = Path(model_dir)
    base = Image.open(model_dir / "base_image.png").convert("RGB")
    layers = load_layers(model_dir / "assets")
    out_dir = model_dir / FOLDER
    made: Dict = {}

    tags = (tag_image or tagger.tag_image)(base)
    scores = (tags or {}).get("scores", {})
    mouth_open_now = scores.get("open mouth", 0) > max(0.5, scores.get("closed mouth", 0))
    smile = scores.get("smile", 0) > 0.5
    described = tagger.describe(tags) if tags else ""
    made["tags"] = described
    # Face-only prompts: the character's clothes/accessory tags invite them into a face patch.
    seed = 7 if seed is None else int(seed)

    eyes = eye_mask(layers)
    if eyes is not None and eyes.sum() > 50:
        skin = skin_colour(base, eyes, layers)
        start = base.copy()
        if skin:
            start.paste(skin, (0, 0, *base.size), Image.fromarray(eyes.astype(np.uint8) * 255))
        painted = generator.inpaint(start, Image.fromarray(eyes.astype(np.uint8) * 255),
                                    "closed eyes", negative=NEGATIVE_EYES, strength=1.0,
                                    seed=seed, task_id=task_id, label="Painting closed eyes")
        _save_layer(painted, eyes, out_dir / "eye_close.png")
        made["eye_close"] = f"{FOLDER}/eye_close.png"

    mouth = mouth_mask(layers)
    if mouth is not None and mouth.sum() > 30:
        if mouth_open_now:
            name, wanted, negative = "mouth_close", "closed mouth", NEGATIVE_CLOSED
        else:
            name, negative = "mouth_open", NEGATIVE_OPEN
            wanted = "open mouth, :d, teeth, smile" if smile else "open mouth, :o, teeth"
        painted = generator.inpaint(base, Image.fromarray(mouth.astype(np.uint8) * 255), wanted,
                                    negative=negative, strength=1.0, seed=seed, task_id=task_id,
                                    label="Painting the " + ("closed" if mouth_open_now else "open") + " mouth")
        _save_layer(painted, _changed(painted, base, mouth), out_dir / f"{name}.png")
        made[name] = f"{FOLDER}/{name}.png"
    made["mouth_drawn"] = "open" if mouth_open_now else "closed"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "expressions.json").write_text(json.dumps(made, indent=2), encoding="utf-8")
    logger.info("Expressions for %s: %s", model_dir.name, ", ".join(k for k in made if k in ("eye_close", "mouth_open", "mouth_close")))
    return made


def read(model_dir: Path) -> Dict:
    path = Path(model_dir) / FOLDER / "expressions.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def as_assets(model_dir: Path, made: Dict, assets: list) -> list:
    """Assets for the painted layers, ordered just above the parts they replace."""
    from src.models.vtuber_model import Asset

    def order_of(*names):
        orders = [a.metadata.get("order", 0) for a in assets if a.layer_type.split("-")[0] in names]
        return max(orders) + 0.5 if orders else 0.5

    out = []
    for key, anchor in (("eye_close", ("eyewhite", "irides", "eyelash")), ("mouth_open", ("mouth",)),
                        ("mouth_close", ("mouth",))):
        if key in made and (Path(model_dir) / made[key]).exists():
            out.append(Asset(layer_type=key, file_path=Path(model_dir) / made[key],
                             metadata={"order": order_of(*anchor), "expression": True}))
    return out
