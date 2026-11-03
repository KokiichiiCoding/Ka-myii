"""
Keep the original character's head layers on an extended body.

See-through re-separates an extended full body with a much smaller head, and small eyes and
mouths separate worst. The source character's facial-feature layers were separated at its own,
larger size; each is scaled into the extension's placement (``info["placed"]``) and replaces the
re-separated layer of the same name, but only where the new picture still shows those pixels.
Only features drawn over the face: swapping the face or hair itself put the source's hidden
forehead under the new bangs (a grey smudge on a real hoodie extension) and reordered ears
against earrings.
Seam check: the extension pastes the resized source picture and keeps it pixel for pixel where
nothing was redrawn, so a layer whose pixels differ (hair edges redrawn at a cut border, an
edited fringe) keeps its new See-through version. Painted expressions follow a kept face.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

HEAD_PARTS = ("eyewhite", "irides", "eyelash", "eyebrow", "nose", "mouth")
FACE_PARTS = ("eyewhite", "irides", "eyelash", "mouth")
MISMATCH = 24   # largest channel difference that is still the same retained pixel
LIMIT = 0.08    # share of a layer's pixels allowed to differ (antialiasing, a few redrawn strands)


def _placer(info: dict, size):
    left, top, w, h = (int(v) for v in info["placed"])

    def place(image: Image.Image) -> Image.Image:
        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        canvas.paste(image.convert("RGBA").resize((w, h), Image.Resampling.LANCZOS), (left, top))
        return canvas
    return place


def transfer(source_dir: Path, model_dir: Path, info: dict) -> List[str]:
    """Replace the new character's head layers with the source's where they still match; returns
    the layer names kept (and copies the source's painted expressions when eyes and mouth were)."""
    source_dir, model_dir = Path(source_dir), Path(model_dir)
    if not info.get("placed") or not (source_dir / "base_image.png").is_file():
        return []
    with Image.open(model_dir / "base_image.png") as img:
        base = img.convert("RGB")
    place = _placer(info, base.size)
    with Image.open(source_dir / "base_image.png") as img:
        retained = np.asarray(place(img))
    differs = np.abs(retained[..., :3].astype(np.int16) - np.asarray(base, np.int16)).max(-1) > MISMATCH
    differs |= retained[..., 3] < 255  # outside the pasted artwork
    targets = {p.stem.split("_", 1)[1]: p for p in (model_dir / "assets").glob("[0-9]*_*.png")}
    kept = []
    for path in sorted((source_dir / "assets").glob("[0-9]*_*.png")):
        name = path.stem.split("_", 1)[1]
        if name.split("-")[0] not in HEAD_PARTS or name not in targets:
            continue
        with Image.open(path) as img:
            layer = place(img)
        alpha = np.asarray(layer)[..., 3] > 128
        if alpha.sum() < 20:
            continue
        share = float(differs[alpha].mean())
        if share > LIMIT:
            logger.info("Head layer %s: %.0f%% redrawn by the extension, keeping the new separation", name, share * 100)
            continue
        layer.save(targets[name])
        kept.append(name)
    if all(any(k.split("-")[0] == part for k in kept) for part in FACE_PARTS):
        _copy_expressions(source_dir, model_dir, place)
    logger.info("Kept %d original head layers on %s: %s", len(kept), model_dir.name, ", ".join(kept) or "none")
    return kept


def _copy_expressions(source_dir: Path, model_dir: Path, place):
    """The source's painted closed eyes, mouths, toggles and vowels, scaled into place."""
    from src.core import expressions

    made = expressions.read(source_dir)
    if not made:
        return
    out = {}
    for key, value in made.items():
        if isinstance(value, dict):  # toggles / visemes: {name: relative path}
            # (variants are not carried: an outfit was drawn for the old body)
            out[key] = {k: v for k, v in value.items() if isinstance(v, str) and _copy(source_dir / v, model_dir / v, place)}
        elif isinstance(value, str) and value.startswith(expressions.FOLDER + "/"):
            if _copy(source_dir / value, model_dir / value, place):
                out[key] = value
        else:
            out[key] = value  # tags, which mouth state was drawn
    (model_dir / expressions.FOLDER).mkdir(parents=True, exist_ok=True)
    (model_dir / expressions.FOLDER / "expressions.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def _copy(src: Path, dst: Path, place) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        place(img).save(dst)
    return True
