"""
Targeted refinement ("detailer"): redraw each hand or the face in its own tight crop at SDXL's
native resolution and paste it back, so a 60 px hand gets ~5x the pixels it had in the picture.

Where: the separated layers (handwear-l/-r, face) or, for an unseparated full-body sketch, the
extension plan's wrist targets and retained face. Prompts describe only the part: the whole
character brief in a small crop invites a tiny second person (seen with the anger-vein toggle).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

PROMPTS = {
    "hands": ("hand, detailed hands, five fingers, anime coloring",
              "bad hands, extra fingers, missing fingers, fused fingers, extra digits, face, person, 1girl"),
    "face": ("face, detailed face, detailed eyes, beautiful detailed eyes, anime coloring",
             "bad face, blurry eyes, asymmetrical eyes, extra eyes, hat, full body, person"),
}
STRENGTH = {"hands": 0.5, "face": 0.35}


def _square(cx: float, cy: float, side: float, W: int, H: int) -> List[int]:
    half = side / 2
    return [max(0, round(cx - half)), max(0, round(cy - half)), min(W, round(cx + half)), min(H, round(cy + half))]


def boxes(model_dir, target: str) -> List[List[int]]:
    """Square boxes [x0, y0, x1, y1] around each hand or the face; [] when they can't be found."""
    from src.core import outpaint

    model_dir = Path(model_dir)
    with Image.open(model_dir / "base_image.png") as img:
        W, H = img.size
    g = outpaint.anatomy(model_dir)  # separated layers, or a sketch's saved extension geometry
    if not g.get("face"):
        return []
    x0, y0, x1, y1 = g["face"]
    fw = x1 - x0
    if target == "face":
        return [_square((x0 + x1) / 2, (y0 + y1) / 2, max(fw, y1 - y0) * 1.5, W, H)]
    out = []
    for side in ("l", "r"):
        for path in (model_dir / "assets").glob(f"*_handwear-{side}.png"):
            with Image.open(path) as img:
                alpha = np.asarray(img.getchannel("A")) > 128
            ys, xs = np.nonzero(alpha)
            if len(xs) < 30:
                continue
            hand = ys > ys.max() - fw * .6  # the end of the arm, not the sleeve
            out.append(_square(xs[hand].mean(), ys[hand].mean(), fw * .9, W, H))
    if out:
        return out
    try:
        saved = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        ext = (saved.get("request") or {}).get("extension") or {}
        # Older plans did not store their pose; it follows from the saved placement.
        points = ext.get("pose_points") or (outpaint.pose_points(ext) if {"canvas", "foot_y", "target_heads"} <= set(ext) else [])
    except (OSError, ValueError, KeyError, TypeError):
        points = []
    if len(points) >= 8:
        hh = g.get("head_height", fw * 1.3)
        out = [_square(points[i][0], points[i][1] + .2 * hh, .9 * hh, W, H) for i in (4, 7)]
    return out
