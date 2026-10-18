"""
Extend a character cut off by the image border: a larger canvas around the artwork and a mask
of what is new, for SDXL to paint (src/core/image_generator.py ``inpaint``).

"full_body": the artwork is scaled down into the top of a same-sized portrait canvas and the
model paints the rest of the body. "uncrop": only the borders the character touches get room
(a hat cut off at the top, an elbow at the side), keeping the framing. Either way a band along
each cut edge is repainted too, so the join is drawn rather than seamed.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

FULL_BODY_TAGS = "full body, standing, legs, shoes, simple background, white background"
FULL_BODY_NEGATIVE = "cropped, out of frame, close-up, portrait, upper body, cowboy shot, feet out of frame"
UNCROP_NEGATIVE = "cropped, out of frame, border, frame"


def touched_borders(fg: np.ndarray, threshold: float = 0.02) -> dict:
    """Which canvas borders the character runs into (share of the border row/column)."""
    return {"top": fg[0].mean() > threshold, "bottom": fg[-1].mean() > threshold,
            "left": fg[:, 0].mean() > threshold, "right": fg[:, -1].mean() > threshold}


def plan(image: Image.Image, fg: np.ndarray, mode: str = "full_body") -> Tuple[Image.Image, np.ndarray, dict]:
    """(canvas, mask, info): the canvas holds the artwork where it goes, the mask is True where
    the model must paint. ``fg`` is the character mask of ``image`` (bool, image size)."""
    image = image.convert("RGB")
    W, H = image.size
    touched = touched_borders(fg)
    background = _background_colour(image, fg)
    if mode == "full_body":
        scale = 0.55
        w, h = int(W * scale), int(H * scale)
        left, top = (W - w) // 2, int(H * (0.05 if touched["top"] else 0.02))
        out_w, out_h = W, H
    elif mode == "uncrop":
        pad = {side: (int((W if side in ("left", "right") else H) * 0.12) if hit else 0) for side, hit in touched.items()}
        grown_w, grown_h = W + pad["left"] + pad["right"], H + pad["top"] + pad["bottom"]
        scale = min(W / grown_w, H / grown_h)  # same output size, the artwork shrinks a little
        w, h = int(W * scale), int(H * scale)
        left = int(pad["left"] * scale) + (W - int(grown_w * scale)) // 2
        top = int(pad["top"] * scale) + (H - int(grown_h * scale)) // 2
        out_w, out_h = W, H
    else:
        raise ValueError("mode must be full_body or uncrop")

    canvas = Image.new("RGB", (out_w, out_h), background)
    canvas.paste(image.resize((w, h), Image.Resampling.LANCZOS), (left, top))
    mask = np.ones((out_h, out_w), bool)
    mask[top:top + h, left:left + w] = False
    band = max(8, int(min(w, h) * 0.05))
    # Repaint a band inside each edge the character was cut at, so the join is drawn.
    if touched["bottom"] or mode == "full_body":
        mask[top + h - band:top + h, left:left + w] = True
    if touched["top"]:
        mask[top:top + band, left:left + w] = True
    if touched["left"]:
        mask[top:top + h, left:left + band] = True
    if touched["right"]:
        mask[top:top + h, left + w - band:left + w] = True
    info = {"touched": [side for side, hit in touched.items() if hit], "placed": [left, top, w, h], "mode": mode}
    return canvas, mask, info


def _background_colour(image: Image.Image, fg: np.ndarray):
    px = np.asarray(image)[~fg] if (~fg).sum() > 100 else np.asarray(image).reshape(-1, 3)
    return tuple(int(v) for v in np.median(px, axis=0))


def character_mask(image: Image.Image) -> np.ndarray:
    """The character (segmentation model, colour key as a fallback)."""
    from src.core.asset_separator import AssetSeparator, _anime_seg_mask

    alpha = _anime_seg_mask(image)
    if alpha is None:
        alpha = np.asarray(AssetSeparator._chroma_cutout(image.convert("RGBA")).split()[-1])
    return alpha > 128

