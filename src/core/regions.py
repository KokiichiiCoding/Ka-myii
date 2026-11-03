"""
Selections for region edits: named parts from a character's separated layers, the background
from the segmentation model, and masks painted in the Studio.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from PIL import Image, ImageFilter

# Part groups offered in the Studio -> the See-through layer names they cover.
PARTS: Dict[str, tuple] = {
    "hair": ("front hair", "back hair", "side hair"),
    "face": ("face", "nose", "mouth", "eyebrow-l", "eyebrow-r", "eyewhite-l", "eyewhite-r",
             "irides-l", "irides-r", "eyelash-l", "eyelash-r"),
    "eyes": ("eyewhite-l", "eyewhite-r", "irides-l", "irides-r", "eyelash-l", "eyelash-r"),
    "mouth": ("mouth",),
    "top": ("topwear", "neckwear", "handwear-l", "handwear-r"),
    "bottom": ("bottomwear", "legwear", "footwear"),
    "accessories": ("headwear", "earwear", "eyewear", "tail", "wings", "objects"),
}
LABELS = {"hair": "Hair", "face": "Face", "eyes": "Eyes", "mouth": "Mouth", "top": "Top & arms",
          "bottom": "Bottom & legs", "accessories": "Accessories", "background": "Background"}


def layer_alphas(assets_dir: Path) -> Dict[str, np.ndarray]:
    from src.core.expressions import load_layers

    return load_layers(assets_dir) if Path(assets_dir).is_dir() else {}


def available_parts(model_dir: Path) -> List[str]:
    """Part groups this character has (any covered layer present), plus the background."""
    names = {p.stem.split("_", 1)[1] for p in Path(model_dir, "assets").glob("[0-9]*_*.png")}
    return [key for key, layers in PARTS.items() if names & set(layers)] + ["background"]


def visible_alphas(alphas: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Each layer's pixels that no layer above it covers (layers come back to front). Layers
    hold hidden, inpainted parts too — back hair behind the body — which must not be repainted."""
    covered = None
    out = {}
    for name in reversed(list(alphas)):
        solid = alphas[name] > 128
        out[name] = solid if covered is None else solid & ~covered
        covered = solid if covered is None else covered | solid
    return out


def part_mask(model_dir: Path, parts: Iterable[str], size) -> np.ndarray:
    """Union of the requested parts' visible pixels, grown a little to cover outlines."""
    w, h = size
    mask = np.zeros((h, w), bool)
    parts = list(parts)
    alphas = layer_alphas(Path(model_dir) / "assets") if any(p in PARTS for p in parts) else {}
    visible = visible_alphas({n: a for n, a in alphas.items() if a.shape == (h, w)})
    for part in parts:
        if part == "background":
            mask |= background_mask(Image.open(Path(model_dir) / "base_image.png"))
        for name in PARTS.get(part, ()):
            if name in visible:
                mask |= visible[name]
    return grow(mask, 6) if mask.any() else mask


def background_mask(image: Image.Image) -> np.ndarray:
    """Everything that is not the character (segmentation model, colour key as a fallback)."""
    from src.core.asset_separator import AssetSeparator, _anime_seg_mask

    alpha = _anime_seg_mask(image)
    if alpha is None:
        alpha = np.asarray(AssetSeparator._chroma_cutout(image.convert("RGBA")).split()[-1])
    return ~grow(alpha > 128, 3)


def painted_mask(data_url: Optional[str], size) -> np.ndarray:
    """A mask painted in the Studio (PNG data URL, white or opaque strokes) at ``size``."""
    w, h = size
    if not data_url:
        return np.zeros((h, w), bool)
    if not data_url.startswith("data:image/png;base64,"):
        raise ValueError("The painted selection must be a PNG data URL")
    try:
        with Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1], validate=True))) as img:
            img.load()
            rgba = img.convert("RGBA")
    except Exception:
        raise ValueError("Could not read the painted selection")
    if rgba.size[0] * rgba.size[1] > 16_000_000:
        raise ValueError("The painted selection is too large")
    rgba = rgba.resize((w, h), Image.Resampling.BILINEAR)
    arr = np.asarray(rgba)
    return (arr[..., 3] > 127) & (arr[..., :3].max(-1) > 127)


def grow(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not mask.any():
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(px * 2 + 1))) > 0


def to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255)
