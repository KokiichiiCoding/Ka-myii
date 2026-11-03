"""
Prepare imported artwork (a Perchance / ChatGPT / commissioned design) for decomposition.

Such images arrive square or oddly sized, small or JPEG-soft, with scenery behind the figure,
the figure off-centre, or several views side by side. See-through and the expression painter
do best on one centred, front-facing character on a plain background at SDXL size, so:

* frame: the largest character figure (segmentation model) is cropped with a margin and fitted
  into an 832x1216 portrait at a consistent size (small images are scaled up);
* white background: everything around the figure becomes white, edges kept soft;
* enhance (optional): a light SDXL redraw of the whole picture restores detail lost to
  upscaling or compression;
* the tagger describes the character, so later edits / extensions / expressions have a brief.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

TARGET = (832, 1216)


def character_alpha(image: Image.Image) -> np.ndarray:
    """Soft character alpha (uint8): segmentation model, colour key as a fallback."""
    from src.core.asset_separator import AssetSeparator, _anime_seg_mask

    alpha = _anime_seg_mask(image)
    if alpha is None:
        alpha = np.asarray(AssetSeparator._chroma_cutout(image.convert("RGBA")).split()[-1])
    return alpha


def main_figure(alpha: np.ndarray) -> np.ndarray:
    """The largest connected figure (reference sheets show several views)."""
    from src.core.asset_separator import _components

    h, w = alpha.shape
    scale = max(1, int(np.ceil(max(h, w) / 256)))
    small = np.asarray(Image.fromarray(alpha).resize((max(1, w // scale), max(1, h // scale)),
                                                      Image.Resampling.BOX)) > 128
    labels = _components(small)
    if labels.max() == 0:
        return alpha > 128
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = labels == sizes.argmax()
    keep = np.asarray(Image.fromarray(keep.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 0
    # Grow back the outline the coarse labelling rounded off, then keep the real alpha inside.
    from src.core.regions import grow

    return grow(keep, scale * 2) & (alpha > 32)


def frame(image: Image.Image, alpha: np.ndarray, target: Tuple[int, int] = TARGET,
          margin: float = 0.06) -> Tuple[Image.Image, np.ndarray]:
    """Crop to the main figure plus a margin and fit it into ``target`` (centred, on the
    image's own background colour); returns the framed image and its alpha."""
    figure = main_figure(alpha)
    ys, xs = np.nonzero(figure)
    if not len(xs):
        return image.convert("RGB").resize(target, Image.Resampling.LANCZOS), alpha
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    fw, fh = x1 - x0, y1 - y0
    tw, th = target
    scale = min(tw * (1 - 2 * margin) / fw, th * (1 - 2 * margin) / fh)
    src = image.convert("RGB")
    a = Image.fromarray(np.where(figure, alpha, 0).astype(np.uint8))
    box = (int(x0), int(y0), int(x1), int(y1))
    sw, sh = max(1, round(fw * scale)), max(1, round(fh * scale))
    crop = src.crop(box).resize((sw, sh), Image.Resampling.LANCZOS)
    crop_a = a.crop(box).resize((sw, sh), Image.Resampling.LANCZOS)
    bg = tuple(int(v) for v in np.median(np.asarray(src)[alpha < 16], axis=0)) if (alpha < 16).sum() > 100 else (255, 255, 255)
    out = Image.new("RGB", target, bg)
    out_a = Image.new("L", target, 0)
    left, top = (tw - sw) // 2, (th - sh) // 2
    # A figure cut by its own picture border stays against that border (it continues there).
    if y1 >= image.height - 2:
        top = th - sh
    if y0 <= 1:
        top = 0
    out.paste(crop, (left, top))
    out_a.paste(crop_a, (left, top))
    return out, np.asarray(out_a)


def white_background(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    """The figure over white, through its soft alpha (anti-aliased edges survive)."""
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    rgb = np.asarray(image.convert("RGB"), np.float32)
    return Image.fromarray((rgb * a + 255.0 * (1 - a)).round().astype(np.uint8))


def prepare(image: Image.Image, frame_figure: bool = True, plain_background: bool = True,
            enhance: float = 0.0, generator=None, tag: Optional[Callable] = None,
            task_id: Optional[str] = None) -> Tuple[Image.Image, Optional[Dict]]:
    """The import pipeline: (prepared image, tags). ``enhance`` > 0 needs ``generator`` (an
    ImageGenerator); ``tag`` (tagger.tag_image) describes the framed figure, not the scenery."""
    image = image.convert("RGB")
    alpha = character_alpha(image)
    if frame_figure:
        image, alpha = frame(image, alpha)
    if plain_background:
        image = white_background(image, alpha)
    tags = tag(image) if tag else None
    if enhance > 0 and generator is not None:
        from src.core import tagger

        description = tagger.describe(tags) if tags else ""
        full = Image.new("L", image.size, 255)
        image = generator.inpaint(image, full, description or "1girl", strength=float(enhance), feather=0,
                                  task_id=task_id, label="Cleaning up the artwork")
        if plain_background:  # the redraw may tint the white a little
            image = white_background(image, character_alpha(image))
    return image, tags
