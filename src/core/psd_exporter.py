"""
Layered file exporter.

Turns the separated layers into artist-ready, layered documents:

* **PSD** (Photoshop / Clip Studio / Live2D Cubism import) via ``pytoshop`` —
  the primary rig-ready deliverable. Layers are cropped to their bounding box
  and positioned by offset, exactly how a hand-authored Live2D PSD is built.
* **ORA** (OpenRaster) — an open, ZIP-based layered format written with only
  the standard library, so a correct layered file is *always* produced even if
  ``pytoshop`` is unavailable.

Both keep the back-to-front draw order and Live2D-friendly layer names.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.models.vtuber_model import Asset

logger = logging.getLogger(__name__)

try:
    import pytoshop
    from pytoshop import enums
    from pytoshop.user import nested_layers

    PYTOSHOP_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    PYTOSHOP_AVAILABLE = False


LayerTuple = Tuple[str, Image.Image, Tuple[int, int]]  # (name, cropped RGBA, (left, top))


def _clean_name(name: str) -> str:
    return name.replace("_", " ").strip().title()


def _ordered_layers(assets: List[Asset]) -> List[Asset]:
    """Return real layers (excluding helper cut-outs) sorted back-to-front."""
    layers = [a for a in assets if a.layer_type not in ("character",) and a.file_path.exists()]
    return sorted(layers, key=lambda a: a.metadata.get("order", 0))


def _crop_to_content(image: Image.Image) -> Optional[Tuple[Image.Image, Tuple[int, int]]]:
    """Crop an RGBA image to its non-transparent bounding box."""
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.split()[-1])
    ys, xs = np.where(alpha > 4)
    if len(xs) == 0:
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return rgba.crop((x0, y0, x1, y1)), (x0, y0)


def _build_layer_tuples(assets: List[Asset]) -> Tuple[List[LayerTuple], Tuple[int, int]]:
    """Load layer PNGs, crop to content, return (back-to-front list, canvas size)."""
    canvas_w = canvas_h = 0
    tuples: List[LayerTuple] = []
    for asset in _ordered_layers(assets):
        img = Image.open(asset.file_path).convert("RGBA")
        canvas_w = max(canvas_w, img.width)
        canvas_h = max(canvas_h, img.height)
        cropped = _crop_to_content(img)
        if cropped is None:
            continue
        region, (left, top) = cropped
        tuples.append((asset.layer_type, region, (left, top)))
    return tuples, (canvas_w, canvas_h)


def export_psd(assets: List[Asset], output_path: Path) -> Optional[Path]:
    """Write a layered .psd. Returns the path, or ``None`` if unavailable."""
    if not PYTOSHOP_AVAILABLE:
        logger.info("pytoshop not installed — skipping PSD (ORA will be used)")
        return None

    tuples, (cw, ch) = _build_layer_tuples(assets)
    if not tuples:
        return None

    layers = []
    # pytoshop stacks the *last* list item on top, which matches our
    # back-to-front ordering (front layer is last).
    for name, region, (left, top) in tuples:
        arr = np.asarray(region)
        h, w = arr.shape[:2]
        layers.append(
            nested_layers.Image(
                name=_clean_name(name),
                visible=True,
                opacity=255,
                top=top,
                left=left,
                bottom=top + h,
                right=left + w,
                channels={
                    0: arr[:, :, 0].copy(),
                    1: arr[:, :, 1].copy(),
                    2: arr[:, :, 2].copy(),
                    -1: arr[:, :, 3].copy(),
                },
            )
        )

    try:
        psd = nested_layers.nested_layers_to_psd(
            layers,
            color_mode=enums.ColorMode.rgb,
            compression=enums.Compression.raw,  # RLE path is broken upstream
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as fh:
            psd.write(fh)
        logger.info("Wrote layered PSD: %s (%d layers)", output_path, len(layers))
        return output_path
    except Exception as exc:  # pragma: no cover
        logger.warning("PSD export failed: %s", exc)
        return None


def export_ora(assets: List[Asset], output_path: Path) -> Optional[Path]:
    """Write a layered OpenRaster (.ora) file using only the standard library."""
    tuples, (cw, ch) = _build_layer_tuples(assets)
    if not tuples:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build stack.xml. In ORA the FIRST <layer> element is the topmost, so we
    # iterate our back-to-front list in reverse.
    image_el = ET.Element("image", {"w": str(cw), "h": str(ch), "version": "0.0.3"})
    stack_el = ET.SubElement(image_el, "stack")

    layer_pngs: Dict[str, Image.Image] = {}
    for idx, (name, region, (left, top)) in enumerate(reversed(tuples)):
        src = f"data/{idx:02d}_{name}.png"
        layer_pngs[src] = region
        ET.SubElement(
            stack_el,
            "layer",
            {
                "name": _clean_name(name),
                "src": src,
                "x": str(left),
                "y": str(top),
                "opacity": "1.0",
                "visibility": "visible",
                "composite-op": "svg:src-over",
            },
        )

    # Merged preview (flattened) for thumbnails.
    merged = compose_flat(assets) or Image.new("RGBA", (cw or 1, ch or 1), (0, 0, 0, 0))

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "image/openraster", compress_type=zipfile.ZIP_STORED)
        zf.writestr("stack.xml", ET.tostring(image_el, encoding="unicode"))
        for src, img in layer_pngs.items():
            with zf.open(src, "w") as fh:
                img.save(fh, format="PNG")
        thumb = merged.convert("RGBA").copy()
        thumb.thumbnail((256, 256))
        with zf.open("Thumbnails/thumbnail.png", "w") as fh:
            thumb.save(fh, format="PNG")
        with zf.open("mergedimage.png", "w") as fh:
            merged.convert("RGBA").save(fh, format="PNG")

    logger.info("Wrote layered ORA: %s (%d layers)", output_path, len(tuples))
    return output_path


def compose_flat(assets: List[Asset]) -> Optional[Image.Image]:
    """Composite all layers back-to-front into a single flattened RGBA image."""
    tuples, (cw, ch) = _build_layer_tuples(assets)
    if not tuples or cw == 0:
        return None
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    for _name, region, (left, top) in tuples:
        canvas.alpha_composite(region, dest=(left, top))
    return canvas


def export_layered_documents(assets: List[Asset], out_dir: Path, model_name: str) -> Dict[str, str]:
    """Write every layered artifact and return a map of produced files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: Dict[str, str] = {}

    psd_path = export_psd(assets, out_dir / f"{model_name}.psd")
    if psd_path:
        produced["psd"] = str(psd_path)

    ora_path = export_ora(assets, out_dir / f"{model_name}.ora")
    if ora_path:
        produced["ora"] = str(ora_path)

    flat = compose_flat(assets)
    if flat is not None:
        flat_path = out_dir / f"{model_name}_flat.png"
        flat.save(flat_path)
        produced["flat"] = str(flat_path)

    return produced
