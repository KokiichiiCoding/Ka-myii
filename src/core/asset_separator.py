"""
Asset separator — decomposes a character image into named, rig-ready layers.

This is the stage that turns a flat illustration into the separated parts a
Live2D rigger needs (hair front/back, head, eyes L/R, brows, mouth, body,
clothing, accessories …), each on its own transparent layer aligned to the
original canvas.

Backends, in priority order (best available wins):

1. **Region map** — when the image was produced by Ka-myii's demo renderer a
   pixel-perfect ``<image>.regions.png`` sidecar is present and gives a clean,
   exact decomposition.
2. **SAM 2** — Meta's Segment Anything 2 automatic masks (if ``sam2`` and a
   checkpoint are installed), classified into semantic layers.
3. **rembg + anatomy** — a foreground cut-out (rembg) intersected with an
   anatomical region template.
4. **Anatomy heuristic** — always-available geometric fallback so the pipeline
   never fails.

Occluded base layers (head behind hair, body under clothes) are hole-filled so
each exported layer is complete — the core idea behind modern single-image
decomposition research ("See-through", SIGGRAPH 2026).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import config
from src.core.demo_character import REGION_COLORS
from src.models.vtuber_model import Asset

try:
    from rembg import remove as rembg_remove

    REMBG_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    REMBG_AVAILABLE = False

logger = logging.getLogger(__name__)

# Layers that should be solid/complete (hole-filled) vs. feature overlays.
BASE_LAYERS = {"hair_back", "body", "clothing", "head", "face", "hair_front", "ear_L", "ear_R"}
FEATURE_LAYERS = {"brow_L", "brow_R", "eye_L", "eye_R", "nose", "mouth", "accessories"}


class AssetSeparator:
    """Decomposes a character image into aligned, named, transparent layers."""

    def __init__(self, use_ai_segmentation: bool = True):
        self.use_ai_segmentation = use_ai_segmentation
        self.layer_order: List[str] = config.ASSET_SEPARATION.get("layers", [])
        self.inpaint_occluded = config.ASSET_SEPARATION.get("inpaint_occluded", True)
        logger.info("AssetSeparator ready (ai=%s, inpaint=%s)", use_ai_segmentation, self.inpaint_occluded)

    # ------------------------------------------------------------------
    def separate(
        self,
        image_path: Path,
        output_dir: Path,
        layers: Optional[List[str]] = None,
    ) -> List[Asset]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path).convert("RGBA")
        layers = layers or self.layer_order
        layers = [layer for layer in layers if layer != "background"]

        masks, backend = self._compute_masks(image, image_path, layers)
        logger.info("Decomposition backend: %s", backend)

        assets: List[Asset] = []
        for order, name in enumerate(layers):
            mask = masks.get(name)
            if mask is None or not mask.any():
                continue
            layer_img = self._extract_layer(image, mask, solidify=name in BASE_LAYERS)
            bbox = self._mask_bbox(mask)
            layer_path = output_dir / f"{order:02d}_{name}.png"
            layer_img.save(layer_path)
            assets.append(
                Asset(
                    layer_type=name,
                    file_path=layer_path,
                    metadata={
                        "order": order,
                        "bbox": bbox,
                        "backend": backend,
                        "is_base": name in BASE_LAYERS,
                        "pixels": int(mask.sum()),
                    },
                )
            )
            logger.info("Extracted layer %-12s (%d px)", name, int(mask.sum()))

        # Always provide a clean, background-removed full-character cut-out too.
        cutout = self._foreground_cutout(image, image_path)
        cutout_path = output_dir / "character.png"
        cutout.save(cutout_path)
        assets.append(Asset(layer_type="character", file_path=cutout_path, metadata={"order": 999}))

        logger.info("Separation produced %d layers", len(assets))
        return assets

    # ------------------------------------------------------------------
    # Mask computation backends
    # ------------------------------------------------------------------
    def _compute_masks(self, image: Image.Image, image_path: Path, layers: List[str]):
        # 1. Region-map sidecar (demo renderer) -> pixel perfect.
        sidecar = image_path.with_suffix(".regions.png")
        if sidecar.exists():
            try:
                return self._masks_from_region_map(sidecar, image.size, layers), "region-map"
            except Exception as exc:  # pragma: no cover
                logger.warning("Region map decode failed (%s); falling back", exc)

        # 2. SAM 2 (optional, only if installed and configured).
        if self.use_ai_segmentation:
            sam_masks = self._masks_from_sam2(image, layers)
            if sam_masks:
                return sam_masks, "sam2"

        # 3 & 4. Anatomy template intersected with a foreground alpha.
        return self._masks_from_anatomy(image, image_path, layers), (
            "rembg+anatomy" if REMBG_AVAILABLE else "anatomy"
        )

    def _masks_from_region_map(self, sidecar: Path, size, layers: List[str]) -> Dict[str, np.ndarray]:
        region = Image.open(sidecar).convert("RGB").resize(size, Image.Resampling.NEAREST)
        arr = np.asarray(region)
        masks: Dict[str, np.ndarray] = {}
        for name in layers:
            color = REGION_COLORS.get(name)
            if color is None:
                continue
            masks[name] = np.all(arr == np.array(color), axis=-1)
        return masks

    def _masks_from_sam2(self, image: Image.Image, layers: List[str]) -> Optional[Dict[str, np.ndarray]]:
        try:
            import torch  # noqa: WPS433
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except Exception:
            return None

        checkpoint = config.ASSET_SEPARATION.get("sam2_checkpoint")
        if not checkpoint or not Path(checkpoint).exists():
            logger.info("SAM2 installed but no checkpoint configured; skipping")
            return None

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = build_sam2(config.ASSET_SEPARATION.get("sam2_model_cfg"), checkpoint, device=device)
            generator = SAM2AutomaticMaskGenerator(sam)
            raw = generator.generate(np.asarray(image.convert("RGB")))
            return self._classify_sam_masks(raw, image.size, layers)
        except Exception as exc:  # pragma: no cover
            logger.warning("SAM2 segmentation failed: %s", exc)
            return None

    def _classify_sam_masks(self, raw_masks, size, layers) -> Dict[str, np.ndarray]:
        """Assign SAM masks to semantic layers by position/size heuristics."""
        w, h = size
        template = self._anatomy_template(w, h)
        masks: Dict[str, np.ndarray] = {}
        for m in sorted(raw_masks, key=lambda x: x.get("area", 0), reverse=True):
            seg = m["segmentation"].astype(bool)
            ys, xs = np.where(seg)
            if len(xs) == 0:
                continue
            cy, cx = ys.mean() / h, xs.mean() / w
            best, best_score = None, -1.0
            for name, tmpl in template.items():
                score = float((seg & tmpl).sum()) / float(seg.sum() + 1)
                if score > best_score:
                    best, best_score = name, score
            if best and best_score > 0.2:
                masks[best] = masks.get(best, np.zeros_like(seg)) | seg
        return masks

    def _masks_from_anatomy(self, image: Image.Image, image_path: Path, layers: List[str]) -> Dict[str, np.ndarray]:
        """Geometric anatomical template intersected with the foreground."""
        w, h = image.size
        cutout = self._foreground_cutout(image, image_path)
        alpha = np.asarray(cutout.split()[-1]) > 16
        template = self._anatomy_template(w, h)
        masks: Dict[str, np.ndarray] = {}
        for name in layers:
            tmpl = template.get(name)
            if tmpl is None:
                continue
            if name in ("hair_back",):
                masks[name] = tmpl & ~alpha if False else tmpl  # hair back may extend beyond alpha
            else:
                masks[name] = tmpl & alpha
        return masks

    def _anatomy_template(self, w: int, h: int) -> Dict[str, np.ndarray]:
        """Build a coarse front-facing anatomy template as boolean masks.

        Proportions match the demo renderer so they line up well on Ka-myii
        output and give a sensible approximation on arbitrary portraits.
        """
        cx = w / 2
        head_r = w * 0.20
        head_cy = h * 0.30
        eye_dx = head_r * 0.45
        eye_cy = head_cy + head_r * 0.12
        eye_rx, eye_ry = head_r * 0.24, head_r * 0.30
        brow_cy = head_cy - head_r * 0.30
        mouth_cy = head_cy + head_r * 0.62
        nose_cy = head_cy + head_r * 0.38
        ear_cy = head_cy + head_r * 0.05
        shoulder_y = head_cy + head_r * 1.25

        def ellipse(cxx, cyy, rx, ry):
            yy, xx = np.ogrid[:h, :w]
            return ((xx - cxx) / rx) ** 2 + ((yy - cyy) / ry) ** 2 <= 1.0

        def rect(x0, y0, x1, y1):
            m = np.zeros((h, w), dtype=bool)
            m[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = True
            return m

        t: Dict[str, np.ndarray] = {}
        t["hair_back"] = ellipse(cx, head_cy + head_r * 0.8, head_r * 1.5, head_r * 2.3)
        t["body"] = rect(cx - head_r * 0.3, head_cy + head_r * 0.7, cx + head_r * 0.3, shoulder_y + 4)
        t["clothing"] = rect(cx - w * 0.22, shoulder_y, cx + w * 0.22, h) & ~rect(0, 0, w, shoulder_y)
        t["head"] = ellipse(cx, head_cy, head_r, head_r * 1.05)
        t["face"] = ellipse(cx, head_cy + head_r * 0.25, head_r * 0.82, head_r * 0.75)
        t["ear_L"] = ellipse(cx - head_r * 0.95, ear_cy, head_r * 0.14, head_r * 0.2)
        t["ear_R"] = ellipse(cx + head_r * 0.95, ear_cy, head_r * 0.14, head_r * 0.2)
        t["brow_L"] = ellipse(cx - eye_dx, brow_cy, eye_rx, head_r * 0.06)
        t["brow_R"] = ellipse(cx + eye_dx, brow_cy, eye_rx, head_r * 0.06)
        t["eye_L"] = ellipse(cx - eye_dx, eye_cy, eye_rx, eye_ry)
        t["eye_R"] = ellipse(cx + eye_dx, eye_cy, eye_rx, eye_ry)
        t["nose"] = ellipse(cx, nose_cy, head_r * 0.08, head_r * 0.1)
        t["mouth"] = ellipse(cx, mouth_cy, head_r * 0.2, head_r * 0.12)
        t["hair_front"] = ellipse(cx, head_cy - head_r * 0.45, head_r * 1.02, head_r * 0.75)
        t["accessories"] = ellipse(cx + head_r * 0.7, head_cy - head_r * 0.75, head_r * 0.18, head_r * 0.18)

        # Make feature regions exclusive of the larger base masks above them.
        for feat in ("eye_L", "eye_R", "brow_L", "brow_R", "nose", "mouth"):
            t["face"] = t["face"] & ~t[feat]
            t["head"] = t["head"] & ~t[feat]
        t["head"] = t["head"] & ~t["face"]
        return t

    # ------------------------------------------------------------------
    # Layer extraction helpers
    # ------------------------------------------------------------------
    def _extract_layer(self, image: Image.Image, mask: np.ndarray, solidify: bool) -> Image.Image:
        rgba = np.asarray(image).copy()
        out = np.zeros_like(rgba)
        out[mask] = rgba[mask]
        out[..., 3] = np.where(mask, 255, 0)

        if solidify and self.inpaint_occluded:
            filled = self._fill_holes(mask)
            holes = filled & ~mask
            if holes.any():
                median = self._median_color(rgba, mask)
                out[holes, 0:3] = median
                out[holes, 3] = 255
        return Image.fromarray(out, "RGBA")

    @staticmethod
    def _median_color(rgba: np.ndarray, mask: np.ndarray):
        pixels = rgba[mask]
        if len(pixels) == 0:
            return np.array([200, 200, 200], dtype=np.uint8)
        return np.median(pixels[:, :3], axis=0).astype(np.uint8)

    @staticmethod
    def _fill_holes(mask: np.ndarray) -> np.ndarray:
        """Fill interior holes of a boolean mask using a fast PIL flood fill."""
        h, w = mask.shape
        padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = mask.astype(np.uint8) * 255
        img = Image.fromarray(255 - padded)  # 255 = background
        ImageDraw.floodfill(img, (0, 0), 128)  # exterior background -> 128
        arr = np.asarray(img)
        holes = arr == 255  # background not reached from the border == interior holes
        filled = padded > 0
        filled |= holes
        return filled[1:-1, 1:-1]

    @staticmethod
    def _mask_bbox(mask: np.ndarray):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return [0, 0, 0, 0]
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    def _foreground_cutout(self, image: Image.Image, image_path: Path) -> Image.Image:
        """Return the character on a transparent background."""
        # Prefer the demo region map (everything that isn't background).
        sidecar = image_path.with_suffix(".regions.png")
        if sidecar.exists():
            region = Image.open(sidecar).convert("RGB").resize(image.size, Image.Resampling.NEAREST)
            arr = np.asarray(region)
            bg = np.all(arr == np.array(REGION_COLORS["background"]), axis=-1)
            out = np.asarray(image).copy()
            out[..., 3] = np.where(bg, 0, 255)
            return Image.fromarray(out, "RGBA")

        if REMBG_AVAILABLE and self.use_ai_segmentation:
            try:
                return rembg_remove(image).convert("RGBA")
            except Exception as exc:  # pragma: no cover
                logger.warning("rembg failed: %s", exc)

        # Fallback: treat near-uniform border colour as background.
        return self._chroma_cutout(image)

    @staticmethod
    def _chroma_cutout(image: Image.Image) -> Image.Image:
        arr = np.asarray(image.convert("RGBA")).copy()
        h, w = arr.shape[:2]
        border = np.concatenate([arr[0, :, :3], arr[-1, :, :3], arr[:, 0, :3], arr[:, -1, :3]])
        bg_color = np.median(border, axis=0)
        dist = np.linalg.norm(arr[..., :3].astype(int) - bg_color, axis=-1)
        arr[..., 3] = np.where(dist < 28, 0, 255)
        return Image.fromarray(arr, "RGBA")


class DummyAssetSeparator(AssetSeparator):
    """In demo mode the real separator already works (no GPU needed), and the
    region-map sidecar makes it pixel-perfect — so we simply reuse it."""

    pass
