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
1b. **See-through** (SIGGRAPH 2026) — when installed in ``engines/see-through``,
   produces 20+ semantic layers with occluded regions genuinely inpainted.
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
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

import config
from src.core import seethrough
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

# Anatomical draw-order priors (lower, upper) for See-through tags. See-through stacks parts by
# predicted depth, which gets thin, nearly coplanar parts wrong — seen on a real run: an iris
# under its eye white and the neck over the hoodie (the colour check then deleted both real parts).
# A "-l"/"-r" part only pairs with the same side.
ORDER_PRIORS = [
    ("back hair", "neck"), ("back hair", "topwear"), ("back hair", "face"),
    ("neck", "face"), ("neck", "topwear"), ("neck", "neckwear"), ("topwear", "neckwear"),
    ("face", "eyewhite"), ("eyewhite", "irides"), ("irides", "eyelash"),
    ("face", "eyebrow"), ("face", "nose"), ("face", "mouth"),
    # Seen: front hair under the face — the source-colour step then baked the bangs INTO the
    # face layer — and a witch hat under the face, dropped by the colour check. Brows are left
    # to depth: Live2D artists often draw them over the bangs.
    ("face", "front hair"), ("eyewhite", "front hair"), ("irides", "front hair"),
    ("eyelash", "front hair"), ("nose", "front hair"), ("mouth", "front hair"),
    ("front hair", "headwear"),
]
# See-through emits near-empty accessory layers (a few noise pixels) when there is none;
# each would still cost an ArtMesh in Cubism. Facial parts are legitimately tiny, so only these.
ACCESSORY_TAGS = {"headwear", "eyewear", "earwear", "neckwear", "objects", "tail", "wings"}


# SkyTNT's anime-segmentation IS-Net (Apache-2.0; the model rembg calls "isnet-anime"), run with
# onnxruntime. Flat-colour keying fails on gradient/vignette backgrounds and on clothing that
# touches the frame; this model was trained to cut anime characters out of anything.
ANIME_SEG = ("skytnt/anime-seg", "isnetis.onnx", "493cb60893f47441b26ec4fb9a306bce9e342982")


def _anime_seg_mask(image: Image.Image) -> Optional[np.ndarray]:
    """Character alpha (uint8, image size), or None when onnxruntime or the 176 MB weights are
    unavailable. Weights are only read from the local Hugging Face cache — the See-through
    installer fetches them — so a missing model never stalls a job on a download."""
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        repo, name, revision = ANIME_SEG
        path = hf_hub_download(repo, name, revision=revision, local_files_only=True)
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as exc:
        logger.info("Anime segmentation unavailable (%s); using the colour-key cut-out", type(exc).__name__)
        return None
    side = 1024  # the model's fixed input: the image letterboxed into a black square
    w0, h0 = image.size
    scale = side / max(w0, h0)
    w, h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
    px, py = (side - w) // 2, (side - h) // 2
    batch = np.zeros((side, side, 3), np.float32)
    batch[py:py + h, px:px + w] = np.asarray(image.convert("RGB").resize((w, h), Image.Resampling.BILINEAR), np.float32) / 255
    mask = session.run(None, {"img": batch.transpose(2, 0, 1)[None]})[0][0, 0, py:py + h, px:px + w]
    mask = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8)).resize((w0, h0), Image.Resampling.BILINEAR)
    return np.asarray(mask)


def _tag_side(name: str):
    return (name[:-2], name[-1]) if name.endswith(("-l", "-r")) else (name, None)


def apply_order_priors(names: List[str]) -> List[str]:
    """Return ``names`` (bottom to top) with each violated prior fixed by moving the upper part
    just above the lower one; every other relative order (See-through's depth) is kept."""
    order = list(names)
    for _ in range(len(order) * len(ORDER_PRIORS)):  # bounded; the priors are acyclic
        moved = False
        for lower, upper in ORDER_PRIORS:
            for a in list(order):
                for b in list(order):
                    (ta, sa), (tb, sb) = _tag_side(a), _tag_side(b)
                    if ta == lower and tb == upper and (sa is None or sb is None or sa == sb)                             and order.index(b) < order.index(a):
                        order.remove(b)
                        order.insert(order.index(a) + 1, b)
                        moved = True
        if not moved:
            break
    return order


class AssetSeparator:
    """Decomposes a character image into aligned, named, transparent layers."""

    def __init__(self, use_ai_segmentation: bool = True):
        self.use_ai_segmentation = use_ai_segmentation
        self.cutout_backend = None  # set by _foreground_cutout; "anime-seg" means the mask is trustworthy
        self.layer_order: List[str] = config.ASSET_SEPARATION.get("layers", [])
        self.inpaint_occluded = config.ASSET_SEPARATION.get("inpaint_occluded", True)
        logger.info("AssetSeparator ready (ai=%s, inpaint=%s)", use_ai_segmentation, self.inpaint_occluded)

    # ------------------------------------------------------------------
    def separate(
        self,
        image_path: Path,
        output_dir: Path,
        layers: Optional[List[str]] = None,
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> List[Asset]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path).convert("RGBA")
        layers = layers or self.layer_order
        layers = [layer for layer in layers if layer != "background"]

        cutout = self._foreground_cutout(image, image_path)
        has_region_map = image_path.with_suffix(".regions.png").exists()
        if not has_region_map and self.use_ai_segmentation and seethrough.available():
            assets = self._separate_seethrough(image_path, output_dir, progress, cutout)
        else:
            assets = self._separate_masks(image, image_path, output_dir, layers, progress)

        # Always provide a clean, background-removed full-character cut-out too.
        cutout_path = output_dir / "character.png"
        cutout.save(cutout_path)
        assets.append(Asset(layer_type="character", file_path=cutout_path, metadata={"order": 999}))

        logger.info("Separation produced %d layers", len(assets))
        return assets

    def _separate_seethrough(self, image_path: Path, output_dir: Path, progress, cutout: Image.Image) -> List[Asset]:
        """Inpainted semantic layers from See-through, mapped back onto the source canvas."""
        from psd_tools import PSDImage

        psd = PSDImage.open(seethrough.decompose(image_path, output_dir / "seethrough", progress))
        # See-through works on a centred square pad of the input, resized to its resolution.
        source = Image.open(image_path).convert("RGBA")
        w, h = source.size
        side = max(w, h)
        px, py = (side - w) // 2, (side - h) // 2
        # Real parts (even inpainted, hidden ones) sit inside the character's silhouette;
        # See-through occasionally hallucinates parts (e.g. headwear) over empty background.
        fg_alpha = np.asarray(cutout.split()[-1])
        silhouette = fg_alpha > 16
        # Chroma/rembg masks can contain holes over pale skin or facial details.
        # Interior holes are still inside the character, not floating hallucinations.
        silhouette = self._fill_holes(silhouette)
        silhouette = np.asarray(Image.fromarray(silhouette).filter(ImageFilter.MaxFilter(9)))

        layers = []
        for layer in psd:  # psd-tools iterates bottom (farthest) to top
            pixels = layer.topil()
            if pixels is None:
                continue
            canvas = Image.new("RGBA", psd.size, (0, 0, 0, 0))
            canvas.paste(pixels.convert("RGBA"), (layer.left, layer.top))
            if psd.size != (side, side):
                canvas = canvas.resize((side, side), Image.Resampling.LANCZOS)
            layers.append((layer.name, canvas.crop((px, py, px + w, py + h))))
        pool: Dict[str, list] = {}
        for name, canvas in layers:
            pool.setdefault(name, []).append(canvas)
        layers = [(order, name, pool[name].pop(0))
                  for order, name in enumerate(apply_order_priors([n for n, _ in layers]))]

        # Where a layer is the visible top layer, it should reproduce the source pixels.
        # shortcut: thresholds tuned on See-through's sample image; revisit if real parts get dropped.
        src_rgb = np.asarray(source.convert("RGB")).astype(np.int16)
        covered = np.zeros((h, w), dtype=bool)
        keep = set()
        for order, name, canvas in reversed(layers):  # front to back
            arr = np.asarray(canvas).astype(np.int16)
            alpha = arr[..., 3] > 16
            if not alpha.any():
                continue
            if name in ACCESSORY_TAGS and alpha.sum() < 0.0005 * w * h:
                logger.info("Dropping empty accessory layer %s (%d px)", name, int(alpha.sum()))
                continue
            visible = (arr[..., 3] > 128) & ~covered
            outside = float((alpha & ~silhouette).sum()) / alpha.sum()
            err = float(np.abs(arr[..., :3] - src_rgb)[visible].mean()) if visible.any() else 0.0
            if outside > 0.5 or (alpha.sum() > 0.002 * w * h and err > 60):
                logger.info("Dropping hallucinated layer %s (%.0f%% outside character, colour error %.0f)", name, outside * 100, err)
                continue
            keep.add(order)
            covered |= arr[..., 3] > 128

        kept = [(name, canvas) for order, name, canvas in layers if order in keep]
        if not kept:
            raise RuntimeError("See-through produced no layers")
        # A segmentation-model mask is reliable enough to clear layer pixels over plain background
        # (seen: a hallucinated ribbon blob beside the head); colour keying can cut real white
        # clothing, so it is only trusted for the whole-layer checks above.
        background = ~silhouette if self.cutout_backend == "anime-seg" else None
        covered = self._resolve_ownership(kept, src_rgb, background)
        leftover = self._leftover_layer(source, fg_alpha, covered)
        if leftover is not None:
            # Uncovered pixels look identical at any depth; just above the back hair suits the
            # usual case (cat ears, ahoge, hair ornaments) when the head moves.
            at = next((i + 1 for i, (name, _) in enumerate(kept) if name == "back hair"), 0)
            kept.insert(at, ("leftover", leftover))

        assets: List[Asset] = []
        for order, (name, canvas) in enumerate(kept):
            alpha = np.asarray(canvas.split()[-1]) > 16
            layer_path = output_dir / f"{order:02d}_{name}.png"
            canvas.save(layer_path)
            assets.append(
                Asset(
                    layer_type=name,
                    file_path=layer_path,
                    metadata={
                        "order": order,
                        "bbox": self._mask_bbox(alpha),
                        "backend": "see-through",
                        "pixels": int(alpha.sum()),
                    },
                )
            )
            logger.info("Extracted layer %-12s (%d px)", name, int(alpha.sum()))
        return assets

    @staticmethod
    def _resolve_ownership(kept: list, src_rgb: np.ndarray, background: Optional[np.ndarray] = None,
                           margin: int = 30) -> np.ndarray:
        """Give every pixel to the part the source actually shows there, in place.

        Per pixel the owner is the opaque part whose predicted colour best matches the source —
        the topmost within ``margin`` of the best, so See-through's order stands unless clearly
        wrong. Parts stacked above the owner are cut there: See-through over-extends them (seen:
        bangs predicted across the cheeks, drawn as skin patches with hair-coloured outlines).
        The owner then takes the source colour, so the artist's visible colours are kept and
        diffusion only invents what is hidden. Returns the union of opaque pixels.
        """
        arrs = [np.asarray(canvas).astype(np.int16) for _, canvas in kept]  # bottom to top
        if background is not None:  # no part may cover what the source shows as background
            for (name, _), a in zip(kept, arrs):
                spill = background & (a[..., 3] > 0)
                if spill.sum() > 0:
                    a[spill, 3] = 0
                    logger.info("Ownership: cleared %d px of %s over the background", int(spill.sum()), name)
        opaque = np.stack([a[..., 3] > 128 for a in arrs])
        err = np.stack([np.abs(a[..., :3] - src_rgb).mean(axis=-1) for a in arrs])
        err = np.where(opaque, err, np.inf)
        acceptable = err <= err.min(axis=0) + margin
        owner = len(arrs) - 1 - np.argmax(acceptable[::-1], axis=0)  # topmost acceptable part
        claimed = opaque.any(axis=0)
        for i, (name, canvas) in enumerate(kept):
            a = arrs[i]
            above = claimed & (owner < i)
            cut = above & opaque[i]
            if cut.any():
                # ...and the semi-transparent fringe around the cut, which would draw the outline.
                near = np.asarray(Image.fromarray(cut.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5))) > 0
                a[cut | (near & above), 3] = 0
                logger.info("Ownership: cut %d px of %s that the source shows belong to a part below", int(cut.sum()), name)
            mine = claimed & (owner == i)
            a[mine, :3] = src_rgb[mine]
            canvas.paste(Image.fromarray(a.astype(np.uint8)))
        return np.stack([a[..., 3] > 128 for a in arrs]).any(axis=0)

    @staticmethod
    def _leftover_layer(source: Image.Image, fg_alpha: np.ndarray, covered: np.ndarray) -> Optional[Image.Image]:
        """Character pixels no See-through layer covers (seen: cat ears it assigned to nothing),
        so the PSD still holds the whole character. ``None`` when only slivers remain."""
        w, h = source.size
        # Background-coloured pixels here are background seen through gaps (e.g. between hair
        # strands) that the border-connected cut-out keeps, not a part See-through missed.
        background = AssetSeparator._background_like(np.asarray(source.convert("RGBA")))
        mask = Image.fromarray(((fg_alpha > 128) & ~covered & ~background).astype(np.uint8) * 255)
        # Opening drops the 1–2 px anti-aliasing fringe along every part's outline.
        mask = mask.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
        area = int((np.asarray(mask) > 0).sum())
        if area < 0.001 * w * h:
            return None
        if area > 0.25 * w * h:  # more likely a failed background cut-out than missing parts
            logger.warning("Skipping leftover layer: %d uncovered px looks like background", area)
            return None
        layer = source.copy()
        layer.putalpha(Image.fromarray(np.minimum(np.asarray(mask), fg_alpha)))
        logger.info("Leftover layer keeps %d character px no part covered", area)
        return layer

    def _separate_masks(self, image: Image.Image, image_path: Path, output_dir: Path, layers: List[str],
                        progress: Optional[Callable[[float, str], None]] = None) -> List[Asset]:
        masks, backend = self._compute_masks(image, image_path, layers)
        logger.info("Decomposition backend: %s", backend)

        assets: List[Asset] = []
        for order, name in enumerate(layers):
            mask = masks.get(name)
            if mask is None or not mask.any():
                continue
            if progress:
                progress(order / len(layers), f"extracting {name.replace('_', ' ')}")
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
        # .copy(): Pillow 12's fromarray shares the buffer, so floodfill edits would not show in np.asarray
        img = Image.fromarray(255 - padded).copy()  # 255 = background
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
        self.cutout_backend = "heuristic"
        # Prefer the demo region map (everything that isn't background).
        sidecar = image_path.with_suffix(".regions.png")
        if sidecar.exists():
            region = Image.open(sidecar).convert("RGB").resize(image.size, Image.Resampling.NEAREST)
            arr = np.asarray(region)
            bg = np.all(arr == np.array(REGION_COLORS["background"]), axis=-1)
            out = np.asarray(image).copy()
            out[..., 3] = np.where(bg, 0, 255)
            return Image.fromarray(out, "RGBA")

        if self.use_ai_segmentation:
            mask = _anime_seg_mask(image)
            if mask is not None:
                self.cutout_backend = "anime-seg"
                out = np.asarray(image.convert("RGBA")).copy()
                out[..., 3] = mask
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
        """Remove the flat background every style preset asks for ("white background").

        The background colour is the most common border colour — not the median, which blends
        into neither colour once an upper-body crop puts clothing on the border — and only its
        region connected to the border is removed, so white eye highlights stay opaque.
        shortcut: assumes a flat background; detailed backgrounds need rembg/anime segmentation.
        """
        arr = np.asarray(image.convert("RGBA")).copy()
        solid = ~AssetSeparator._background_like(arr)
        # Flood the background in from the top and sides only: framing crops the body at the
        # bottom edge, and a white shirt touching it would otherwise be flooded away as well.
        solid[-1, :] = True
        solid = AssetSeparator._fill_holes(solid)
        solid[-1, :] = solid[-2, :]
        arr[..., 3] = np.where(solid, 255, 0)
        return Image.fromarray(arr, "RGBA")

    @staticmethod
    def _background_like(arr: np.ndarray) -> np.ndarray:
        """Pixels close to the most common border colour (the flat preset background)."""
        border = np.concatenate([arr[0, :, :3], arr[-1, :, :3], arr[:, 0, :3], arr[:, -1, :3]])
        colours, counts = np.unique(border // 8, axis=0, return_counts=True)
        bg_color = colours[counts.argmax()] * 8 + 4
        return np.linalg.norm(arr[..., :3].astype(int) - bg_color, axis=-1) < 28


class DummyAssetSeparator(AssetSeparator):
    """In demo mode the real separator already works (no GPU needed), and the
    region-map sidecar makes it pixel-perfect — so we simply reuse it."""

    pass
