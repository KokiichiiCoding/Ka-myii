"""Separate browser-rig PSD export; never alters the artist's Cubism document."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageOps

EXPORT_VERSION = 16  # 16: alternate hair/outfit toggles; 15: A/I/U/E/O mouths; 14: expression toggle layers; 13: heal mouth sockets for narrow O / toothy smile; 12: clean cloth (no strand copies or cracks) under hair; 10: neck shows under a moving choker; 8: opaque healed eye sockets
_export_lock = threading.Lock()
_ALIASES = {
    'hair_back': 'back hair', 'hair_front': 'front hair', 'head': 'face',
    'body': 'neck', 'clothing': 'topwear', 'accessories': 'headwear',
    'eye_l': 'eyewhite', 'eye_r': 'eyewhite',
    'brow_l': 'eyebrow', 'brow_r': 'eyebrow',
    'ear_l': 'ears', 'ear_r': 'ears', 'mouth': 'mouth_open',
    'leftover': 'objects',  # uncovered character pixels; the rig groups them by position
}


def preview_name(name: str) -> str:
    name = re.sub(r'-(?:l|r)$', '', name.strip().lower())
    return _ALIASES.get(name, name)


def preview_layers(assets_dir: Path):
    """Merge aligned pairs at their last original position in painter's order."""
    merged = {}
    size = None
    for path in sorted(assets_dir.glob('*.png')):
        name = re.sub(r'^\d+_', '', path.stem)
        if name in ('character', 'background'):
            continue
        name = preview_name(name)
        with Image.open(path) as source:
            image = source.convert('RGBA')
        if size is None:
            size = image.size
        if image.size != size:
            raise ValueError('Preview layers must share the same aligned canvas')
        if name in merged:
            image = Image.alpha_composite(merged.pop(name), image)
        merged[name] = image
    items = [(name, image) for name, image in merged.items() if image.getbbox()]
    return _with_expressions(_clean_hidden_skin(items), Path(assets_dir).parent, size)


def _blur_skin(values, radius):
    """Three separable box passes, in float precision (8-bit masks cause colour rings)."""
    radius = max(1, round(radius - 0.5))
    width = radius * 2 + 1
    result = values.astype(np.float64)
    for _ in range(3):
        for axis in (0, 1):
            padding = [(0, 0)] * result.ndim
            padding[axis] = (radius, radius)
            padded = np.pad(result, padding, mode="edge")
            prefix = np.cumsum(padded, axis=axis)
            shape = list(prefix.shape)
            shape[axis] = 1
            prefix = np.concatenate((np.zeros(shape), prefix), axis=axis)
            hi, lo = [slice(None)] * result.ndim, [slice(None)] * result.ndim
            hi[axis], lo[axis] = slice(width, None), slice(None, -width)
            result = (prefix[tuple(hi)] - prefix[tuple(lo)]) / width
    return result


def _clean_hidden_skin(items):
    """Repair preview skin beneath moving hair and facial parts.

    Decomposition leaves faint copies of bangs/eyes/brows in the face. They are invisible at
    rest but appear when those parts move. Hair repair stays under fully opaque paint; facial
    repair includes a small halo to remove ghost outlines at translucent eyelash/brow edges.
    Normalized blur interpolates local skin without drawing ink into the fill. This affects
    only the disposable preview PSD; artist assets and their exports remain unchanged.
    """
    covering = {"front hair", "side hair", "headwear", "eyewhite", "irides", "eyelash",
                "eyebrow", "nose", "mouth_open", "mouth_close"}
    out = list(items)
    for index, (name, image) in enumerate(items):
        if name != "face":
            continue
        face = np.array(image)
        cover = np.zeros(face.shape[:2], np.uint8)
        features = np.zeros(face.shape[:2], np.uint8)
        eyes = np.zeros(face.shape[:2], np.uint8)
        for role, part in items[index + 1:]:
            if role in covering:
                alpha = np.asarray(part)[..., 3]
                cover = np.maximum(cover, alpha)
                if role in {"eyewhite", "irides", "eyelash", "eyebrow", "mouth_open", "mouth_close"}:
                    features = np.maximum(features, alpha)
                    if role in {"eyewhite", "irides", "eyelash"}:
                        eyes = np.maximum(eyes, alpha)
        _, xs = np.nonzero(face[..., 3] > 240)
        if not len(xs):
            continue
        pad = max(2, round((xs.max() - xs.min()) * 0.014))
        facial_area = np.asarray(Image.fromarray((features > 8).astype(np.uint8) * 255).filter(
            ImageFilter.MaxFilter(pad * 2 + 1))) > 0
        # The whole eye socket must be clean under a repositioned lid, including gaps
        # between white/iris/lash masks. Keep the surrounding face outside these ellipses.
        sockets = Image.new("L", image.size)
        draw = ImageDraw.Draw(sockets)
        cx = (xs.min() + xs.max()) / 2
        for left in (True, False):
            mask = eyes > 8
            if left:
                mask[:, int(cx):] = False
            else:
                mask[:, :int(cx)] = False
            ey, ex = np.nonzero(mask)
            if len(ex) < 10:
                continue
            px, py = (ex.max() - ex.min()) * .12 + 4, (ey.max() - ey.min()) * .18 + 4
            draw.ellipse((ex.min() - px, ey.min() - py, ex.max() + px, ey.max() + py), fill=255)
        facial_area |= np.asarray(sockets) > 0
        # The separator also punches transparent ink-shaped holes in the face. Leaving
        # those holes shows dark back hair through a closed eye even after RGB cleanup.
        silhouette = ImageOps.expand(Image.fromarray((face[..., 3] > 16).astype(np.uint8) * 255), 1)
        ImageDraw.floodfill(silhouette, (0, 0), 128)
        interior = np.asarray(silhouette)[1:-1, 1:-1] != 128
        repaired_socket = facial_area & interior
        hidden = ((cover == 255) & (face[..., 3] > 240)) | repaired_socket
        known = (cover < 8) & ~facial_area & (face[..., 3] > 240)
        if not _fill_hidden(face, hidden, known, float(max(4, (xs.max() - xs.min()) * 0.055))):
            continue
        face[repaired_socket, 3] = 255
        out[index] = (name, Image.fromarray(face))
    # Shoulders shrug and the collar bends under hanging hair. Clothing hidden beneath opaque
    # hair holds faint copies of the strands, and the separator cut the hair's outline out of
    # it as thin transparent cracks; both get the surrounding cloth's colour instead.
    grow = lambda mask, size, f=ImageFilter.MaxFilter: np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(f(size))) > 0
    for index, (name, image) in enumerate(out):
        if name not in ("topwear", "handwear"):
            continue
        hair = np.zeros(image.size[::-1], np.uint8)
        for role, part in out[index + 1:]:
            if role in ("front hair", "side hair"):
                hair = np.maximum(hair, np.asarray(part)[..., 3])
        cloth = np.array(image)
        opaque = cloth[..., 3] > 240
        cracks = grow(grow(opaque, 5), 5, ImageFilter.MinFilter) & ~opaque & grow(hair > 8, 5)
        if _fill_hidden(cloth, ((hair == 255) & opaque) | cracks, (hair < 8) & opaque, max(4.0, cloth.shape[1] * 0.02)):
            cloth[cracks, 3] = 255
            out[index] = (name, Image.fromarray(cloth))
    # A choker moves with the bending neck. The separator often paints a copy of it into the
    # neck and a grey one into the clothing beneath; moving would reveal them as a band.
    names = [name for name, _ in out]
    if "neckwear" not in names or "neck" not in names:
        return out
    first, last = names.index("neck"), len(names) - 1 - names[::-1].index("neckwear")
    choker = np.zeros(out[first][1].size[::-1], np.uint8)
    for role, part in out[first + 1:last + 1]:
        if role == "neckwear":
            choker = np.maximum(choker, np.asarray(part)[..., 3])
    neck = np.array(out[first][1])
    opaque = neck[..., 3] > 240
    if _fill_hidden(neck, (choker == 255) & opaque, (choker < 8) & opaque, max(4.0, neck.shape[1] * 0.02)):
        out[first] = ("neck", Image.fromarray(neck))
    # Only where the choker moves: above the rig's default neck pivot (85 % down the neck).
    rows = np.nonzero((neck[..., 3] > 8).any(axis=1))[0]
    if not len(rows):
        return out
    moving = np.arange(neck.shape[0])[:, None] < rows[0] + 0.85 * (rows[-1] - rows[0])
    for index in range(first + 1, last):
        if names[index] == "topwear":
            cloth = np.array(out[index][1])
            cloth[(choker > 8) & opaque & moving, 3] = 0   # the neck shows through instead
            out[index] = ("topwear", Image.fromarray(cloth))
    return out


def _fill_hidden(rgba, hidden, known, radius):
    """Paint `hidden` pixels with a normalized blur of `known` ones (in place)."""
    if not hidden.any() or known.sum() < 30:
        return False
    values = np.concatenate((rgba[..., :3] * known[..., None], known[..., None]), axis=2)
    blurred = _blur_skin(values, radius)
    weights = blurred[..., 3]
    fill = np.broadcast_to(np.median(rgba[..., :3][known], axis=0), blurred[..., :3].shape).copy()
    np.divide(blurred[..., :3], np.maximum(weights[..., None], 1e-8), out=fill,
              where=(weights > 0.02)[..., None])
    rgba[hidden, :3] = np.clip(np.rint(fill[hidden]), 0, 255).astype(np.uint8)
    return True


def _with_expressions(items, model_dir: Path, size):
    """Insert the painted expression layers (src/core/expressions.py) where the rig expects
    them: closed eyes just above the eye parts (bangs stay on top), the generated mouth state
    next to the drawn one. A generated open mouth makes the drawn mouth the closed one."""
    from src.core import expressions

    made = expressions.read(model_dir)
    names = [n for n, _ in items]

    def art(relative):
        path = model_dir / relative
        if not path.exists():
            return None
        with Image.open(path) as source:
            image = source.convert('RGBA')
        return image if image.size == size and image.getbbox() else None

    def after(*targets):
        hits = [i for i, n in enumerate(names) if n in targets]
        return hits[-1] + 1 if hits else len(names)

    for key in ('eye_close', 'mouth_open', 'mouth_close'):
        image = art(made[key]) if key in made else None
        if image is None:
            continue
        if key == 'eye_close':
            at = after('eyewhite', 'irides', 'eyelash')
        else:
            if key == 'mouth_open' and 'mouth_open' in names:  # the drawn mouth is the closed one
                names[names.index('mouth_open')] = 'mouth_close'
                items = [('mouth_close' if n == 'mouth_open' else n, im) for n, im in items]
            at = after('mouth_open', 'mouth_close')
        items.insert(at, (key, image))
        names.insert(at, key)
    for vowel, path in made.get('visemes', {}).items():  # lip-sync shapes beside the open mouth
        image = art(path) if vowel in expressions.VISEMES else None
        if image is not None:
            at = after('mouth_open', 'mouth_close', *(f'mouth_{v}' for v in expressions.VISEMES))
            items.insert(at, (f'mouth_{vowel}', image))
            names.insert(at, f'mouth_{vowel}')
    # Toggles (hidden until switched on): over the face under the bangs, or on top for a drop
    # drawn over the hair.
    for name, path in made.get('toggles', {}).items():
        image = art(path) if name in expressions.TOGGLES else None
        if image is None:
            continue
        bangs = [i for i, n in enumerate(names) if n == 'front hair']
        at = len(names) if expressions.TOGGLES[name].region in expressions.OVER_HAIR else min(bangs + [after(
            'face', 'nose', 'eyewhite', 'irides', 'eyelash', 'eyebrow', 'eye_close', 'mouth_open', 'mouth_close')])
        label = f'toggle {expressions.TOGGLES[name].label}'  # the preview's button text
        items.insert(at, (label, image))
        names.insert(at, label)
    # Alternate hair/outfits: where the replaced part's top layer is; the rig hides that part
    # while one is on ("toggle Hair: short bob" -> rigger.js VARIANT_PARTS).
    for variant in made.get('variants', {}).values():
        image = art(variant['path'])
        if image is not None:
            at = after(preview_name(variant['after']), variant['after'])
            items.insert(at, (f"toggle {variant['label']}", image))
            names.insert(at, f"toggle {variant['label']}")
    return items


def ensure_preview_psd(model_dir: Path) -> Path:
    """Lazily export old/new models. Atomically cache by source file revision."""
    from psd_tools import PSDImage
    assets_dir = model_dir / 'assets'
    files = sorted(assets_dir.glob('*.png'))
    if not files:
        raise FileNotFoundError('This character has no separated layers to preview')
    revision = [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                for p in files + sorted((model_dir / 'expressions').glob('*'))]
    key = hashlib.sha256(json.dumps([EXPORT_VERSION, revision]).encode()).hexdigest()[:16]
    out_dir = model_dir / 'live_preview'
    output = out_dir / f'preview-{key}.psd'
    with _export_lock:
        if output.exists():
            return output
        layers = preview_layers(assets_dir)
        if not layers:
            raise ValueError('This character has no visible layers to preview')
        psd = PSDImage.new('RGBA', layers[0][1].size, depth=8)
        psd.background_color = None  # preserve transparency in the merged preview too
        for name, image in layers:
            bbox = image.getbbox()
            psd.create_pixel_layer(image.crop(bbox), name=name, left=bbox[0], top=bbox[1])
        # PSD's negative layer count declares the merged image's alpha channel.
        # psd-tools 1.25 creates RGBA headers but does not set this flag for new files.
        info = psd._record.layer_and_mask_information.layer_info
        info.layer_count = -abs(info.layer_count)
        out_dir.mkdir(parents=True, exist_ok=True)
        temporary = out_dir / f'{uuid.uuid4().hex}.tmp'
        try:
            psd.save(temporary)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        for stale in out_dir.glob('preview-*.psd'):  # exports for older layer revisions
            if stale != output:
                try:
                    stale.unlink()
                except OSError:  # still being served on Windows; the next export retries
                    pass
    return output
