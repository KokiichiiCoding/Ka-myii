"""Separate browser-rig PSD export; never alters the artist's Cubism document."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from pathlib import Path

from PIL import Image

EXPORT_VERSION = 3
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
    return [(name, image) for name, image in merged.items() if image.getbbox()]


def ensure_preview_psd(model_dir: Path) -> Path:
    """Lazily export old/new models. Atomically cache by source file revision."""
    from psd_tools import PSDImage
    assets_dir = model_dir / 'assets'
    files = sorted(assets_dir.glob('*.png'))
    if not files:
        raise FileNotFoundError('This character has no separated layers to preview')
    revision = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files]
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
