"""
A PNGtuber pack: the character in its four talk/blink states as transparent PNGs, the format
veadotube mini, PNGTuber Plus, Discord reactive-image overlays and similar apps animate from a
microphone. Built from the cut-out and the painted expression layers (src/core/expressions.py);
those hold only the pixels that change, so each state is the cut-out with them composited on top.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict

from PIL import Image

from src.core import expressions

# file name: (talking, blinking)
STATES = {"idle": (False, False), "talk": (True, False), "blink": (False, True), "talk_blink": (True, True)}

README = """Ka-myii PNGtuber pack: {name}

idle.png        mouth closed, eyes open
talk.png        mouth open,   eyes open
blink.png       mouth closed, eyes closed
talk_blink.png  mouth open,   eyes closed

veadotube mini: add a state and drop the four images on its slots (idle / talking /
blinking / talking + blinking).
PNGTuber Plus: add idle.png and talk.png as sprites, set talk.png to "show when speaking" and
idle.png to "show when quiet"; blink.png is the blink sprite.
Discord reactive overlays (e.g. Reactive Images): use idle.png for inactive, talk.png for speaking.
OBS image-switcher plugins: the same four files, same size and alignment.
{note}"""


def states(model_dir: Path, padding: int = 16) -> Dict[str, Image.Image]:
    """{state: RGBA image}, all cropped to the same box around the character."""
    model_dir = Path(model_dir)
    cutout = model_dir / "assets" / "character.png"
    if not cutout.exists():
        raise FileNotFoundError("This character has no cut-out; separate its layers first")
    base = Image.open(cutout).convert("RGBA")
    made = expressions.read(model_dir)

    def layer(key):
        path = model_dir / made[key] if made.get(key) else None
        return Image.open(path).convert("RGBA") if path and path.exists() else None

    eyes, mouth_open, mouth_close = layer("eye_close"), layer("mouth_open"), layer("mouth_close")
    if not (mouth_open or mouth_close):
        raise ValueError("Add blink & talk to this character first (Gallery), so it has a second mouth shape")
    out = {}
    for name, (talking, blinking) in STATES.items():
        image = base.copy()
        mouth = mouth_open if talking else mouth_close  # whichever the artwork did not draw
        for extra in (mouth, eyes if blinking else None):
            if extra is not None and extra.size == image.size:
                image.alpha_composite(extra)
        out[name] = image
    box = base.getchannel("A").getbbox() or (0, 0, *base.size)
    box = (max(0, box[0] - padding), max(0, box[1] - padding),
           min(base.width, box[2] + padding), min(base.height, box[3] + padding))
    return {name: image.crop(box) for name, image in out.items()}


def pack(model_dir: Path, name: str) -> io.BytesIO:
    """The zip: four PNGs and a README."""
    images = states(model_dir)
    note = "" if expressions.read(model_dir).get("eye_close") else "\nNo closed eyes were painted: the blink images match the open ones.\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for state, image in images.items():
            png = io.BytesIO()
            image.save(png, "PNG", optimize=True)
            z.writestr(f"{state}.png", png.getvalue())
        z.writestr("README.txt", README.format(name=name, note=note))
    buf.seek(0)
    return buf
