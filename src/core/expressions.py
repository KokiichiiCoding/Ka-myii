"""
Character-specific expression art: closed eyes and the other mouth state, painted by SDXL, and
optional toggles (blush, tears, heart eyes, > < ...) the preview switches on and off.

The preview rig blinks by fading the open-eye parts out and an ``eye_close`` layer in, and
talks by fading ``mouth_close`` out and a vertically stretched ``mouth_open`` in. See-through
gives one drawn mouth and no closed eyes, so without this the rig used generic clip-art. Here
both are inpainted in the character's own style and saved as extra layers in
``<model>/expressions/`` (outside ``assets/``, which rebuild_layers.py replaces).

Recipe (tuned on a real result, see AGENTS.md): eyes are a geometric ellipse per eye around
everything eye-like, minus bangs drawn on top, flooded with the surrounding skin colour and
repainted at full strength as "closed eyes" — a looser mask left iris strokes the model then
drew around. The mouth gets room to open; the tagger says which state the drawing already is.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, NamedTuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)

FOLDER = "expressions"
EYE_PARTS = ("eyewhite", "irides", "eyelash")
NEGATIVE_EYES = "open eyes, iris, pupils, eyeball, heart, sparkle, tears, glasses"
# Seen: with the whole character description, the model drew a black mask over the chin.
NEGATIVE_OPEN = "closed mouth, tongue out, fang, mask, choker, beard, facial hair, black"
NEGATIVE_CLOSED = "open mouth, teeth, tongue, mask, beard, facial hair"


class Toggle(NamedTuple):
    label: str
    region: str                    # where it goes (toggle_mask)
    prompt: Optional[str] = None   # None: a manga symbol drawn as vector art (draw_symbol)
    negative: str = ""
    strength: float = 1.0
    flood: bool = False            # cover the eyes with skin first, as for closed eyes


# Optional toggles: painted overlays the preview switches on and off; hidden layers in the PSD.
TOGGLES = {
    # Many characters are drawn with a light blush already; the toggle must still show.
    "blush": Toggle("Blush", "cheeks", "heavy blush, nose blush", "tears, sweat, open mouth, hair", 0.7),
    "tears": Toggle("Tears", "tears", "tears, streaming tears, crying", "open mouth, blood, hair", 0.75),
    # Painted, these small patches over hair became a tiny person (anger vein) or nothing (sweat).
    "sweat": Toggle("Sweat drop", "temple"),
    "anger": Toggle("Anger vein", "forehead"),
    "hearts": Toggle("Heart eyes", "eyes", "heart-shaped pupils, open eyes", "closed eyes, glasses", 0.7),
    "stars": Toggle("Star eyes", "eyes", "star-shaped pupils, sparkling eyes, open eyes", "closed eyes, glasses", 0.7),
    "squeeze": Toggle("> < eyes", "eyes", ">_<, squeezing eyes, closed eyes", "open eyes, iris, pupils, tears", 1.0, flood=True),
    "spiral": Toggle("Spiral eyes", "eyes", "@_@, spiral eyes, dizzy", "pupils, closed eyes, glasses", 1.0, flood=True),
    "gloom": Toggle("Gloom", "gloom"),
    "sparkles": Toggle("Sparkle aura", "aura"),
    # Manga symbols beside the head, drawn locally (no diffusion); the rig floats them with Idle.
    "question": Toggle("Question mark", "above"),
    "exclaim": Toggle("Exclamation", "above"),
    "zzz": Toggle("Zzz", "above"),
    "notes": Toggle("Music notes", "notes"),
    "hearts_float": Toggle("Floating hearts", "hearts"),
    # Mouth emotes replace the animated mouth while on (rigger.js MOUTH_EMOTES).
    "tongue": Toggle("Tongue out", "mouth", ":p, tongue out", "teeth, fang", 1.0),
    # Painted, ":3" erased the mouth instead (witch, GPU); an ω line over skin reads every time.
    "catmouth": Toggle("Cat mouth", "mouth"),
    "fang": Toggle("Fang", "mouth", "fang, skin fang, smile, open mouth", "tongue out, closed mouth", 1.0),
}
# Painted on top of the hair (the rest go under the front hair, which sways over them).
OVER_HAIR = ("temple", "forehead", "aura", "above", "notes", "hearts")
# Japanese vowel mouths for lip sync (Live2D's AIUEO approach: distinct shapes, not one open
# mouth scaled). Danbooru-style tags the checkpoint knows; teeth only where a vowel shows them.
VISEMES = {
    "a": ("open mouth, :d, wide open mouth", "closed mouth, tongue out, fang, teeth, mask, beard, black"),
    "i": ("grin, clenched teeth, wide mouth, teeth", "open mouth, tongue, fang, mask, beard, black"),
    "u": ("pout, small mouth, :o", "wide mouth, teeth, tongue, smile, mask, beard, black"),
    "e": ("open mouth, wide mouth, upper teeth", "closed mouth, tongue out, fang, mask, beard, black"),
    "o": (":o, open mouth, round mouth", "wide mouth, teeth, smile, tongue, mask, beard, black"),
}


def load_layers(assets_dir: Path) -> Dict[str, np.ndarray]:
    """Alpha (uint8) of each separated layer by name ("02_front hair.png" -> "front hair"), back to
    front. Only the numbered layers: character.png is the whole cut-out, not a part."""
    out = {}
    for path in sorted(Path(assets_dir).glob("[0-9]*_*.png")):
        name = path.stem.split("_", 1)[1]
        with Image.open(path) as im:
            out[name] = np.asarray(im.convert("RGBA"))[..., 3]
    return out


def _grow(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(px * 2 + 1))) > 0


def _zeros(layers):
    return np.zeros(next(iter(layers.values())).shape, bool)


def eye_boxes(layers: Dict[str, np.ndarray]) -> list:
    """(x0, y0, x1, y1) around each eye's white, iris and lashes."""
    boxes = []
    for side in "lr" if layers else "":
        m = _zeros(layers)
        for part in EYE_PARTS:
            if f"{part}-{side}" in layers:
                m |= layers[f"{part}-{side}"] > 16
        ys, xs = np.nonzero(m)
        if len(xs) >= 20:
            boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    return boxes


def eye_mask(layers: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """An ellipse per eye around its white, iris and lashes, minus bangs drawn on top."""
    boxes = eye_boxes(layers)
    if not boxes:
        return None
    shape = next(iter(layers.values())).shape
    canvas = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    for x0, y0, x1, y1 in boxes:
        px, py = (x1 - x0) * 0.12 + 4, (y1 - y0) * 0.18 + 4
        draw.ellipse((x0 - px, y0 - py, x1 + px, y1 + py), fill=255)
    mask = np.asarray(canvas) > 0
    if "front hair" in layers:
        mask &= ~(layers["front hair"] > 128)
    return mask


def mouth_mask(layers: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Room for the mouth to open: an ellipse around it, kept inside the face."""
    if "mouth" not in layers:
        return None
    ys, xs = np.nonzero(layers["mouth"] > 16)
    if len(xs) < 10:
        return None
    width = xs.max() - xs.min()
    shape = layers["mouth"].shape
    canvas = Image.new("L", (shape[1], shape[0]), 0)
    # Room around the lips and below them: the probe that opened convincingly used ~16 px.
    ImageDraw.Draw(canvas).ellipse((xs.min() - 16, ys.min() - 12, xs.max() + 16,
                                    ys.max() + max(20, width * 0.45)), fill=255)
    mask = np.asarray(canvas) > 0
    if "face" in layers:
        mask &= _grow(layers["face"] > 128, 2)
    if "front hair" in layers:
        mask &= ~(layers["front hair"] > 128)
    return mask


def toggle_mask(layers: Dict[str, np.ndarray], region: str) -> Optional[np.ndarray]:
    """Where a toggle is painted, placed from the eyes and the face (image left/right)."""
    if region == "eyes":
        return eye_mask(layers)
    if region == "mouth":
        return mouth_mask(layers)
    boxes = eye_boxes(layers)
    if not boxes or "face" not in layers:
        return None
    face = layers["face"] > 128
    xs = np.nonzero(face)[1]
    fx0, fx1, fw = xs.min(), xs.max(), xs.max() - xs.min()
    top = min(b[1] for b in boxes)
    canvas = Image.new("L", (face.shape[1], face.shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    if region in ("aura", "notes", "hearts"):
        draw.rectangle((fx0-.55*fw,top-.85*fw,fx1+.55*fw,top+.9*fw),fill=255)
        return np.asarray(canvas)>0
    if region == "above":  # over the head's image-right side; below the canvas top when cropped
        h, w = .6 * fw, .45 * fw
        y0 = max(2, top - .95 * fw)
        draw.rectangle((fx1 - .15 * fw, y0, fx1 - .15 * fw + w, y0 + h), fill=255)
        return np.asarray(canvas) > 0
    if region == "gloom":
        draw.ellipse((fx0,top-.5*fw,fx1,top+.55*fw),fill=255)
        return (np.asarray(canvas)>0) & face
    if region in ("cheeks", "tears"):
        for x0, y0, x1, y1 in boxes:
            ew, cx = x1 - x0, (x0 + x1) / 2
            draw.ellipse((cx - 0.6 * ew, y1 + 0.05 * ew, cx + 0.6 * ew, y1 + 0.7 * ew) if region == "cheeks"
                         else (cx - 0.5 * ew, (y0 + y1) / 2, cx + 0.5 * ew, y1 + 1.4 * ew), fill=255)
    elif region == "temple":
        cx, cy = fx1 - 0.1 * fw, top - 0.15 * fw
        draw.ellipse((cx - 0.11 * fw, cy - 0.17 * fw, cx + 0.11 * fw, cy + 0.17 * fw), fill=255)
    elif region == "forehead":
        cx, cy = fx0 + 0.25 * fw, top - 0.3 * fw
        draw.ellipse((cx - 0.2 * fw, cy - 0.2 * fw, cx + 0.2 * fw, cy + 0.2 * fw), fill=255)
    else:
        raise ValueError(f"unknown toggle region {region!r}")
    mask = np.asarray(canvas) > 0
    if region not in OVER_HAIR:  # on the skin only: off the face outline, eyes, nose, mouth and bangs
        mask &= ~_grow(~face, 4)
        for name, alpha in layers.items():
            if name.split("-")[0] in EYE_PARTS + ("eyebrow", "nose", "mouth"):
                mask &= ~_grow(alpha > 16, 2)
        if "front hair" in layers:
            mask &= ~(layers["front hair"] > 128)
    return mask if mask.sum() > 30 else None


def skin_colour(base: Image.Image, region: np.ndarray, layers: Dict[str, np.ndarray]):
    """Median colour of the face just around ``region`` (no hair, brows or eyes)."""
    ring = _grow(region, 14) & ~region
    if "face" in layers:
        ring &= layers["face"] > 128
    for name in ("front hair", "eyebrow-l", "eyebrow-r"):
        if name in layers:
            ring &= ~(layers[name] > 128)
    pixels = np.asarray(base.convert("RGB"))[ring]
    return tuple(int(v) for v in np.median(pixels, axis=0)) if len(pixels) > 20 else None


def _soft(mask: np.ndarray, feather: int = 4) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(feather))


def _changed(painted: Image.Image, base: Image.Image, mask: np.ndarray, grow: int = 5) -> np.ndarray:
    """The part of ``mask`` the painting really changed (plus a margin): a mouth layer that is
    just the mouth, not a plate of skin, so the rig's mouth anchor and stretch fit it."""
    diff = np.abs(np.asarray(painted.convert("RGB"), np.int16) - np.asarray(base.convert("RGB"), np.int16)).max(-1)
    changed = _grow(mask & (diff > 32), grow) & mask
    return changed if changed.sum() > 30 else mask


def draw_symbol(region: str, mask: np.ndarray, kind: Optional[str] = None) -> Image.Image:
    """A crisp manga symbol in ``mask``'s box (full-canvas RGBA): a sweat drop at the temple, an
    anger vein on the forehead, gloom, glints, notes, hearts, steam, or ``kind`` (question /
    exclaim / zzz) above the head. Drawn 4x larger and scaled down for clean edges."""
    import math

    ys, xs = np.nonzero(mask)
    x0, y0, k = xs.min(), ys.min(), 4
    w, h = (xs.max() + 1 - x0) * k, (ys.max() + 1 - y0) * k
    art = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(art)
    line = max(2 * k, min(w, h) // 14)
    if region == "temple":
        # Teardrop: a round bottom and two tangents up to the tip.
        r = min(w * 0.4, h * 0.3)
        cx, cy, tip = w / 2, h - r - line, line
        b = math.acos(min(1.0, r / (cy - tip)))
        points = [(cx, tip)] + [(cx + r * math.cos(t), cy + r * math.sin(t))
                                for t in np.linspace(b - math.pi / 2, 1.5 * math.pi - b, 48)]
        draw.polygon(points, fill=(195, 228, 255, 255))
        draw.line(points + [points[0]], fill=(60, 105, 170, 255), width=line, joint="curve")
        draw.ellipse((cx - r * 0.55, cy - r * 0.5, cx - r * 0.2, cy + r * 0.05), fill=(255, 255, 255, 255))
    elif region == "forehead":
        # Four arcs around the corners, bulging towards a gap in the middle.
        c, d, r, line = w / 2, min(w, h) * 0.22, min(w, h) * 0.21, line * 6 // 5
        bands = [(-1, -1, 0), (1, -1, 90), (1, 1, 180), (-1, 1, 270)]
        for colour, grow, spread in (((95, 20, 35, 255), line // 4, 4), ((228, 48, 72, 255), 0, 0)):
            for sx, sy, start in bands:
                ox, oy = c + sx * d, h / 2 + sy * d
                draw.arc((ox - r - grow, oy - r - grow, ox + r + grow, oy + r + grow), start + 14 - spread,
                         start + 76 + spread, fill=colour, width=line + 2 * grow)
    elif region == "gloom":
        for y in range(h):
            fade=max(0,1-y/max(1,h-1))**1.1
            draw.line((0,y,w,y),fill=(55,67,122,round(125*fade)),width=1)
        for x in (.22,.36,.5,.64,.78):
            draw.line((x*w,.14*h,x*w,.55*h),fill=(63,68,123,95),width=max(k,w//100))
    elif region in ("above", "notes", "hearts"):
        _draw_manga(draw, kind if region == "above" and kind else region, w, h, k)
    elif region == "aura":
        # Original four-point glints arranged beside the head, clear of the eye centres.
        for x,y,r,col in [(.12,.38,.065,(255,221,136,245)),(.87,.27,.075,(255,234,166,245)),
                          (.09,.72,.035,(174,226,239,235)),(.88,.69,.045,(174,226,239,235)),
                          (.23,.09,.035,(255,242,201,230)),(.76,.91,.028,(255,236,172,220))]:
            cx,cy=x*w,y*h;radius=r*min(w,h)
            pts=[(cx,cy-radius),(cx+radius*.25,cy-radius*.25),(cx+radius*.7,cy),
                 (cx+radius*.25,cy+radius*.25),(cx,cy+radius),(cx-radius*.25,cy+radius*.25),
                 (cx-radius*.7,cy),(cx-radius*.25,cy-radius*.25)]
            draw.polygon(pts,fill=col)
            draw.line(pts+[pts[0]],fill=(161,132,86,160),width=max(1,k//2),joint="curve")
    else:
        raise ValueError(f"no symbol for {region!r}")
    layer = Image.new("RGBA", (mask.shape[1], mask.shape[0]), (0, 0, 0, 0))
    layer.paste(art.resize((w // k, h // k), Image.Resampling.LANCZOS), (int(x0), int(y0)))
    if region == "gloom":
        from PIL import ImageChops
        layer.putalpha(ImageChops.multiply(layer.getchannel("A"),Image.fromarray(mask.astype(np.uint8)*255)))
    return layer


# Positions around the head (share of the aura box), clear of the face like the sparkles.
AROUND = [(.11, .36, .09), (.88, .25, .1), (.1, .72, .06), (.9, .66, .07), (.24, .08, .06)]


def _draw_manga(draw, kind: str, w: int, h: int, k: int):
    """Question mark, exclamation, Zzz, notes or hearts in a ``w`` x ``h`` box,
    each a bold colour with a dark outline so it reads over any artwork."""
    import math

    line = max(3 * k, min(w, h) // 9)

    def stroke(points, colour, width=line, arc=None):
        for fill, extra in (((40, 30, 50, 255), width // 2), (colour, 0)):
            if arc:
                draw.arc(arc[0], arc[1], arc[2], fill=fill, width=width + 2 * extra)
            else:
                draw.line(points, fill=fill, width=width + 2 * extra, joint="curve")

    def dot(cx, cy, r, colour):
        draw.ellipse((cx - r - line // 2, cy - r - line // 2, cx + r + line // 2, cy + r + line // 2), fill=(40, 30, 50, 255))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=colour)

    if kind == "question":
        blue = (95, 155, 255, 255)
        box = (w * .18, h * .06, w * .82, h * .5)
        stroke(None, blue, arc=(box, 180, 405))
        ex, ey = w * .5 + w * .32 * math.cos(math.radians(45)), h * .28 + h * .22 * math.sin(math.radians(45))
        stroke([(ex, ey), (w * .5, h * .58), (w * .5, h * .68)], blue)
        dot(w * .5, h * .86, line * .7, blue)
    elif kind == "exclaim":
        red = (255, 85, 85, 255)
        draw.polygon([(w * .26, h * .04), (w * .74, h * .04), (w * .6, h * .7), (w * .4, h * .7)], fill=(40, 30, 50, 255))
        draw.polygon([(w * .31, h * .07), (w * .69, h * .07), (w * .56, h * .67), (w * .44, h * .67)], fill=red)
        dot(w * .5, h * .86, line, red)
    elif kind == "zzz":
        violet = (130, 140, 255, 255)
        for i, s in enumerate((.55, .4, .3)):
            x0, y0 = w * (.02 + .3 * i), h * (.62 - .3 * i)
            zw, zh = w * s, h * s * .62
            stroke([(x0, y0), (x0 + zw, y0), (x0, y0 + zh), (x0 + zw, y0 + zh)], violet, max(2 * k, int(line * .8 * (1 - .2 * i))))
    elif kind == "notes":
        # Eighth notes: an oval head, a stem and a flag; outlines scale with each note.
        for (x, y, r), colour in zip(AROUND, [(255, 120, 180, 255), (110, 185, 255, 255), (255, 200, 90, 255), (150, 230, 160, 255)]):
            cx, cy, s = x * w, y * h, r * min(w, h) * .8
            lw, rim = max(2 * k, int(s * .16)), max(k, int(s * .07))
            stem = (cx + s * .5, cy - s * 1.7)
            for fill, extra in (((40, 30, 50, 255), rim), (colour, 0)):
                draw.line([(cx + s * .5, cy), stem], fill=fill, width=lw + 2 * extra)
                draw.line([stem, (stem[0] + s * .6, stem[1] + s * .6)], fill=fill, width=lw + 2 * extra, joint="curve")
                draw.ellipse((cx - s * .6 - extra, cy - s * .42 - extra, cx + s * .6 + extra, cy + s * .42 + extra), fill=fill)
    elif kind == "hearts":
        pink = (255, 95, 145, 255)
        for x, y, r in AROUND:
            cx, cy, s = x * w, y * h, r * min(w, h) / 16
            pts = [(cx + s * 16 * math.sin(t) ** 3, cy - s * (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)))
                   for t in np.linspace(0, 2 * math.pi, 60)]
            draw.polygon(pts, fill=pink, outline=(120, 30, 60, 255), width=max(2, k))


def _cat_mouth(base: Image.Image, mask: np.ndarray, layers: Dict[str, np.ndarray]) -> Image.Image:
    """An ω cat mouth in the drawn mouth's own line colour, over skin that hides the old mouth."""
    rgb = np.asarray(base.convert("RGB"))
    mouth = layers["mouth"] > 128
    ink = rgb[mouth]
    ink = tuple(int(v) for v in ink[np.argsort(ink.sum(-1))[:max(1, len(ink) // 5)]].mean(0)) if len(ink) else (90, 40, 50)
    skin = skin_colour(base, mask, layers) or tuple(int(v) for v in rgb[mask].mean(0))
    ys, xs = np.nonzero(mouth if mouth.any() else mask)
    cx, cy, half = xs.mean(), ys.mean(), max(4, (xs.max() - xs.min()) * .42)
    k = 4
    art = Image.new("RGBA", (base.width * k, base.height * k), (0, 0, 0, 0))
    draw = ImageDraw.Draw(art)
    w = max(2 * k, int(half * k * .16))
    for sx in (-1, 1):  # two arcs meeting in the middle: ω
        x0 = (cx + (sx - 1) * half / 2) * k
        draw.arc((x0, (cy - half * .45) * k, x0 + half * k, (cy + half * .45) * k), 20, 160, fill=ink + (255,), width=w)
    layer = Image.new("RGBA", base.size, skin + (0,))
    layer.putalpha(_soft(mask & _grow(mouth, 3) if mouth.any() else mask, 2))
    layer.alpha_composite(art.resize(base.size, Image.Resampling.LANCZOS))
    return layer


def _save_layer(painted: Image.Image, mask: np.ndarray, path: Path):
    layer = painted.convert("RGBA")
    layer.putalpha(_soft(mask))
    path.parent.mkdir(parents=True, exist_ok=True)
    layer.save(path)


def generate(generator, model_dir: Path, seed: Optional[int] = None,
             task_id: Optional[str] = None, tag_image=None) -> Dict:
    """Paint the expression layers for a separated character; returns what was made
    (also written to ``expressions/expressions.json``)."""
    from src.core import tagger

    model_dir = Path(model_dir)
    base = Image.open(model_dir / "base_image.png").convert("RGB")
    layers = load_layers(model_dir / "assets")
    out_dir = model_dir / FOLDER
    made: Dict = {k: v for k, v in read(model_dir).items() if k in ("toggles", "visemes", "variants")}  # painted separately

    tags = (tag_image or tagger.tag_image)(base)
    scores = (tags or {}).get("scores", {})
    mouth_open_now = scores.get("open mouth", 0) > max(0.5, scores.get("closed mouth", 0))
    smile = scores.get("smile", 0) > 0.5
    described = tagger.describe(tags) if tags else ""
    made["tags"] = described
    # Face-only prompts: the character's clothes/accessory tags invite them into a face patch.
    seed = 7 if seed is None else int(seed)

    eyes = eye_mask(layers)
    if eyes is not None and eyes.sum() > 50:
        skin = skin_colour(base, eyes, layers)
        start = base.copy()
        if skin:
            start.paste(skin, (0, 0, *base.size), Image.fromarray(eyes.astype(np.uint8) * 255))
        painted = generator.inpaint(start, Image.fromarray(eyes.astype(np.uint8) * 255),
                                    "closed eyes", negative=NEGATIVE_EYES, strength=1.0,
                                    seed=seed, task_id=task_id, label="Painting closed eyes")
        _save_layer(painted, eyes, out_dir / "eye_close.png")
        made["eye_close"] = f"{FOLDER}/eye_close.png"

    mouth = mouth_mask(layers)
    if mouth is not None and mouth.sum() > 30:
        if mouth_open_now:
            name, wanted, negative = "mouth_close", "closed mouth", NEGATIVE_CLOSED
        else:
            name, negative = "mouth_open", NEGATIVE_OPEN
            wanted = "open mouth, :d, teeth, smile" if smile else "open mouth, :o, teeth"
        painted = generator.inpaint(base, Image.fromarray(mouth.astype(np.uint8) * 255), wanted,
                                    negative=negative, strength=1.0, seed=seed, task_id=task_id,
                                    label="Painting the " + ("closed" if mouth_open_now else "open") + " mouth")
        _save_layer(painted, _changed(painted, base, mouth), out_dir / f"{name}.png")
        made[name] = f"{FOLDER}/{name}.png"
    made["mouth_drawn"] = "open" if mouth_open_now else "closed"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "expressions.json").write_text(json.dumps(made, indent=2), encoding="utf-8")
    logger.info("Expressions for %s: %s", model_dir.name, ", ".join(k for k in made if k in ("eye_close", "mouth_open", "mouth_close")))
    return made


def paint_toggles(generator, model_dir: Path, names, seed: Optional[int] = None,
                  task_id: Optional[str] = None) -> Dict:
    """Paint the chosen TOGGLES onto a separated character (added to expressions.json)."""
    model_dir = Path(model_dir)
    base = Image.open(model_dir / "base_image.png").convert("RGB")
    layers = load_layers(model_dir / "assets")
    made = read(model_dir)
    toggles = made.setdefault("toggles", {})
    seed = 7 if seed is None else int(seed)
    for name in names:
        spec = TOGGLES[name]
        mask = toggle_mask(layers, spec.region)
        if mask is None:
            logger.warning("No place to paint %s on %s (no eyes/face layer)", name, model_dir.name)
            continue
        path = model_dir / FOLDER / f"toggle_{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if spec.prompt is None and spec.region == "mouth":
            _cat_mouth(base, mask, layers).save(path)
        elif spec.prompt is None:
            draw_symbol(spec.region, mask, name).save(path)
        else:
            start = base.copy()
            skin = skin_colour(base, mask, layers) if spec.flood else None
            if skin:
                start.paste(skin, (0, 0, *base.size), Image.fromarray(mask.astype(np.uint8) * 255))
            painted = generator.inpaint(start, Image.fromarray(mask.astype(np.uint8) * 255), spec.prompt,
                                        negative=spec.negative, strength=spec.strength, seed=seed,
                                        task_id=task_id, label=f"Painting {spec.label}")
            _save_layer(painted, mask, path)
        toggles[name] = f"{FOLDER}/toggle_{name}.png"
    (model_dir / FOLDER).mkdir(parents=True, exist_ok=True)
    (model_dir / FOLDER / "expressions.json").write_text(json.dumps(made, indent=2), encoding="utf-8")
    return made


def paint_visemes(generator, model_dir: Path, seed: Optional[int] = None, task_id: Optional[str] = None) -> Dict:
    """Paint the five vowel mouths (VISEMES) where the mouth is (added to expressions.json)."""
    model_dir = Path(model_dir)
    base = Image.open(model_dir / "base_image.png").convert("RGB")
    mouth = mouth_mask(load_layers(model_dir / "assets"))
    made = read(model_dir)
    if mouth is None or mouth.sum() <= 30:
        logger.warning("No mouth layer to paint vowel mouths on %s", model_dir.name)
        return made
    seed = 7 if seed is None else int(seed)
    visemes = made.setdefault("visemes", {})
    for vowel, (wanted, negative) in VISEMES.items():
        painted = generator.inpaint(base, Image.fromarray(mouth.astype(np.uint8) * 255), wanted,
                                    negative=negative, strength=1.0, seed=seed, task_id=task_id,
                                    label=f"Painting the {vowel.upper()} mouth")
        _save_layer(painted, _changed(painted, base, mouth), model_dir / FOLDER / f"mouth_{vowel}.png")
        visemes[vowel] = f"{FOLDER}/mouth_{vowel}.png"
    (model_dir / FOLDER / "expressions.json").write_text(json.dumps(made, indent=2), encoding="utf-8")
    return made


def read(model_dir: Path) -> Dict:
    path = Path(model_dir) / FOLDER / "expressions.json"
    try:
        # utf-8-sig: a BOM from a Windows editor must not read as "nothing painted" (a later
        # job would then overwrite the record).
        return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def as_assets(model_dir: Path, made: Dict, assets: list) -> list:
    """Assets for the painted layers, ordered just above the parts they replace."""
    from src.models.vtuber_model import Asset

    def order_of(*names):
        orders = [a.metadata.get("order", 0) for a in assets if a.layer_type.split("-")[0] in names]
        return max(orders) + 0.5 if orders else 0.5

    out = []
    for key, anchor in (("eye_close", ("eyewhite", "irides", "eyelash")), ("mouth_open", ("mouth",)),
                        ("mouth_close", ("mouth",))):
        if key in made and (Path(model_dir) / made[key]).exists():
            out.append(Asset(layer_type=key, file_path=Path(model_dir) / made[key],
                             metadata={"order": order_of(*anchor), "expression": True}))
    for vowel, path in made.get("visemes", {}).items():
        if vowel in VISEMES and (Path(model_dir) / path).exists():
            out.append(Asset(layer_type=f"mouth_{vowel}", file_path=Path(model_dir) / path,
                             metadata={"order": order_of("mouth") + 0.1, "expression": True}))
    for vid, variant in made.get("variants", {}).items():  # alternate hair/outfits (src/core/variants.py)
        if (Path(model_dir) / variant["path"]).exists():
            out.append(Asset(layer_type=f"variant {vid}", file_path=Path(model_dir) / variant["path"],
                             metadata={"order": order_of(variant["after"].split("-")[0]) + 0.3, "expression": True}))
    top = max((a.metadata.get("order", 0) for a in assets), default=0)
    for name, path in made.get("toggles", {}).items():
        if name in TOGGLES and (Path(model_dir) / path).exists():
            over = TOGGLES[name].region in OVER_HAIR
            out.append(Asset(layer_type=f"toggle {name}", file_path=Path(model_dir) / path, metadata={
                "order": top + 1 if over else order_of("face", "nose", "mouth", *EYE_PARTS) + 0.25,
                "expression": True}))
    return out
