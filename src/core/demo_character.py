"""
Procedural demo character renderer.

Draws a clean, flat-shaded, front-facing anime character in the same "reference
sheet" style that rigs well in Live2D. It is used when Ka-myii runs without
model weights (demo mode) so the *entire* downstream pipeline — decomposition,
PSD export and Live2D packaging — can be exercised and demonstrated end-to-end.

Crucially the renderer also produces a **region map**: a flat image where every
pixel is painted with a unique per-region id colour. The asset separator reads
this sidecar (when present) to perform a pixel-perfect decomposition, giving a
high-quality demo without any segmentation model.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Tuple

from PIL import Image, ImageDraw, ImageFilter

RGB = Tuple[int, int, int]

# Unique id colour per region (used only in the region map).
REGION_COLORS: Dict[str, RGB] = {
    "background": (0, 0, 0),
    "hair_back": (10, 10, 10),
    "body": (20, 20, 20),
    "clothing": (30, 30, 30),
    "head": (40, 40, 40),
    "ear_L": (50, 50, 50),
    "ear_R": (60, 60, 60),
    "face": (70, 70, 70),
    "brow_L": (80, 80, 80),
    "brow_R": (90, 90, 90),
    "eye_L": (100, 100, 100),
    "eye_R": (110, 110, 110),
    "nose": (120, 120, 120),
    "mouth": (130, 130, 130),
    "hair_front": (140, 140, 140),
    "accessories": (150, 150, 150),
}

# Named colour palette for parsing hair / eye / outfit colours from the prompt.
_COLOR_WORDS: Dict[str, RGB] = {
    "blue": (90, 140, 220),
    "cyan": (120, 200, 220),
    "teal": (70, 170, 170),
    "pink": (240, 150, 190),
    "magenta": (210, 90, 170),
    "purple": (160, 120, 210),
    "violet": (170, 130, 220),
    "red": (210, 90, 90),
    "crimson": (190, 60, 70),
    "orange": (240, 160, 90),
    "yellow": (240, 215, 110),
    "blonde": (240, 220, 150),
    "blond": (240, 220, 150),
    "green": (110, 190, 130),
    "mint": (150, 220, 180),
    "brown": (160, 110, 80),
    "black": (60, 60, 70),
    "white": (235, 235, 240),
    "silver": (200, 205, 215),
    "grey": (170, 170, 180),
    "gray": (170, 170, 180),
}

_SKIN = (255, 226, 200)
_SKIN_SHADOW = (245, 200, 180)
_OUTLINE = (60, 48, 60)
_BLUSH = (255, 170, 170)


def _seeded_rng(prompt: str, seed) -> int:
    base = f"{prompt}|{seed}"
    return int(hashlib.sha256(base.encode()).hexdigest(), 16)


def _pick_color(prompt: str, keyword_group, default: RGB, salt: int) -> RGB:
    """Pick a colour: honour an explicit colour word in the prompt, else vary by salt."""
    low = prompt.lower()
    for word, rgb in _COLOR_WORDS.items():
        if word in low:
            # Only accept the colour if it's near the relevant keyword group,
            # otherwise fall through to a deterministic choice.
            for kw in keyword_group:
                if kw in low:
                    idx = low.find(kw)
                    cidx = low.find(word)
                    if abs(idx - cidx) < 24:
                        return rgb
    palette = list(_COLOR_WORDS.values())
    return palette[salt % len(palette)]


def _darker(rgb: RGB, amount: float = 0.82) -> RGB:
    return tuple(max(0, int(c * amount)) for c in rgb)


def _lighter(rgb: RGB, amount: float = 1.12) -> RGB:
    return tuple(min(255, int(c * amount)) for c in rgb)


def _compute_geometry(w: int, h: int, rng: int) -> dict:
    """Compute all region coordinates once so the pretty render and the region
    map stay perfectly aligned."""
    cx = w / 2
    head_r = w * 0.20
    head_cy = h * 0.30
    jitter = ((rng % 7) - 3) * 0.004 * w  # tiny deterministic variety

    return {
        "w": w,
        "h": h,
        "cx": cx,
        "head_r": head_r,
        "head_cy": head_cy,
        "head_box": (cx - head_r, head_cy - head_r * 1.05, cx + head_r, head_cy + head_r * 1.1),
        "eye_dx": head_r * 0.45,
        "eye_cy": head_cy + head_r * 0.12,
        "eye_rx": head_r * 0.24,
        "eye_ry": head_r * 0.30,
        "brow_cy": head_cy - head_r * 0.30,
        "mouth_cy": head_cy + head_r * 0.62,
        "nose_cy": head_cy + head_r * 0.38,
        "ear_cy": head_cy + head_r * 0.05,
        "shoulder_y": head_cy + head_r * 1.25,
        "hip_y": h * 0.98,
        "shoulder_half": w * 0.20,
        "hip_half": w * 0.34,
        "jitter": jitter,
    }


def _draw_regions(draw: ImageDraw.ImageDraw, g: dict):
    """Flat silhouettes painted with per-region id colours (the region map)."""
    cx, w, h = g["cx"], g["w"], g["h"]

    def col(name):
        return REGION_COLORS[name]

    # hair back — broad shape behind everything
    draw.ellipse(
        (cx - g["head_r"] * 1.5, g["head_cy"] - g["head_r"] * 1.5,
         cx + g["head_r"] * 1.5, g["head_cy"] + g["head_r"] * 3.2),
        fill=col("hair_back"),
    )
    # body + clothing (dress trapezoid)
    draw.polygon(
        [
            (cx - g["shoulder_half"], g["shoulder_y"]),
            (cx + g["shoulder_half"], g["shoulder_y"]),
            (cx + g["hip_half"], g["hip_y"]),
            (cx - g["hip_half"], g["hip_y"]),
        ],
        fill=col("clothing"),
    )
    # neck/body skin patch
    draw.rectangle(
        (cx - g["head_r"] * 0.28, g["head_cy"] + g["head_r"] * 0.7,
         cx + g["head_r"] * 0.28, g["shoulder_y"] + 4),
        fill=col("body"),
    )
    # ears
    draw.ellipse((cx - g["head_r"] * 1.08, g["ear_cy"] - g["head_r"] * 0.18,
                  cx - g["head_r"] * 0.82, g["ear_cy"] + g["head_r"] * 0.18), fill=col("ear_L"))
    draw.ellipse((cx + g["head_r"] * 0.82, g["ear_cy"] - g["head_r"] * 0.18,
                  cx + g["head_r"] * 1.08, g["ear_cy"] + g["head_r"] * 0.18), fill=col("ear_R"))
    # head / face
    draw.ellipse(g["head_box"], fill=col("head"))
    draw.ellipse((g["head_box"][0] + g["head_r"] * 0.15, g["head_box"][1] + g["head_r"] * 0.5,
                  g["head_box"][2] - g["head_r"] * 0.15, g["head_box"][3]), fill=col("face"))
    # brows
    draw.ellipse((cx - g["eye_dx"] - g["eye_rx"], g["brow_cy"] - 6,
                  cx - g["eye_dx"] + g["eye_rx"], g["brow_cy"] + 6), fill=col("brow_L"))
    draw.ellipse((cx + g["eye_dx"] - g["eye_rx"], g["brow_cy"] - 6,
                  cx + g["eye_dx"] + g["eye_rx"], g["brow_cy"] + 6), fill=col("brow_R"))
    # eyes
    draw.ellipse((cx - g["eye_dx"] - g["eye_rx"], g["eye_cy"] - g["eye_ry"],
                  cx - g["eye_dx"] + g["eye_rx"], g["eye_cy"] + g["eye_ry"]), fill=col("eye_L"))
    draw.ellipse((cx + g["eye_dx"] - g["eye_rx"], g["eye_cy"] - g["eye_ry"],
                  cx + g["eye_dx"] + g["eye_rx"], g["eye_cy"] + g["eye_ry"]), fill=col("eye_R"))
    # nose + mouth
    draw.ellipse((cx - 4, g["nose_cy"] - 4, cx + 4, g["nose_cy"] + 4), fill=col("nose"))
    draw.ellipse((cx - g["head_r"] * 0.18, g["mouth_cy"] - 6,
                  cx + g["head_r"] * 0.18, g["mouth_cy"] + 10), fill=col("mouth"))
    # hair front (fringe)
    draw.polygon(
        [
            (cx - g["head_r"] * 1.02, g["head_cy"] - g["head_r"] * 0.2),
            (cx - g["head_r"] * 0.9, g["head_cy"] - g["head_r"] * 1.1),
            (cx + g["head_r"] * 0.9, g["head_cy"] - g["head_r"] * 1.1),
            (cx + g["head_r"] * 1.02, g["head_cy"] - g["head_r"] * 0.2),
            (cx + g["head_r"] * 0.5, g["head_cy"] - g["head_r"] * 0.55),
            (cx, g["head_cy"] - g["head_r"] * 0.2),
            (cx - g["head_r"] * 0.5, g["head_cy"] - g["head_r"] * 0.55),
        ],
        fill=col("hair_front"),
    )
    # accessory (side clip)
    draw.ellipse((cx + g["head_r"] * 0.55, g["head_cy"] - g["head_r"] * 0.9,
                  cx + g["head_r"] * 0.85, g["head_cy"] - g["head_r"] * 0.6), fill=col("accessories"))


def _draw_pretty(draw: ImageDraw.ImageDraw, g: dict, hair: RGB, eyes: RGB, outfit: RGB):
    """The presentable, flat-shaded character with clean outlines."""
    cx = g["cx"]
    hair_back = _darker(hair, 0.8)
    hair_front = _lighter(hair, 1.05)
    ow = max(2, int(g["w"] * 0.004))  # outline width

    # hair back
    draw.ellipse(
        (cx - g["head_r"] * 1.5, g["head_cy"] - g["head_r"] * 1.5,
         cx + g["head_r"] * 1.5, g["head_cy"] + g["head_r"] * 3.2),
        fill=hair_back, outline=_OUTLINE, width=ow,
    )
    # dress / clothing
    draw.polygon(
        [
            (cx - g["shoulder_half"], g["shoulder_y"]),
            (cx + g["shoulder_half"], g["shoulder_y"]),
            (cx + g["hip_half"], g["hip_y"]),
            (cx - g["hip_half"], g["hip_y"]),
        ],
        fill=outfit, outline=_OUTLINE,
    )
    # collar accent
    draw.polygon(
        [
            (cx - g["shoulder_half"] * 0.5, g["shoulder_y"]),
            (cx + g["shoulder_half"] * 0.5, g["shoulder_y"]),
            (cx, g["shoulder_y"] + g["head_r"] * 0.5),
        ],
        fill=_lighter(outfit, 1.15),
    )
    # neck
    draw.rectangle(
        (cx - g["head_r"] * 0.28, g["head_cy"] + g["head_r"] * 0.7,
         cx + g["head_r"] * 0.28, g["shoulder_y"] + 4),
        fill=_SKIN_SHADOW, outline=_OUTLINE, width=ow,
    )
    # ears
    for sign, _name in ((-1, "ear_L"), (1, "ear_R")):
        ex = cx + sign * g["head_r"] * 0.95
        draw.ellipse((ex - g["head_r"] * 0.13, g["ear_cy"] - g["head_r"] * 0.18,
                      ex + g["head_r"] * 0.13, g["ear_cy"] + g["head_r"] * 0.18),
                     fill=_SKIN, outline=_OUTLINE, width=ow)
    # head
    draw.ellipse(g["head_box"], fill=_SKIN, outline=_OUTLINE, width=ow)
    # blush
    for sign in (-1, 1):
        bx = cx + sign * g["head_r"] * 0.55
        draw.ellipse((bx - g["head_r"] * 0.16, g["eye_cy"] + g["eye_ry"] * 0.6,
                      bx + g["head_r"] * 0.16, g["eye_cy"] + g["eye_ry"] * 1.2), fill=_BLUSH)
    # brows
    for sign in (-1, 1):
        ex = cx + sign * g["eye_dx"]
        draw.line((ex - g["eye_rx"], g["brow_cy"] + 4, ex + g["eye_rx"], g["brow_cy"]),
                  fill=_darker(hair, 0.7), width=max(3, ow + 1))
    # eyes (whites, iris, highlight)
    for sign in (-1, 1):
        ex = cx + sign * g["eye_dx"]
        box = (ex - g["eye_rx"], g["eye_cy"] - g["eye_ry"], ex + g["eye_rx"], g["eye_cy"] + g["eye_ry"])
        draw.ellipse(box, fill=(255, 255, 255), outline=_OUTLINE, width=ow)
        iris = (ex - g["eye_rx"] * 0.7, g["eye_cy"] - g["eye_ry"] * 0.7,
                ex + g["eye_rx"] * 0.7, g["eye_cy"] + g["eye_ry"] * 0.9)
        draw.ellipse(iris, fill=eyes)
        draw.ellipse((ex - g["eye_rx"] * 0.3, g["eye_cy"] - g["eye_ry"] * 0.2,
                      ex + g["eye_rx"] * 0.2, g["eye_cy"] + g["eye_ry"] * 0.5), fill=_darker(eyes, 0.6))
        draw.ellipse((ex - g["eye_rx"] * 0.5, g["eye_cy"] - g["eye_ry"] * 0.5,
                      ex - g["eye_rx"] * 0.1, g["eye_cy"] - g["eye_ry"] * 0.1), fill=(255, 255, 255))
    # nose
    draw.line((cx, g["nose_cy"] - 3, cx - 4, g["nose_cy"] + 3), fill=_SKIN_SHADOW, width=ow)
    # mouth
    draw.arc((cx - g["head_r"] * 0.18, g["mouth_cy"] - 12, cx + g["head_r"] * 0.18, g["mouth_cy"] + 10),
             start=20, end=160, fill=(200, 90, 110), width=max(3, ow + 1))
    # hair front fringe
    draw.polygon(
        [
            (cx - g["head_r"] * 1.02, g["head_cy"] - g["head_r"] * 0.2),
            (cx - g["head_r"] * 0.9, g["head_cy"] - g["head_r"] * 1.1),
            (cx + g["head_r"] * 0.9, g["head_cy"] - g["head_r"] * 1.1),
            (cx + g["head_r"] * 1.02, g["head_cy"] - g["head_r"] * 0.2),
            (cx + g["head_r"] * 0.5, g["head_cy"] - g["head_r"] * 0.55),
            (cx, g["head_cy"] - g["head_r"] * 0.2),
            (cx - g["head_r"] * 0.5, g["head_cy"] - g["head_r"] * 0.55),
        ],
        fill=hair_front, outline=_OUTLINE, width=ow,
    )
    # accessory clip
    ax0 = cx + g["head_r"] * 0.55
    draw.ellipse((ax0, g["head_cy"] - g["head_r"] * 0.9, ax0 + g["head_r"] * 0.3, g["head_cy"] - g["head_r"] * 0.6),
                 fill=(245, 120, 150), outline=_OUTLINE, width=ow)


def render(prompt: str, width: int, height: int, seed=None):
    """Render the demo character. Returns ``(composite_rgba, region_map_rgb)``."""
    width = max(256, int(width))
    height = max(256, int(height))
    rng = _seeded_rng(prompt or "ka-myii", seed)

    hair = _pick_color(prompt or "", ("hair",), _COLOR_WORDS["blue"], rng)
    eyes = _pick_color(prompt or "", ("eye", "eyes"), _COLOR_WORDS["green"], rng >> 8)
    outfit = _pick_color(prompt or "", ("dress", "outfit", "clothes", "uniform", "shirt"),
                         _COLOR_WORDS["pink"], rng >> 16)

    g = _compute_geometry(width, height, rng)

    # Soft pastel background gradient for presentation.
    composite = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    bg = Image.new("RGBA", (width, height), (248, 246, 252, 255))
    top = _lighter(outfit, 1.4)
    for y in range(height):
        t = y / height
        r = int(248 * (1 - t) + top[0] * 0.25 * t)
        gg = int(246 * (1 - t) + top[1] * 0.25 * t)
        b = int(252 * (1 - t) + top[2] * 0.25 * t)
        ImageDraw.Draw(bg).line((0, y, width, y), fill=(r, gg, b, 255))
    composite = Image.alpha_composite(composite, bg)

    char = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    _draw_pretty(ImageDraw.Draw(char), g, hair, eyes, outfit)
    composite = Image.alpha_composite(composite, char)

    # Region map (flat ids, no anti-aliasing blending issues -> use nearest).
    region_map = Image.new("RGB", (width, height), REGION_COLORS["background"])
    _draw_regions(ImageDraw.Draw(region_map), g)

    return composite.convert("RGBA"), region_map


def render_demo_character(prompt: str, width: int, height: int, seed=None) -> Image.Image:
    """Convenience wrapper returning just the presentable composite image."""
    composite, _ = render(prompt, width, height, seed)
    return composite
