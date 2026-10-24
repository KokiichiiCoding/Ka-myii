"""
Automatic review of a drawn full body: a second face or figure, shoes or a hat cut off by the
border, a tilted shoulder line, hands the tagger calls bad. Pixel checks need no model; tag
checks use the WD tagger (src/core/tagger.py) when it is installed. Each finding is a hint with
a suggested fix for the Studio ("reseed", "uncrop", "repaint" a box, "refine_hands"), not a verdict.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

DUPLICATE_TAGS = ("multiple girls", "2girls", "3girls", "multiple boys", "2boys", "multiple views", "clone")
TAG_THRESHOLD = 0.35
EYE_COLOURS = ("blue", "green", "aqua", "red", "purple", "brown", "yellow", "pink", "grey", "black", "orange", "heterochromia")
TILT_LIMIT = 0.1  # of a head height between the two shoulder lines
HEADS_LIMIT = 0.15  # relative miss of the planned height in heads


def face_tiles(image, info: Optional[dict]) -> list:
    """Square crops down the body below the retained head, for the tagger to look for a second
    face in. The whole picture is rated "solo" even with a second head on the chest, but a tile
    with a face gets eye-colour / "looking at viewer" tags (0.45-0.9 on a pasted second head)
    and a body tile almost none (<= 0.06 on real full bodies)."""
    g = (info or {}).get("result_geometry")
    if not g:
        return []
    W, H = image.size
    hh, chin, cx = g["head_height"], g["face"][3], g["axis_x"]
    side = int(hh * 1.8)
    tiles, y = [], int(chin + .35 * hh)
    while y < H - side * .5:
        tiles.append(image.crop((int(cx - side / 2), y, int(cx + side / 2), y + side)))
        y += int(side * .6)
    return tiles


def _has_face(scores: dict) -> bool:
    eyes = max((v for k, v in scores.items() if k.endswith(" eyes") and k.split()[0] in EYE_COLOURS), default=0)
    return eyes > TAG_THRESHOLD or scores.get("looking at viewer", 0) > 0.4


def review(image, fg: np.ndarray, info: Optional[dict] = None, tags: Optional[dict] = None,
           tile_tags: Optional[list] = None) -> list:
    """Findings for ``image`` (RGB) with character mask ``fg`` (bool, same size):
    [{"id", "message", "fix", "box"?, "prompt"?}]. ``info`` is the extension plan (its
    result_geometry places the retained head); ``tags`` the tagger output, when available, and
    ``tile_tags`` its output for ``face_tiles``."""
    H, W = fg.shape
    scores = (tags or {}).get("scores", {})
    out = []
    if (max(scores.get(t, 0) for t in DUPLICATE_TAGS) > TAG_THRESHOLD
            or any(_has_face(t.get("scores", {})) for t in tile_tags or [])):
        out.append({"id": "duplicate", "fix": "reseed",
                    "message": "A second face or figure may have been drawn."})
    band = max(2, H // 150)
    if fg[-band:].mean() > 0.01 or scores.get("feet out of frame", 0) > 0.5:
        out.append({"id": "feet", "fix": "uncrop", "message": "The shoes run into the bottom edge."})
    if fg[:band].mean() > 0.01:
        out.append({"id": "top", "fix": "uncrop", "message": "The hat or hair runs into the top edge."})
    g = (info or {}).get("result_geometry")
    tilt = shoulder_tilt(np.asarray(image.convert("RGB")), fg, g) if g else None
    if tilt is not None and abs(tilt) > TILT_LIMIT:
        x0, _, x1, chin = g["face"]
        hh, cx = g["head_height"], g["axis_x"]
        out.append({"id": "shoulders", "fix": "repaint", "tilt": round(tilt, 3),
                    "message": f"The shoulders look uneven: one sits about {abs(tilt) * 100:.0f}% of a head lower.",
                    "box": [round(max(0, cx - hh)), round(chin), round(min(W, cx + hh)), round(min(H, chin + .9 * hh))],
                    "prompt": "level shoulders, symmetrical shoulders, straight-on"})
    heads = heads_tall(fg, g) if g and not any(f["id"] in ("feet", "top") for f in out) else None
    target = (info or {}).get("target_heads") or 6.75
    if heads is not None and abs(heads / target - 1) > HEADS_LIMIT:
        out.append({"id": "head_size", "fix": "reseed",
                    "message": f"The head looks too {'big' if heads < target else 'small'} for the body "
                               f"(about {heads:.1f} heads tall; aiming for {target:g})."})
    if scores.get("bad hands", 0) > TAG_THRESHOLD:
        out.append({"id": "hands", "fix": "refine_hands", "message": "The hands may be malformed."})
    return out


def heads_tall(fg: np.ndarray, g: dict):
    """Height in heads as drawn, or None: the head crown to chin plus the body below the chin,
    both in typical heads for this face (outpaint.body_unit), the measure the plan aims for.
    This checkpoint's own full bodies measure 6.3-6.7; the first guided extensions 8.0-8.4,
    which is why their heads looked small."""
    from src.core.outpaint import HEAD_PER_FACE

    x0, _, x1, chin = g["face"]
    unit, cx = HEAD_PER_FACE * max(1, x1 - x0), g["axis_x"]
    crown = g.get("crown_y", chin - g["head_height"])
    rows = np.flatnonzero(fg[:, max(0, int(cx - 1.5 * unit)):int(cx + 1.5 * unit)].any(1))
    if not len(rows) or rows.max() <= chin or chin <= crown:
        return None
    return float((rows.max() - crown) / unit)


def _reach(row: np.ndarray, c: int, step: int, gap: int = 4) -> int:
    """How far the run of ``row`` through column ``c`` extends in direction ``step``."""
    last, misses, x = c, 0, c
    while 0 <= x < len(row):
        if row[x]:
            last, misses = x, 0
        else:
            misses += 1
            if misses > gap:
                break
        x += step
    return abs(last - c)


def shoulder_tilt(rgb: np.ndarray, fg: np.ndarray, g: dict) -> Optional[float]:
    """(left shoulder line - right shoulder line) / head height, or None when it can't be told.

    Hair-coloured pixels are left out (long hair over a shoulder is not the shoulder); each
    side's line is the first row below the chin where the body, connected to the neck, reaches
    0.4 head heights out from the standing axis.
    """
    H, W = fg.shape
    x0, _, x1, chin = (int(round(v)) for v in g["face"])
    hh, cx = float(g["head_height"]), int(round(g["axis_x"]))
    crown = int(round(g.get("crown_y", chin - hh)))
    top = slice(max(0, int(crown + .05 * hh)), max(1, int(crown + .25 * hh))), slice(max(0, x0), max(1, x1))
    body = fg.copy()
    if fg[top].sum() >= 20:
        hair = np.median(rgb[top][fg[top]], axis=0)
        body &= np.abs(rgb.astype(np.int16) - hair.astype(np.int16)).sum(-1) > 60
    if not 0 <= cx < W:
        return None
    reach, lines = .4 * hh, {}
    for y in range(max(0, chin), min(H, int(chin + 1.2 * hh))):
        for side, step in (("l", -1), ("r", 1)):
            if side not in lines and _reach(body[y], cx, step) >= reach:
                lines[side] = y
        if len(lines) == 2:
            return (lines["l"] - lines["r"]) / hh
    return None
