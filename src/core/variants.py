"""
Alternate hairstyles and outfits for a separated character, with the hidden regions drawn.

The character is composited without the part being replaced; See-through's other layers are
complete underneath it (amodal). The part's visible area, grown, is repainted from that picture
by SDXL with the new description, and the variant keeps only pixels that are character (anime
segmentation) and changed from what was underneath. Where the new part is smaller the layers
underneath show through (a shoulder under shorter hair); where nothing was under it the repaint
drew it, and that belongs to the variant. The preview hides the replaced layers while a variant
is on (rigger.js: "toggle Hair: …" layers); exports carry it as a hidden layer.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageFilter

from src.core import expressions

logger = logging.getLogger(__name__)

PARTS = {"hair": ("front hair", "side hair", "back hair"), "top": ("topwear", "neckwear"),
         "bottom": ("bottomwear",), "headwear": ("headwear",)}
LABELS = {"hair": "Hair", "top": "Top", "bottom": "Bottom", "headwear": "Headwear"}
# Brief tags about the replaced part would fight the new description ("long hair" vs "bob").
PART_WORDS = {"hair": ("hair", "bangs", "ponytail", "twintails", "braid", "bun", "ahoge", "sidelocks"),
              "top": ("shirt", "hoodie", "jacket", "sweater", "coat", "blouse", "top", "dress", "uniform", "vest",
                      "collar", "sleeve", "necktie", "cardigan", "kimono", "armor"),
              "bottom": ("skirt", "shorts", "pants", "trousers", "jeans", "dress", "kimono"),
              "headwear": ("hat", "cap", "hood", "headwear", "beret", "crown", "helmet", "headband", "hairband")}
FEATURES = ("eyewhite", "irides", "eyelash", "eyebrow", "mouth", "nose")
CHANGED = 28  # channel difference that counts as newly drawn


def brief_without(prompt: str, part: str) -> str:
    """The character brief minus tags that describe the part being replaced."""
    return ", ".join(t for t in (s.strip() for s in (prompt or "").split(","))
                     if t and not any(w in t.lower() for w in PART_WORDS[part]))


def _grow(mask: np.ndarray, px: int) -> np.ndarray:
    px = max(1, int(px))
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(px * 2 + 1))) > 0


def paint(generator, model_dir: Path, part: str, description: str, seed: Optional[int] = None,
          task_id: Optional[str] = None, brief: str = "") -> Dict:
    """Paint one alternate ``part`` ("hair", "top", ...) from ``description``; adds it to
    expressions.json["variants"] and returns the updated record."""
    from src.core import outpaint

    if part not in PARTS:
        raise ValueError(f"Choose one of {', '.join(PARTS)}")
    model_dir = Path(model_dir)
    paths = sorted((model_dir / "assets").glob("[0-9]*_*.png"))
    names = [p.stem.split("_", 1)[1] for p in paths]
    layers = []
    for p in paths:
        with Image.open(p) as img:
            layers.append(img.convert("RGBA"))
    replaced = [i for i, n in enumerate(names) if n.split("-")[0] in PARTS[part]]
    if not replaced:
        raise ValueError(f"This character has no {LABELS[part].lower()} layer to replace")
    size = layers[0].size
    if part == "hair" and any(n.split("-")[0] == "headwear" for n in names):
        # See-through files part of a hat's crown under "back hair": above the face it stays,
        # as hat, or hiding the hair left hat fragments (witch, GPU).
        face_top = next((int(np.nonzero(np.asarray(l)[..., 3] > 128)[0].min()) for n, l in zip(names, layers)
                         if n == "face" and (np.asarray(l)[..., 3] > 128).any()), 0)
        for i in [i for i in replaced if names[i] == "back hair"]:
            crown = np.asarray(layers[i]).copy()
            crown[face_top:, :, 3] = 0
            rest = np.asarray(layers[i]).copy()
            rest[:face_top, :, 3] = 0
            layers[i] = Image.fromarray(rest)
            layers.insert(i, Image.fromarray(crown)); names.insert(i, "back hair crown")
            replaced = [j + 1 if j >= i else j for j in replaced]
    under = Image.new("RGBA", size, (255, 255, 255, 255))
    for i, layer in enumerate(layers):
        if i not in replaced:
            under.alpha_composite(layer)
    alphas = [np.asarray(l)[..., 3] for l in layers]
    # Where the old part shows: its pixels not covered by a layer drawn above it.
    region = np.zeros(alphas[0].shape, bool)
    for i in replaced:
        covered = np.zeros_like(region)
        for j in range(i + 1, len(layers)):
            if j not in replaced:
                covered |= alphas[j] > 200
        region |= (alphas[i] > 128) & ~covered
    face = next((a for n, a in zip(names, alphas) if n == "face"), None)
    on_face = face is not None and (face > 128).any()
    fys, fxs = np.nonzero(face > 128) if on_face else (np.array([size[1] * .3]), np.array([0, size[0] * .2]))
    fw, chin = np.ptp(fxs), fys.max()
    region = _grow(region, fw * (.06 if part in ("hair", "headwear") else .04))
    # Differential change map: the head area is drawn anew; below the chin the picture underneath
    # (the complete clothes under long hair) keeps its structure and is only redrawn late.
    # Painted at full strength there, a witch's braids became a cape over her dress.
    change = np.where(np.arange(size[1])[:, None] < chin + .2 * fw, 255, 150).astype(np.uint8) * region
    start = under.convert("RGB")
    words = PART_WORDS[part] + (PART_WORDS["headwear"] if part == "hair" else ())
    prompt = ", ".join(p for p in (description, ", ".join(t for t in (x.strip() for x in (brief or "").split(","))
                                                           if t and not any(w in t.lower() for w in words))) if p)
    negative = "extra arms, extra person, text, frame" + (", hat, headwear" if part == "hair" else "")
    painted = generator.inpaint(start, Image.fromarray(change), prompt, negative=negative, strength=1.0,
                                seed=7 if seed is None else int(seed), task_id=task_id, graded=True,
                                label=f"Painting the alternate {LABELS[part].lower()}")
    diff = np.abs(np.asarray(painted, np.int16) - np.asarray(start, np.int16)).max(-1) > CHANGED
    keep = region & _grow(diff, 2) & outpaint.character_mask(painted)
    # Layers drawn over the part (a hat over the hair, a bag over the coat) stay themselves.
    for j in range(max(replaced) + 1, len(layers)):
        keep &= ~(alphas[j] > 128)
    if part in ("hair", "headwear"):  # blinking and lip sync stay on the face's own layers
        for n, a in zip(names, alphas):
            if n.split("-")[0] in FEATURES:
                keep &= ~_grow(a > 16, max(2, fw * .04))
        if on_face:
            # Over the face: only inside the old bangs (not the back hair behind it). Grown bangs
            # reached between a witch's eyes and kept repainted lash fragments.
            front = names.index("face")
            bangs = np.zeros_like(keep)
            for i in replaced:
                if i > front:
                    bangs |= alphas[i] > 128
            # Inside them the repainted forehead is kept too: cleaner than See-through's guess
            # of the skin under the old bangs, which shows faint strands.
            keep &= ~((face > 128) & ~bangs)
        # Elsewhere the new hair covers only background: body, ears and the like keep their own
        # (complete) layers. Black hair over a black dress cannot be told apart by colour, and
        # painted there the witch's braids came back as a cape. Over bare canvas a new pixel must
        # stand out from it (old hair's outline otherwise left a grey haze).
        others = np.zeros_like(keep)
        for i, a in enumerate(alphas):
            if i not in replaced and names[i] != "face":
                others |= a > 128
        bare = ~others & ~((face > 128) if on_face else np.zeros_like(keep))
        keep &= ~others
        # Over bare canvas a pixel must look more like the new hair (its colour from the new bangs)
        # than like the canvas: where See-through put part of a hat in the back hair, the gap
        # was filled with grey haze that is neither.
        rgb = np.asarray(painted, np.float32)
        sample = keep & ((face > 128) if on_face else keep)
        hair = np.median(rgb[sample if sample.sum() > 50 else keep], axis=0) if keep.any() else np.zeros(3)
        canvas = np.median(rgb[bare & ~region], axis=0) if (bare & ~region).any() else np.full(3, 255.)
        keep &= ~(bare & (np.abs(rgb - hair).sum(-1) >= np.abs(rgb - canvas).sum(-1)))
    if keep.sum() < 50:
        raise RuntimeError("The new design did not change anything; try a more different description")
    made = expressions.read(model_dir)
    variants = made.setdefault("variants", {})
    vid = next(f"{part}-{k}" for k in range(1, 1000) if f"{part}-{k}" not in variants)
    relative = f"{expressions.FOLDER}/variant_{vid}.png"
    layer = painted.convert("RGBA")
    layer.putalpha(Image.fromarray(keep.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(.8)))
    if "back hair crown" in names:  # the rig hides all of the back hair while this is on
        layer = Image.alpha_composite(layers[names.index("back hair crown")], layer)
    (model_dir / expressions.FOLDER).mkdir(parents=True, exist_ok=True)
    layer.save(model_dir / relative)
    short = " ".join(description.split())[:28]
    variants[vid] = {"part": part, "label": f"{LABELS[part]}: {short}", "path": relative,
                     "after": names[max(replaced)]}  # the preview stacks it where the old part's top layer was
    (model_dir / expressions.FOLDER / "expressions.json").write_text(json.dumps(made, indent=2), encoding="utf-8")
    logger.info("Variant %s on %s: %d px", vid, model_dir.name, int(keep.sum()))
    return made
