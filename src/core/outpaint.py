"""
Extend a character cut off by the image border: a larger canvas around the artwork and a mask
of what is new, for SDXL to paint (src/core/image_generator.py ``inpaint``).

"full_body": the artwork is scaled down into the top of a same-sized portrait canvas and the
model paints the rest of the body. "uncrop": only the borders the character touches get room
(a hat cut off at the top, an elbow at the side), keeping the framing. Either way the artwork's
edges are redrawn gradually (a graded mask), deepest where it was cut, so the join is drawn.
"""
from __future__ import annotations

import json
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# The natural stance of a VTuber model sheet: standing straight-on, arms relaxed at the sides.
FULL_BODY_TAGS = "full body, standing, straight-on, arms at sides, legs, feet, shoes, simple background, white background"
FULL_BODY_NEGATIVE = ("cropped, out of frame, close-up, portrait, upper body, cowboy shot, feet out of frame, "
                      "from above, from below, sitting, multiple views, border, frame")
# Framing tags in a saved brief that would fight a full-body extension.
CROP_TAGS = {"upper body", "waist up", "portrait", "close-up", "cowboy shot", "bust", "bust shot"}
# Artistic targets, not an anatomy rule: owners can retain compact designs deliberately.
PROPORTIONS = {"balanced": 6.75, "compact": 5.5, "tall": 7.5}
HEADS = PROPORTIONS["balanced"]
# Crown-to-chin height of a typical head in face widths, measured on this checkpoint's own full
# bodies (1.33 on both). Bodies are sized from the face so voluminous hair cannot shrink it.
HEAD_PER_FACE = 1.33


def body_unit(geometry: dict) -> float:
    """A typical head height for this face (px): what proportions and the pose are built on."""
    face = geometry.get("face")
    return HEAD_PER_FACE * (face[2] - face[0]) if face else float(geometry.get("head_height") or 1)
UNCROP_NEGATIVE = "cropped, out of frame, border, frame"


def touched_borders(fg: np.ndarray, threshold: float = 0.02) -> dict:
    """Which canvas borders the character runs into (share of the border row/column)."""
    return {"top": fg[0].mean() > threshold, "bottom": fg[-1].mean() > threshold,
            "left": fg[:, 0].mean() > threshold, "right": fg[:, -1].mean() > threshold}


def plan(image: Image.Image, fg: np.ndarray, mode: str = "full_body",
         head: Optional[int] = None, geometry: Optional[dict] = None,
         proportions: str = "balanced", keep_face: bool = True, identity: Optional[np.ndarray] = None,
         guides: Optional[dict] = None) -> Tuple[Image.Image, np.ndarray, dict]:
    """(canvas, mask, info): the canvas holds the artwork where it goes, the mask (uint8) is how
    much the model may repaint each pixel. ``fg`` is the character mask of ``image`` (bool, image size); ``head``
    the head's height in px, when known, so a full body gets natural proportions. ``guides``: the
    owner's dragged pose targets (see ``check_guides``)."""
    if proportions not in PROPORTIONS:
        raise ValueError("Choose balanced, compact or tall proportions")
    geometry = geometry or {}
    head = geometry.get("head_height") or head
    image = image.convert("RGB")
    W, H = image.size
    touched = touched_borders(fg)
    background = _background_colour(image, fg)
    if mode == "full_body":
        # Measure crown-to-chin, not only the visible face under the fringe. Align the
        # neck's centre to the new standing axis rather than centring the old canvas.
        target = PROPORTIONS[proportions]
        head_top = H * (.12 if geometry.get("large_headwear") else .04)
        # Crown to chin as drawn, then the body in typical heads for this face: a fluffy hairdo
        # made the head "tall", which shrank the face on a broad body (Sol's hoodie extension).
        if head and geometry.get("face"):
            figure = (geometry["face"][3] - geometry.get("crown_y", geometry["face"][3] - head)) + (target - 1) * body_unit(geometry)
            scale = min(0.6, (H * .94 - head_top) / figure)
        else:
            scale = min(0.6, (H*.94-head_top) / (head * target)) if head else 0.55 * 5.5 / target
        w, h = max(2, int(W * scale)), max(2, int(H * scale))
        axis = float(geometry.get("axis_x", W / 2))
        left = int(np.clip(round(W / 2 - axis * scale), 0, W - w))
        top = max(0, round(head_top - geometry.get("crown_y", 0) * scale))
        top = min(top, H - h)
        out_w, out_h = W, H
    elif mode == "uncrop":
        pad = {side: (int((W if side in ("left", "right") else H) * 0.12) if hit else 0) for side, hit in touched.items()}
        grown_w, grown_h = W + pad["left"] + pad["right"], H + pad["top"] + pad["bottom"]
        scale = min(W / grown_w, H / grown_h)  # same output size, the artwork shrinks a little
        w, h = int(W * scale), int(H * scale)
        left = int(pad["left"] * scale) + (W - int(grown_w * scale)) // 2
        top = int(pad["top"] * scale) + (H - int(grown_h * scale)) // 2
        out_w, out_h = W, H
    else:
        raise ValueError("mode must be full_body or uncrop")

    canvas = Image.new("RGB", (out_w, out_h), background)
    canvas.paste(image.resize((w, h), Image.Resampling.LANCZOS), (left, top))
    # A change map for differential inpainting (0 keep … 255 repaint): new canvas 255; inside the
    # artwork 0, rising towards its edges. Where the character was cut off the ramp reaches deep,
    # so a cut brim or sleeve is reshaped into the new drawing instead of continued as a block;
    # where only background meets, a few px blend the tone.
    m = min(w, h)
    cut = dict(touched, bottom=touched["bottom"] or mode == "full_body")

    def ramp(dist, side):
        full, depth = ((0.03 * h, 0.2 * h) if side == "bottom" else (0.06 * m, 0.25 * m)) if cut[side] else (0, 0.04 * m)
        return np.clip((depth - dist) / (depth - full), 0, 1)

    ys, xs = np.arange(h)[:, None], np.arange(w)[None, :]
    change = np.maximum(np.maximum(ramp(ys, "top"), ramp(h - 1 - ys, "bottom")),
                        np.maximum(ramp(xs, "left"), ramp(w - 1 - xs, "right")))
    if mode == "full_body" and geometry.get("face"):
        # Let the lower chest/outfit be drawn as one continuous form with the new body.
        # Keep the collar and face; a hard retained torso rectangle encouraged bent joins.
        fx0, fy0, fx1, chin = geometry["face"]
        # Redraw from the jaw down: keeping the old neck and collar to 0.1 head below the chin left
        # a plate of the original's shading and a floating collar (hoodie, witch).
        t = np.clip((ys - chin * scale) / max(1, head * scale * .15), 0, 1)
        change = np.maximum(change, t * t * (3 - 2 * t))
        if keep_face:
            # Soft protected inner face, including eyes/nose/mouth. Hair/cropped edges can
            # still be repaired; this stops a broad edge ramp repainting the identity.
            if identity is not None and identity.shape == (H, W):
                kept = np.asarray(Image.fromarray(identity).resize((w,h), Image.Resampling.LANCZOS), dtype=np.float32) / 255
                change = np.minimum(change, 1 - kept)
            else:
                ix0, iy0, ix1, iy1 = geometry.get("identity_box", geometry["face"])
                cx, cy = (ix0 + ix1) * scale / 2, (iy0 + iy1) * scale / 2
                rx, ry = max(1, (ix1 - ix0) * scale * .52), max(1, (iy1 - iy0) * scale * .52)
                r = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
                change = np.minimum(change, np.clip((r - .82) / .22, 0, 1))
    mask = np.full((out_h, out_w), 255, np.uint8)
    mask[top:top + h, left:left + w] = np.rint(change * 255).astype(np.uint8)
    info = {"touched": [side for side, hit in touched.items() if hit], "placed": [left, top, w, h], "mode": mode,
            "proportions": proportions, "target_heads": PROPORTIONS[proportions], "keep_face": keep_face,
            "canvas": [W, H], "measured": bool(head), "standing_axis": W / 2, "foot_y": H * .94}
    if geometry.get("face"):
        info["result_geometry"] = {"large_headwear": bool(geometry.get("large_headwear")), "face": [round(left + v * scale) if i % 2 == 0 else round(top + v * scale)
                                              for i, v in enumerate(geometry["face"])],
                                   "head_height": max(1, round(head * scale)),
                                   "unit": round(body_unit(geometry) * scale, 1),
                                   "axis_x": left + geometry.get("axis_x", W / 2) * scale,
                                   "crown_y": top + geometry.get("crown_y", 0) * scale}
        if geometry.get("identity_box"):
            info["result_geometry"]["identity_box"] = [round(left + v * scale) if i % 2 == 0 else round(top + v * scale)
                                                        for i,v in enumerate(geometry["identity_box"])]
    if mode == "full_body":
        if guides:
            info["guides"] = guides
        info["pose_points"] = pose_points(info)
    return canvas, mask, info


def anatomy(model_dir) -> dict:
    """Head and standing axis from separated art, excluding sparse ahoge/ear/hat tips.

    Face layers can begin below the crown; hair occupancy above them supplies the missing
    height. A substantial row width rejects single strands. Saved extension geometry makes
    an artwork-first result measurable before it has been separated.
    """
    face = next(iter(sorted((model_dir / "assets").glob("*_face.png"))), None)
    if face is None:
        try:
            saved = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
            g = ((saved.get("request") or {}).get("extension") or {}).get("result_geometry", {})
            if g and all(np.isfinite(v) for v in [*g["face"], g["head_height"], g["axis_x"], g["crown_y"]]):
                return g
        except (OSError, ValueError, TypeError, KeyError):
            pass
        return {}
    with Image.open(face) as img:
        mask = np.asarray(img.getchannel("A")) > 128
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {}
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    fw, fh = x1 - x0, y1 - y0
    crown = y0
    for name in ("front hair", "back hair", "side hair"):
        for path in (model_dir / "assets").glob("*_" + name + ".png"):
            with Image.open(path) as img:
                hair = np.asarray(img.getchannel("A")) > 128
            if hair.shape != mask.shape:
                continue
            counts = hair[:y0 + 1, max(0, x0):x1].sum(axis=1)
            rows = np.flatnonzero(counts >= max(4, fw * .18))
            if len(rows):
                crown = min(crown, int(rows[0]))
    height = int(np.clip(y1 - crown, max(fh, fw * 1.05), fh * 1.4))
    crown = y1 - height
    axis = (x0 + x1) / 2
    neck = next(iter(sorted((model_dir / "assets").glob("*_neck.png"))), None)
    if neck:
        with Image.open(neck) as img:
            n = np.asarray(img.getchannel("A")) > 128
        if n.shape == mask.shape:
            nx = np.nonzero(n[max(0, y1 - int(fh * .04)):min(n.shape[0], y1 + int(fh * .18))])[1]
            if len(nx):
                axis = float(np.clip(np.median(nx), axis - fw * .15, axis + fw * .15))
    ink_top = []
    for path in (model_dir / "assets").glob("*.png"):
        if any(part in path.stem for part in ("eyewhite", "eyebrow", "eyelash")):
            with Image.open(path) as img:
                box = img.getchannel("A").point(lambda a: 255 if a > 128 else 0).getbbox()
            if box:
                ink_top.append(box[1])
    keep_top = max(y0, min(ink_top) - fh * .08) if ink_top else y0
    large_headwear = False
    for path in (model_dir / "assets").glob("*_headwear.png"):
        with Image.open(path) as img:
            hat = np.asarray(img.getchannel("A")) > 128
        if hat.sum() > mask.sum() * .25:
            large_headwear = True
    return {"large_headwear": large_headwear, "face": [x0, y0, x1, y1], "head_height": height, "crown_y": crown, "axis_x": axis,
            "identity_box": [x0, round(keep_top), x1, y1]}


def identity_mask(model_dir, geometry: dict) -> Optional[np.ndarray]:
    """Keep one coherent head silhouette, not a small face plate over a newly drawn head.

    Only the head is retained: the lower outfit is free to become a continuous body.
    Cropped border strips still get repaired so a cut hat does not retain a straight edge.
    """
    from PIL import ImageChops, ImageFilter

    face = next(iter(sorted((model_dir / "assets").glob("*_face.png"))), None)
    if face is None:
        return None
    with Image.open(face) as img:
        keep = img.getchannel("A")
    head_names=("face","front hair","back hair","side hair","headwear","ears","earwear",
                "eyewhite","irides","eyebrow","eyelash","mouth","nose")
    for path in (model_dir / "assets").glob("[0-9]*_*.png"):
        name=path.stem.split("_",1)[1].split("-")[0]
        if name not in head_names:
            continue
        with Image.open(path) as img:
            if img.size==keep.size:
                keep=ImageChops.lighter(keep,img.getchannel("A"))
    arr=(np.array(keep)>128).astype(np.float32)
    H,W=arr.shape; yy,xx=np.ogrid[:H,:W]
    chin=geometry["face"][3]; head=geometry["head_height"]
    v=np.clip((yy-chin)/max(1,head*.10),0,1)
    arr*=1-v*v*(3-2*v)
    # Keep the same spatial repair strips as the extension mask for cut artwork.
    border=max(1,min(W,H)*.10)
    edge=np.minimum(np.minimum(xx,W-1-xx),np.minimum(yy,H-1-yy))
    arr*=np.clip(edge/border,0,1)
    return np.array(Image.fromarray(np.uint8(np.clip(arr*255,0,255))).filter(ImageFilter.GaussianBlur(.5)))


def head_height(model_dir) -> Optional[int]:
    """Compatibility helper for callers that only need the measured crown-to-chin height."""
    return anatomy(model_dir).get("head_height")


def full_body_prompt(prompt: str) -> str:
    """The brief without framing tags such as "upper body"."""
    return ", ".join(t for t in (s.strip() for s in prompt.split(",")) if t and t.lower() not in CROP_TAGS)


def _background_colour(image: Image.Image, fg: np.ndarray):
    px = np.asarray(image)[~fg] if (~fg).sum() > 100 else np.asarray(image).reshape(-1, 3)
    return tuple(int(v) for v in np.median(px, axis=0))


def character_mask(image: Image.Image) -> np.ndarray:
    """The character (segmentation model, colour key as a fallback)."""
    from src.core.asset_separator import AssetSeparator, _anime_seg_mask

    alpha = _anime_seg_mask(image)
    if alpha is None:
        alpha = np.asarray(AssetSeparator._chroma_cutout(image.convert("RGBA")).split()[-1])
    return alpha > 128



# Apache-2.0; downloaded from Hugging Face only for guided full-body extension.
POSE_MODEL = "xinsir/controlnet-openpose-sdxl-1.0"
POSE_REVISION = "23f966cd5cfdd3f7729c903e243d87152162d2b7"


def pose_points(info: dict) -> list:
    """OpenPose's 18 body joints, anchored to the actual retained face, not a generic canvas.

    These are artistic stance targets; not an inferred pose/skeleton from the input image.
    """
    W,H=info["canvas"];g=info.get("result_geometry") or {}
    # Body measures come from the face ("unit", see body_unit); older plans used the head height.
    hh=float(g.get("unit") or g.get("head_height",H*.90/info["target_heads"]))
    crown=float(g.get("crown_y",H*.04));cx=float(g.get("axis_x",W/2))
    face=g.get("face",[cx-hh*.33,crown+hh*.15,cx+hh*.33,crown+hh])
    x0,y0,x1,chin=face;fw=x1-x0
    nose=((x0+x1)/2,y0+(chin-y0)*.72)
    neck=(cx,chin+hh*.18)
    # Dragged guides are [half width, y] per mirrored pair: level shoulders and hips by design.
    guides=info.get("guides") or {}
    # Shoulder joints 0.53 typical heads out (0.7 face widths): drawn silhouettes then measure
    # ~1.6 face widths, like this checkpoint's own figures (1.5-1.75); 0.58 head heights made
    # extended bodies ~1.9 wide.
    shoulders,shoulder=guides.get("shoulders",(hh*.53,chin+hh*.34))
    ankles,foot=guides.get("ankles",(hh*.29,info["foot_y"]-hh*.45))
    hips,hip=guides.get("hips",(hh*.29,neck[1]+(foot-neck[1])*.36))
    knee=hip+(foot-hip)*.52;knees=(hips+ankles)/2;wrists=shoulders+hh*.2
    elbow=shoulder+(hip+hh*.55-shoulder)*.53;wrist=hip+hh*.55
    points=[nose,neck,(cx-shoulders,shoulder),(cx-wrists,elbow),(cx-wrists,wrist),
            (cx+shoulders,shoulder),(cx+wrists,elbow),(cx+wrists,wrist),
            (cx-hips,hip),(cx-knees,knee),(cx-ankles,foot),
            (cx+hips,hip),(cx+knees,knee),(cx+ankles,foot),
            (nose[0]-fw*.24,nose[1]-fw*.15),(nose[0]+fw*.24,nose[1]-fw*.15),
            (nose[0]-fw*.46,nose[1]-fw*.04),(nose[0]+fw*.46,nose[1]-fw*.04)]
    return [(float(np.clip(x,1,W-2)),float(np.clip(y,1,H-2))) for x,y in points]


GUIDES = ("shoulders", "hips", "ankles")


def check_guides(data, canvas) -> Optional[dict]:
    """Validated pose guides from the Studio: {"shoulders": [half width, y], ...} in canvas px,
    top to bottom. Raises ValueError for anything else."""
    if data is None:
        return None
    W, H = canvas
    if not isinstance(data, dict) or not set(data) <= set(GUIDES):
        raise ValueError("Guides must be shoulders, hips and ankles")
    out = {}
    for name in GUIDES:
        if name not in data:
            continue
        v = data[name]
        if (not isinstance(v, (list, tuple)) or len(v) != 2
                or any(isinstance(n, bool) or not isinstance(n, (int, float)) for n in v)
                or not (4 <= v[0] <= W / 2 and 0 <= v[1] <= H)):
            raise ValueError(f"The {name} guide is outside the canvas")
        out[name] = [float(v[0]), float(v[1])]
    ys = [out[n][1] for n in GUIDES if n in out]
    if ys != sorted(ys):
        raise ValueError("Keep the shoulders above the hips and the hips above the ankles")
    return out or None


def pose_image(info: dict) -> Image.Image:
    """Original OpenPose-format line drawing. No pose detector or OpenCV dependency."""
    import colorsys
    from PIL import ImageDraw

    W,H=info["canvas"];points=pose_points(info)
    image=Image.new("RGB",(W,H));draw=ImageDraw.Draw(image)
    links=[(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(8,9),(9,10),
           (1,11),(11,12),(12,13),(1,0),(0,14),(14,16),(0,15),(15,17)]
    colours=[tuple(round(c*255) for c in colorsys.hsv_to_rgb(i/18,1,1)) for i in range(18)]
    width=max(3,round(max(W,H)/128));radius=max(3,round(width*.6))
    for i,(a,b) in enumerate(links):
        draw.line([points[a],points[b]],fill=colours[i],width=width)
    for i,(x,y) in enumerate(points):
        draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=colours[i])
    return image
