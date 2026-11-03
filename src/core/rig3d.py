"""
Rig a Pixal3D mesh of a Ka-myii character (mesh3d.py) into a VRM 0.x avatar: a humanoid
skeleton, skin weights, the VRM T-pose, blink / mouth shapes and an MToon material.

- **Joints** come from DWPose on the artwork (pose.py) — the very pixels the mesh was built
  from — and are lifted into the mesh along the Pixal3D camera's rays to the middle of the
  first solid each ray crosses, so an arm in front of a cape gets the arm's depth.
- **Weights**: torso bones (hips → head) along the spine; arm and leg bones only inside capsules
  whose radius is the limb's measured thickness, so a cape behind an arm or a thigh beside a
  hand stays with the body; then smoothed over the mesh.
- **T-pose**: VRM apps assume the rest pose is a T-pose (they lower the arms from it), so the
  A-pose arms are raised with those weights and that becomes the bind pose.
- **Expressions**: the character's own painted closed eyes and mouth (expressions.py) become
  decals — copies of the face triangles under them, UV-mapped by the same camera — kept just
  under the skin and pushed out by a morph target, so any VRM app blinks and talks with them.
"""
from __future__ import annotations

import io
import json
import logging
import math
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

MIN_MATCH = 0.85      # photo-paint silhouette agreement below which the camera is not trusted
RIG_VERSION = 4       # bump when a CPU refresh improves existing avatars (2: emotion presets, eye decals; 3: chest springs; 4: drawn face marks)
DECAL_GAP = 0.002     # metres a hidden expression decal sits under the skin (later ones a bit more)
ON_FACE = ("cheeks", "tears", "eyes", "temple", "forehead", "mouth", "gloom")   # expression toggle regions
# Drawn locally for every avatar that has no painted one, so VRM apps' Joy / Angry / Sorrow
# hotkeys (and the 3D stage) always have something to show.
DRAWN_MARKS = ("blush", "gloom", "anger", "sweat")
CONFIDENT = 0.3       # DWPose score for a usable joint

# VRM 0.x humanoid bones: (name, parent). "left" is the character's left.
BONES = [
    ("hips", None), ("spine", "hips"), ("chest", "spine"), ("neck", "chest"), ("head", "neck"),
    ("leftShoulder", "chest"), ("leftUpperArm", "leftShoulder"), ("leftLowerArm", "leftUpperArm"),
    ("leftHand", "leftLowerArm"),
    ("rightShoulder", "chest"), ("rightUpperArm", "rightShoulder"), ("rightLowerArm", "rightUpperArm"),
    ("rightHand", "rightLowerArm"),
    ("leftUpperLeg", "hips"), ("leftLowerLeg", "leftUpperLeg"), ("leftFoot", "leftLowerLeg"),
    ("leftToes", "leftFoot"),
    ("rightUpperLeg", "hips"), ("rightLowerLeg", "rightUpperLeg"), ("rightFoot", "rightLowerLeg"),
    ("rightToes", "rightFoot"),
]
INDEX = {name: i for i, (name, _) in enumerate(BONES)}


# ------------------------------------------------------------------------------------------ GLB

_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_glb(path: Path) -> Tuple[dict, bytes]:
    data = Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<III", data)
    if magic != 0x46546C67 or version != 2:
        raise ValueError(f"{Path(path).name} is not a binary glTF 2.0 file")
    n, kind = struct.unpack_from("<II", data, 12)
    gltf = json.loads(data[20:20 + n])
    binary = b""
    if 20 + n + 8 <= len(data):
        m, _ = struct.unpack_from("<II", data, 20 + n)
        binary = data[28 + n:28 + n + m]
    return gltf, binary


def accessor(gltf: dict, binary: bytes, i: int) -> np.ndarray:
    a = gltf["accessors"][i]
    view = gltf["bufferViews"][a["bufferView"]]
    dtype = np.dtype(_COMPONENT[a["componentType"]])
    width = _WIDTH[a["type"]]
    start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = view.get("byteStride") or dtype.itemsize * width
    rows = np.ndarray((a["count"], width), dtype, binary, start, (stride, dtype.itemsize))
    return rows.copy() if width > 1 else rows[:, 0].copy()


def load_mesh(path: Path) -> dict:
    """Positions, normals, UVs, triangles and the base-colour image of a single-material GLB."""
    gltf, binary = read_glb(path)
    pos, nrm, uv, faces = [], [], [], []
    for mesh in gltf["meshes"]:
        for prim in mesh["primitives"]:
            attrs = prim["attributes"]
            base = sum(len(p) for p in pos)
            pos.append(accessor(gltf, binary, attrs["POSITION"]).astype(np.float64))
            nrm.append(accessor(gltf, binary, attrs["NORMAL"]).astype(np.float64) if "NORMAL" in attrs
                       else np.zeros_like(pos[-1]))
            uv.append(accessor(gltf, binary, attrs["TEXCOORD_0"]).astype(np.float64))
            idx = accessor(gltf, binary, prim["indices"]) if "indices" in prim else np.arange(len(pos[-1]))
            faces.append(idx.reshape(-1, 3).astype(np.int64) + base)
    material = gltf["materials"][gltf["meshes"][0]["primitives"][0]["material"]]
    tex = gltf["textures"][material["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
    image = gltf["images"][tex["source"]]
    view = gltf["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    pos = np.concatenate(pos)
    nrm = np.concatenate(nrm)
    if not np.abs(nrm).sum():
        nrm = vertex_normals(pos, np.concatenate(faces))
    return {"pos": pos, "nrm": nrm, "uv": np.concatenate(uv), "faces": np.concatenate(faces),
            "image": binary[start:start + view["byteLength"]], "mime": image.get("mimeType", "image/png")}


def vertex_normals(pos: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n = np.cross(pos[faces[:, 1]] - pos[faces[:, 0]], pos[faces[:, 2]] - pos[faces[:, 0]])
    out = np.zeros_like(pos)
    for k in range(3):
        np.add.at(out, faces[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------------------- camera

class Camera:
    """Pixal3D's front camera plus the photo-paint fit: mesh points <-> artwork pixels."""

    def __init__(self, cam: dict):
        paint = cam.get("paint") or {}
        if paint.get("score", 0) < MIN_MATCH or not paint.get("mesh_moments"):
            raise ValueError("The 3D shape does not match the artwork closely enough to rig it "
                             f"(silhouette match {paint.get('score', 0):.2f}). Try another seed.")
        if paint.get("azimuth", 0) % 360 or paint.get("flipped"):
            raise ValueError("The 3D model came out turned around. Try another seed.")
        self.f = 0.5 / math.tan(math.radians(cam["fov"]) / 2)
        (self.sx, self.sy, ss), (self.dx, self.dy, ds) = paint["mesh_moments"], paint["photo_moments"]
        self.k = ds / ss
        self.x0, self.y0, self.side = cam["box"]

    def to_image(self, p: np.ndarray) -> np.ndarray:
        """[N,3] mesh points -> [N,2] artwork pixels."""
        depth = np.maximum(self.f - p[:, 2], 1e-6)
        x = 0.5 + self.f * p[:, 0] / depth
        y = 0.5 - self.f * p[:, 1] / depth
        px = self.dx + (x - self.sx) * self.k
        py = self.dy + (y - self.sy) * self.k
        return np.stack([self.x0 + px * self.side, self.y0 + py * self.side], 1)

    def ray(self, uv) -> Tuple[np.ndarray, np.ndarray]:
        """Artwork pixel -> (origin, unit direction) in the mesh frame."""
        px, py = (uv[0] - self.x0) / self.side, (uv[1] - self.y0) / self.side
        x, y = self.sx + (px - self.dx) / self.k, self.sy + (py - self.dy) / self.k
        d = np.array([(x - 0.5) / self.f, (0.5 - y) / self.f, -1.0])
        return np.array([0.0, 0.0, self.f]), d / np.linalg.norm(d)


def ray_hits(tri: np.ndarray, o: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Sorted distances at which the ray crosses the triangles [M,3,3] (Möller–Trumbore)."""
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    p = np.cross(d, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
    s = o - tri[:, 0]
    u = (s * p).sum(1) * inv
    q = np.cross(s, e1)
    v = (q @ d) * inv
    t = (q * e2).sum(1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return np.sort(t[hit])


def lift(cam: Camera, tri: np.ndarray, uv) -> Optional[np.ndarray]:
    """3D point in the middle of the first solid the ray through artwork pixel ``uv`` crosses
    (the body's centre line); None when the ray misses."""
    o, d = cam.ray(uv)
    t = ray_hits(tri, o, d)
    t = t[np.concatenate([[True], np.diff(t) > 1e-5])] if len(t) else t   # an edge hit counts twice
    if len(t) >= 2:
        return o + d * (t[0] + t[1]) / 2
    return o + d * t[0] if len(t) else None


def lift_limb(cam: Camera, tri: np.ndarray, uv, radius: float, spread: float) -> Optional[np.ndarray]:
    """A limb joint: DWPose puts it on the limb's outline, where a ray can graze the limb and hit
    a cape behind it, so take the front-most surface among nearby rays, ``radius`` deeper."""
    best = None
    for dx in (-spread, 0, spread):
        for dy in (-spread, 0, spread):
            o, d = cam.ray((uv[0] + dx, uv[1] + dy))
            t = ray_hits(tri, o, d)
            if len(t) and (best is None or (o + d * t[0])[2] > best[2]):
                best = o + d * t[0]
    if best is None:
        return None
    o, d = cam.ray(uv)   # back on the keypoint's own ray, at that depth minus the radius
    return o + d * (o[2] - (best[2] - radius)) / -d[2]


# --------------------------------------------------------------------------------------- joints

def skeleton(kp: np.ndarray, cam: Camera, tri: np.ndarray, pos: np.ndarray) -> Tuple[dict, dict, bool]:
    """Joint positions (mesh frame) for every bone in BONES, limb radii, and whether the picture
    shows the hips (a full body rather than a bust), from DWPose points."""
    from src.core.pose import BODY, FACE0, L_HAND0, R_HAND0

    def pt(name):
        x, y, s = kp[BODY[name]]
        return np.array([x, y]) if s >= CONFIDENT else None

    def mean(*pts):
        pts = [p for p in pts if p is not None]
        return np.mean(pts, 0) if pts else None

    l_sh, r_sh = pt("l_shoulder"), pt("r_shoulder")
    if l_sh is None or r_sh is None:
        raise ValueError("Could not find the character's shoulders in the artwork; a front-facing "
                         "character works best (Gallery: Redraw facing forward).")
    chin = kp[FACE0 + 8, :2] if kp[FACE0 + 8, 2] >= CONFIDENT else mean(pt("nose"), pt("l_ear"), pt("r_ear"))
    if chin is None:
        raise ValueError("Could not find the character's face in the artwork; a front-facing "
                         "character works best (Gallery: Redraw facing forward).")
    mid_sh = (l_sh + r_sh) / 2
    sh_width = np.linalg.norm(l_sh - r_sh)
    # limb radii from anime proportions of the shoulder width (rays can't measure an arm that
    # touches a cape); pixels -> mesh units at the camera's focus
    unit = sh_width / (cam.side * cam.k)
    radii = {k: v * unit for k, v in dict(UpperArm=0.12, LowerArm=0.1, Hand=0.07, HandTip=0.1, UpperLeg=0.2,
                                          LowerLeg=0.15, Foot=0.11, Toes=0.1).items()}
    j: Dict[str, np.ndarray] = {}

    def at(uv, limb=None):
        p = lift_limb(cam, tri, uv, radii[limb], 0.04 * sh_width) if limb else lift(cam, tri, uv)
        if p is None:   # just outside the silhouette: centre depth of the nearest vertices
            img = cam.to_image(pos)
            near = np.argsort(((img - uv) ** 2).sum(1))[:200]
            z = (pos[near, 2].max() + pos[near, 2].min()) / 2
            o, d = cam.ray(uv)
            p = o + d * (o[2] - z) / -d[2]
        return p

    j["neck"] = at(mid_sh + (chin - mid_sh) * 0.25)
    j["head"] = at(mid_sh + (chin - mid_sh) * 0.8)
    hips2 = mean(pt("l_hip"), pt("r_hip"))
    full_body = hips2 is not None
    if hips2 is None:   # bust: hips below the picture, the torso as long as the shoulders are wide
        hips2 = mid_sh + np.array([0, 1.6 * sh_width])
    j["hips"] = at(hips2)
    j["spine"] = j["hips"] + (j["neck"] - j["hips"]) * 0.3
    j["chest"] = j["hips"] + (j["neck"] - j["hips"]) * 0.65
    for side, sh, hand0 in (("left", l_sh, L_HAND0), ("right", r_sh, R_HAND0)):
        s = side[0]
        j[f"{side}UpperArm"] = at(sh, "UpperArm")
        j[f"{side}Shoulder"] = j["neck"] + (j[f"{side}UpperArm"] - j["neck"]) * 0.2 + \
            np.array([0, (j["chest"][1] - j["neck"][1]) * 0.15, 0])
        elbow, wrist = pt(f"{s}_elbow"), pt(f"{s}_wrist")
        radii[f"{side}Arm"] = float(elbow is not None and wrist is not None)   # 0: the arm claims no vertices
        if elbow is None or wrist is None:   # arm hidden: hang it straight down beside the body
            down = np.array([0, sh_width * 0.75])
            elbow, wrist = sh + down, sh + 2 * down
        j[f"{side}LowerArm"] = at(elbow, "LowerArm")
        j[f"{side}Hand"] = at(wrist, "Hand")
        hand = kp[hand0 + 9:hand0 + 13]
        tip = hand[:, :2].mean(0) if hand[:, 2].mean() >= CONFIDENT else wrist + (wrist - elbow) * 0.4
        j[f"{side}HandTip"] = at(tip, "HandTip")
        hip, knee, ankle = pt(f"{s}_hip"), pt(f"{s}_knee"), pt(f"{s}_ankle")
        sign = 1 if np.dot(sh - mid_sh, [1, 0]) > 0 else -1
        if hip is None:
            hip = hips2 + np.array([sign * sh_width * 0.25, 0])
        radii[f"{side}Leg"] = float(knee is not None and ankle is not None)
        if knee is None or ankle is None:   # hidden legs (long skirt, cut-off picture)
            knee, ankle = hip + np.array([0, sh_width * 1.6]), hip + np.array([0, sh_width * 3.2])
        j[f"{side}UpperLeg"] = at(hip)
        j[f"{side}LowerLeg"] = at(knee, "LowerLeg")
        j[f"{side}Foot"] = at(ankle, "Foot")
        toe = mean(pt(f"{s}_big_toe"), pt(f"{s}_small_toe"))
        j[f"{side}Toes"] = at(toe, "Toes") if toe is not None else j[f"{side}Foot"] + np.array([0, -0.5, 1.5]) * radii["Foot"]
    # anime eyes sit a little below the middle of the head: eye line to chin ~45% of its height
    eyes = mean(pt("l_eye"), pt("r_eye"))
    tall = (chin[1] - eyes[1]) / 0.45 if eyes is not None else 1.1 * sh_width
    j["headTop"] = j["head"] + np.array([0, tall / (cam.side * cam.k), 0])
    return j, radii, full_body


# -------------------------------------------------------------------------------------- weights

def _segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Distance from points [N,3] to segment a-b and the clamped parameter t in [0, 1]."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1), t


def chain_weights(p: np.ndarray, points: List[np.ndarray], bones: List[int], blend: float = 0.25,
                  count: int = len(BONES)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weights of a bone chain (bone i spans points[i] -> points[i+1]); each vertex goes to its
    nearest segment, shared with the neighbour within ``blend`` x the shorter segment of a joint
    (so a long head segment does not reach down into the neck). Returns (weights [N,count],
    distance to the chain, index of the nearest segment)."""
    n = len(bones)
    seg = [_segment(p, points[i], points[i + 1]) for i in range(n)]
    d, t = np.stack([s[0] for s in seg], 1), np.stack([s[1] for s in seg], 1)
    lens = np.array([np.linalg.norm(points[i + 1] - points[i]) for i in range(n)])
    half = [blend * min(lens[i - 1], lens[i]) for i in range(1, n)]   # blend half-width at joint i
    near = d.argmin(1)
    rows = np.arange(len(p))
    along = t[rows, near] * lens[near]
    hs = np.array([0.0] + half)[near]          # blend zone at the segment's start
    he = np.array(half + [0.0])[near]          # and at its end
    w = np.zeros((len(p), count))
    own = np.ones(len(p))
    lo = along < hs
    hi = ~lo & (lens[near] - along < he)
    own[lo] = 0.5 + 0.5 * along[lo] / hs[lo]
    own[hi] = 0.5 + 0.5 * (lens[near] - along)[hi] / he[hi]
    w[rows, np.array(bones)[near]] = own
    prev = np.array(bones)[np.maximum(near - 1, 0)]
    nxt = np.array(bones)[np.minimum(near + 1, len(bones) - 1)]
    w[rows[lo], prev[lo]] += 1 - own[lo]
    w[rows[hi], nxt[hi]] += 1 - own[hi]
    return w, d[rows, near], near


def _capsule(dist: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """1 within 1.15x the limb radius, fading to 0 at 1.6x."""
    return np.clip((1.6 * radius - dist) / np.maximum(0.45 * radius, 1e-9), 0, 1)


# Only actual hair should acquire spring chains. Ears, earrings and hats are attached
# to the head too, but are not strands and should not bend like hair below the jaw.
HAIR_LAYERS = ("front hair", "back hair", "side hair")
HEAD_ATTACHMENTS = ("headwear", "ears", "earwear")


def skin_weights(pos: np.ndarray, faces: np.ndarray, j: dict, radii: dict,
                 parts: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Per-vertex weights over BONES [N, len(BONES)] (rows sum to 1) and a mask of the faces to
    keep. Pass welded faces (``weld``) so smoothing crosses UV seams.

    ``parts`` (from spread_parts): which part each vertex belongs to, read from the artwork's
    layers. With it, an arm is exactly the arm's vertices (a cape or thigh next to it is not), a
    side lock hangs from the head, and triangles where Pixal3D fused a hand or forearm to the
    body are cut so the T-pose does not drag the body along. Without it, limbs are capsules."""
    known = parts is not None and (parts != "").any()
    # torso by height along the spine, so the lower face is the head's, not the short neck's
    flat = lambda p: p * np.array([0.0, 1.0, 0.0])
    torso = [flat(j[k]) for k in ("hips", "spine", "chest", "neck", "head", "headTop")]
    w, _, _ = chain_weights(flat(pos), torso, [INDEX[b] for b in ("hips", "spine", "chest", "neck", "head")])
    if known:   # hair and rigid head accessories follow the head below the pivot
        tall = j["headTop"][1] - j["head"][1]
        hair = np.clip((pos[:, 1] - (j["head"][1] - 1.2 * tall)) / (0.8 * tall), 0, 1)
        h = np.where(parts == "head_attachment", 1.0, hair * (parts == "hair"))
        w = w * (1 - h[:, None])
        w[:, INDEX["head"]] += h
    cut = np.zeros(len(faces), bool)
    for side in ("left", "right"):
        # arms: shoulder bone from the neck, then upper arm, forearm, hand
        pts = [j[f"{side}Shoulder"], j[f"{side}UpperArm"], j[f"{side}LowerArm"], j[f"{side}Hand"], j[f"{side}HandTip"]]
        bones = [INDEX[f"{side}{b}"] for b in ("Shoulder", "UpperArm", "LowerArm", "Hand")]
        wa, dist, seg = chain_weights(pos, pts, bones)
        radius = np.array([0, radii["UpperArm"], radii["LowerArm"], radii["HandTip"]])[seg] * radii.get(f"{side}Arm", 1)
        if known:
            inside = ((parts == f"arm-{side[0]}") & (dist < 2.5 * radius) & (seg >= 1)).astype(float)
            # fused to the body away from the shoulder: cut (a contact, not where the arm grows)
            upper = np.linalg.norm(pts[2] - pts[1])
            far = inside.astype(bool) & (np.linalg.norm(pos - pts[1], axis=1) > 0.5 * upper)
            cut |= far[faces].any(1) & ~inside.astype(bool)[faces].all(1)
        else:
            inside = _capsule(dist, radius) * (seg >= 1)
        w = w * (1 - inside[:, None]) + wa * inside[:, None]
        leg_pts = [j["hips"], j[f"{side}UpperLeg"], j[f"{side}LowerLeg"], j[f"{side}Foot"], j[f"{side}Toes"]]
        leg_bones = [INDEX["hips"]] + [INDEX[f"{side}{b}"] for b in ("UpperLeg", "LowerLeg", "Foot")]
        wl, dist, seg = chain_weights(pos, leg_pts, leg_bones)
        radius = np.array([0, radii["UpperLeg"], radii["LowerLeg"], radii["Foot"]])[seg] * radii.get(f"{side}Leg", 1)
        # the other leg's capsule must not take this one: only this side of the body's middle
        side_ok = np.sign(pos[:, 0] - j["hips"][0]) == np.sign(j[f"{side}UpperLeg"][0] - j["hips"][0])
        if known:   # a skirt over the thighs stays with the hips
            inside = ((parts == "leg") & (dist < 2.5 * radius) & (seg >= 1) & side_ok).astype(float)
        else:
            inside = _capsule(dist, radius) * (seg >= 1) * side_ok
        w = w * (1 - inside[:, None]) + wl * inside[:, None]
    return smooth(w, faces[~cut], len(pos)), ~cut


def weld(pos: np.ndarray) -> np.ndarray:
    """For each vertex, the first vertex at the same position. Pixal3D's mesh is split along its
    UV seams (38k open edges on the bard); topology work done on ``weld(pos)[faces]`` sees across
    them, otherwise labels stop and weights differ (cracks) at every seam."""
    _, first, inverse = np.unique(np.round(pos, 6), axis=0, return_index=True, return_inverse=True)
    return first[inverse.ravel()]


def cap_holes(pos, nrm, uv, w, faces, before, canon=None):
    """Close the holes left where skin_weights cut fused contacts: a fan around each opened loop's
    centre, carrying that side's weights, so a raised arm or the body behind it has no gap.
    ``canon`` (from weld) finds the loops across UV seams. Returns (pos, nrm, uv, w, faces) with
    the cap vertices and triangles appended."""
    def directed(f):
        return set(zip(f.ravel().tolist(), f[:, [1, 2, 0]].ravel().tolist()))
    canon = np.arange(len(pos)) if canon is None else canon
    now, old = directed(canon[faces]), directed(canon[before])
    succ: Dict[int, List[int]] = {}
    for a, b in now:   # opened: the face across this edge was cut away
        if (b, a) not in now and (b, a) in old:
            succ.setdefault(a, []).append(b)
    loops, seen = [], set()
    for start in succ:
        if start in seen or len(succ[start]) != 1:
            continue
        loop, v = [start], succ[start][0]
        while v != start and len(succ.get(v, ())) == 1 and v not in seen and len(loop) < 5000:
            loop.append(v)
            v = succ[v][0]
        seen.update(loop)
        if v == start and len(loop) >= 3:
            loops.append(loop)
    if not loops:
        return pos, nrm, uv, w, faces
    n = len(pos)
    centre = np.array([pos[l].mean(0) for l in loops])
    normal = np.array([nrm[l].mean(0) for l in loops])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    weight = np.array([w[l].mean(0) for l in loops])
    fans = [[l[(i + 1) % len(l)], l[i], n + k] for k, l in enumerate(loops) for i in range(len(l))]
    return (np.concatenate([pos, centre]), np.concatenate([nrm, normal]),
            np.concatenate([uv, uv[[l[0] for l in loops]]]),   # one side's colour, not a blend of charts
            np.concatenate([w, weight / weight.sum(1, keepdims=True)]), np.concatenate([faces, fans]))


PART_OF = {"handwear-l": "arm-l", "handwear-r": "arm-r", "legwear": "leg", "footwear": "leg",
           **{name: "hair" for name in HAIR_LAYERS},
           **{name: "head_attachment" for name in HEAD_ATTACHMENTS}}


def spread_parts(faces: np.ndarray, pos: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Give every vertex the part of the nearest labelled front-surface vertex, measured over
    the mesh surface (multi-source Dijkstra): the back of an arm is still the arm."""
    import heapq

    n = len(pos)
    part = np.array([PART_OF.get(l, "body") if l else "" for l in labels], object)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, 1), axis=0)
    edges = np.concatenate([edges, edges[:, ::-1]])
    edges = edges[np.argsort(edges[:, 0], kind="stable")]
    length = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
    start = np.searchsorted(edges[:, 0], np.arange(n + 1))
    nbr, length = edges[:, 1].tolist(), length.tolist()
    dist = [math.inf] * n
    heap = [(0.0, v) for v in np.nonzero(part != "")[0].tolist()]
    for _, v in heap:
        dist[v] = 0.0
    out = part.tolist()
    while heap:
        d, v = heapq.heappop(heap)
        if d > dist[v]:
            continue
        for k in range(start[v], start[v + 1]):
            u, nd = nbr[k], d + length[k]
            if nd < dist[u]:
                dist[u] = nd
                out[u] = out[v]
                heapq.heappush(heap, (nd, u))
    return np.array(out, object)


def front_labels(model_dir: Path, cam: "Camera", pos: np.ndarray, tolerance: float) -> np.ndarray:
    """For vertices on the surface the artwork shows (within ``tolerance`` of the front-most
    one at their pixel): the name of the top See-through layer there, without -l/-r for
    everything but the hands. "" for hidden vertices or characters without layers.
    (``[0-9][0-9]_*`` skips expression layers, which live outside assets/ anyway.)"""
    out = np.full(len(pos), "", object)
    files = sorted((model_dir / "assets").glob("[0-9][0-9]_*.png"))
    if not files:
        return out
    top = None
    names = []
    for k, f in enumerate(files):   # files are numbered back to front
        alpha = np.asarray(Image.open(f).convert("RGBA"))[..., 3] > 128
        top = np.where(alpha, k, -1 if top is None else top)
        name = f.stem[3:]
        names.append(name if name.startswith("handwear") else name.rsplit("-", 1)[0] if name[-2:] in ("-l", "-r") else name)
    img = cam.to_image(pos)
    h, w = top.shape
    # A mesh can extend beyond a cropped image. Clamping those projections to the
    # border used to give them the border's label and could even make them occlude
    # valid front vertices. Ignore such samples altogether while retaining safe indices.
    valid = (np.isfinite(img).all(axis=1) & np.isfinite(pos[:, 2]) &
             (img[:, 0] >= 0) & (img[:, 0] < w) & (img[:, 1] >= 0) & (img[:, 1] < h))
    safe = np.where(valid[:, None], img, 0)
    x, y = safe[:, 0].astype(int), safe[:, 1].astype(int)
    # front-most depth per cell of a coarse grid (vertex splats), widened over neighbours
    cell = max(2, int(round(cam.side / 300)))
    gx, gy = x // cell, y // cell
    zbuf = np.full((h // cell + 3, w // cell + 3), -np.inf)
    np.maximum.at(zbuf, (gy[valid] + 1, gx[valid] + 1), pos[valid, 2])
    wide = np.max([np.roll(np.roll(zbuf, a, 0), b, 1) for a in (-1, 0, 1) for b in (-1, 0, 1)], 0)
    seen = valid & (pos[:, 2] >= wide[gy + 1, gx + 1] - tolerance)
    layer = top[y, x]
    ok = seen & (layer >= 0)
    out[ok] = np.array(names, object)[layer[ok]]
    return out


def smooth(w: np.ndarray, faces: np.ndarray, n: int, iterations: int = 6) -> np.ndarray:
    """Average weights with mesh neighbours (hides capsule borders), keep the 4 largest."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])
    degree = np.bincount(edges[:, 0], minlength=n)[:, None].astype(float)
    for _ in range(iterations):
        acc = np.zeros_like(w)
        np.add.at(acc, edges[:, 0], w[edges[:, 1]])
        w = 0.5 * w + 0.5 * np.where(degree > 0, acc / np.maximum(degree, 1), w)
    top = np.argsort(-w, 1)[:, :4]
    keep = np.zeros_like(w)
    rows = np.arange(len(w))[:, None]
    keep[rows, top] = w[rows, top]
    return keep / np.maximum(keep.sum(1, keepdims=True), 1e-12)


# ---------------------------------------------------------------------------------- hair springs

def hair_springs(pos: np.ndarray, parts: Optional[np.ndarray], w: np.ndarray, j: dict,
                 sectors: int = 8, links: int = 3) -> Tuple[np.ndarray, dict]:
    """Spring-bone chains for hair hanging below the jaw, one per direction around the head
    that has some (side locks, a braid, long back hair), so VRM apps swing it with physics.
    Returns weights with a column added per chain bone, and the springs for write_vrm (joint
    positions under "joints"; colliders are added once the model is in metres)."""
    springs = {"bones": [], "roots": [], "colliders": [], "joints": {}}
    if parts is None:
        return w, springs
    tall = j["headTop"][1] - j["head"][1]
    top = j["head"][1] + 0.15 * tall   # chains hang from just above the jaw
    hang = (parts == "hair") & (pos[:, 1] < top)
    angle = np.arctan2(pos[:, 0] - j["head"][0], pos[:, 2] - j["head"][2])
    sector = ((angle + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
    chains = []
    for s in range(sectors):
        sel = hang & (sector == s)
        if sel.sum() < 30 or top - pos[sel, 1].min() < 0.4 * tall:
            continue   # nothing there, or short hair the head carries rigidly
        bottom = pos[sel, 1].min()
        idx = np.nonzero(sel)[0]
        pts = []
        for y in np.linspace(top, bottom, links + 1):   # the strand's middle at each height
            near = idx[np.argsort(np.abs(pos[idx, 1] - y))[:30]]   # (hair may start below the jaw)
            p = pos[near].mean(0)
            pts.append(np.array([p[0], y, p[2]]))
        chains.append((s, sel, pts))
    if not chains:
        return w, springs
    w = np.concatenate([w, np.zeros((len(w), len(chains) * links))], 1)
    for c, (s, sel, pts) in enumerate(chains):
        names = [f"hair{s}_{k}" for k in range(links)]
        cols = [len(BONES) + c * links + k for k in range(links)]
        springs["bones"] += [(n, "head" if k == 0 else names[k - 1]) for k, n in enumerate(names)]
        springs["roots"].append(names[0])
        springs["joints"].update({n: pts[k] for k, n in enumerate(names)})
        cw, _, _ = chain_weights(pos[sel], pts, cols, count=w.shape[1])
        # rigid with the head at the root, fully on the chain one link lower
        ramp = np.clip((top - pos[sel, 1]) / ((top - pts[-1][1]) / links), 0, 1)[:, None]
        w[sel] = w[sel] * (1 - ramp) + cw * ramp
    return w, springs


def bust_springs(pos: np.ndarray, parts: Optional[np.ndarray], w: np.ndarray, j: dict,
                 springs: dict) -> np.ndarray:
    """A short spring bone under each side of the chest's front (root inside the ribs, tip at the
    surface) so VRM apps bounce the chest with body motion. Adds to ``springs`` (its own
    "bust_roots" group) and returns the weights with the new columns."""
    sw = abs(j["leftUpperArm"][0] - j["rightUpperArm"][0])
    y = j["neck"][1] - 0.36 * (j["neck"][1] - j["hips"][1])   # about the chest joint's height
    mid, back = j["chest"][0], j["chest"][2]   # the chest joint sits mid-depth; the front is +Z
    torso = w[:, INDEX["chest"]] + w[:, INDEX["spine"]]   # not arms, neck, hair
    if parts is not None and (parts != "").any():
        torso = torso * (parts == "body")
    if not (torso > 0.5).any() or pos[torso > 0.5, 1].min() > y - 0.16 * sw:
        return w   # the mesh ends at the chest (a bust): a bouncing cut edge looks broken
    added = []
    for side, sign in (("L", 1.0), ("R", -1.0)):   # Pixal3D frame: the character's left is +X
        cx = mid + sign * 0.22 * sw
        d2 = ((pos[:, 0] - cx) ** 2 + (pos[:, 1] - y) ** 2) / (0.16 * sw) ** 2
        near = (d2 < 0.5) & (torso > 0.5) & (pos[:, 2] > back)
        if near.sum() < 20:
            continue   # no torso front there (cropped, or covered by something else)
        front = np.percentile(pos[near, 2], 95)
        depth = max(front - back, 1e-6)
        g = 0.85 * np.exp(-d2) * torso * np.clip((pos[:, 2] - back) / (0.6 * depth), 0, 1)
        g[g < 0.02] = 0
        added.append((side, g, np.array([cx, y, back + 0.3 * depth]), np.array([cx, y, front])))
    if not added:
        return w
    w = np.concatenate([w, np.zeros((len(w), 2 * len(added)))], 1)
    for k, (side, g, root, tip) in enumerate(added):
        col = w.shape[1] - 2 * len(added) + 2 * k
        w *= (1 - g)[:, None]
        w[:, col] += g
        springs["bones"] += [(f"bust{side}", "chest"), (f"bust{side}_tip", f"bust{side}")]
        springs.setdefault("bust_roots", []).append(f"bust{side}")
        springs["joints"].update({f"bust{side}": root, f"bust{side}_tip": tip})
    return w


# --------------------------------------------------------------------------------------- T-pose

def _rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking direction a onto direction b."""
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    if an < 1e-9 or bn < 1e-9:
        raise ValueError("Could not determine an arm direction from the pose; a front-facing character with visible arms works best.")
    a, b = a / an, b / bn
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        # Opposite directions need a half-turn, not the identity. Pick an axis
        # perpendicular to the arm, favouring the least-aligned coordinate axis.
        axis = np.eye(3)[np.argmin(np.abs(a))]
        axis = np.cross(a, axis)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / (v @ v))


def t_pose(pos: np.ndarray, nrm: np.ndarray, w: np.ndarray, j: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Raise both arms to horizontal (straight elbows) by linear blend skinning."""
    j = dict(j)
    mats = {i: (np.eye(3), np.zeros(3)) for i in range(w.shape[1])}   # x -> R x + t per bone (hair too)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        s, e, wr = j[f"{side}UpperArm"], j[f"{side}LowerArm"], j[f"{side}Hand"]
        out = np.array([sign, 0, 0])   # Pixal3D frame: the character faces +Z, its left is +X
        r1 = _rotation(e - s, out)
        e2 = s + r1 @ (e - s)
        r2 = _rotation(r1 @ (wr - e), out) @ r1
        t1, t2 = s - r1 @ s, e2 - r2 @ e
        mats[INDEX[f"{side}UpperArm"]] = (r1, t1)
        mats[INDEX[f"{side}LowerArm"]] = mats[INDEX[f"{side}Hand"]] = (r2, t2)
        j[f"{side}LowerArm"] = e2
        j[f"{side}Hand"] = r2 @ wr + t2
        j[f"{side}HandTip"] = r2 @ j[f"{side}HandTip"] + t2
    new_pos = np.zeros_like(pos)
    new_nrm = np.zeros_like(nrm)
    for i, (r, t) in mats.items():
        if w[:, i].any():
            new_pos += w[:, i:i + 1] * (pos @ r.T + t)
            new_nrm += w[:, i:i + 1] * (nrm @ r.T)
    return new_pos, new_nrm / np.maximum(np.linalg.norm(new_nrm, axis=1, keepdims=True), 1e-12), j


# ---------------------------------------------------------------------------------- expressions

def cover_open_eyes(model_dir: Path, layer: Image.Image) -> Image.Image:
    """Painted closed eyes are feathered for the 2D rig, which hides the open-eye layers under
    them. On the mesh the open eyes are baked into the texture and showed through, so over the
    open eyes the decal becomes opaque: the closed eyes on the face layer's inpainted skin, and
    the artwork itself where bangs cross the eyes (hair stays hair)."""
    files = {f.stem[3:]: f for f in (model_dir / "assets").glob("[0-9][0-9]_*.png")}
    eyes = [f for n, f in files.items() if n.rsplit("-", 1)[0] in ("eyewhite", "irides", "eyelash")]
    if not eyes or "face" not in files:
        return layer
    alpha = lambda f: np.asarray(Image.open(f).convert("RGBA"))[..., 3]
    mask = np.zeros(layer.size[::-1], bool)
    for f in eyes:
        mask |= alpha(f) > 8
    cover = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5))) > 0
    painted = np.asarray(Image.alpha_composite(Image.open(files["face"]).convert("RGBA"), layer)).copy()
    if "front hair" in files and (model_dir / "base_image.png").exists():
        hair = alpha(files["front hair"]) >= 128
        painted[hair, :3] = np.asarray(Image.open(model_dir / "base_image.png").convert("RGB"))[hair]
    out = np.asarray(layer).copy()
    out[cover, :3] = painted[cover, :3]
    out[cover, 3] = 255
    return Image.fromarray(out)


def decals(model_dir: Path, cam: Camera, pos: np.ndarray, nrm: np.ndarray, faces: np.ndarray,
           kp: np.ndarray) -> List[dict]:
    """Expression decals: [{name, image (RGBA crop), faces (into pos), uv, shown}] where ``shown``
    means the decal is visible in the neutral face (the artwork's mouth is open, so the closed
    mouth is shown until "A")."""
    from src.core import expressions
    from src.core.pose import BODY

    made = expressions.read(model_dir)
    wanted = []
    if made.get("eye_close"):
        layer = cover_open_eyes(model_dir, Image.open(model_dir / made["eye_close"]).convert("RGBA"))
        a = np.asarray(layer).copy()
        right = a.copy()
        eyes = kp[[BODY["l_eye"], BODY["r_eye"]]]
        if eyes[:, 2].min() >= CONFIDENT:   # split at the middle between the eyes: blink left / right
            cut, left_is_right_of_cut = eyes[:, 0].mean(), eyes[0, 0] > eyes[:, 0].mean()
        else:   # no confident eyes: the middle of the painted closed eyes, a front view's left on the right
            xs = np.nonzero(a[..., 3].max(0) > 8)[0]
            cut, left_is_right_of_cut = ((xs.min() + xs.max()) / 2 if len(xs) else a.shape[1] / 2), True
        left_side = np.arange(a.shape[1]) > cut if left_is_right_of_cut else np.arange(a.shape[1]) < cut
        a[:, ~left_side, 3] = 0
        right[:, left_side, 3] = 0
        wanted += [("blink_l", Image.fromarray(a), False), ("blink_r", Image.fromarray(right), False)]
    visemes = {v: p for v, p in (made.get("visemes") or {}).items() if (model_dir / p).exists()}
    if visemes.get("a"):
        # An originally open mouth needs its painted closed mouth as the neutral face.
        # Each vowel hides that neutral decal in the VRM blend-shape groups below.
        if made.get("mouth_close") and (model_dir / made["mouth_close"]).exists():
            wanted.append(("_mouth_neutral", Image.open(model_dir / made["mouth_close"]).convert("RGBA"), True))
        wanted.append(("a", Image.open(model_dir / visemes["a"]).convert("RGBA"), False))
    elif made.get("mouth_open"):
        wanted.append(("a", Image.open(model_dir / made["mouth_open"]).convert("RGBA"), False))
    elif made.get("mouth_close"):
        wanted.append(("a", Image.open(model_dir / made["mouth_close"]).convert("RGBA"), True))
    for vowel in ("i", "u", "e", "o"):
        if visemes.get(vowel):
            wanted.append((vowel, Image.open(model_dir / visemes[vowel]).convert("RGBA"), False))
    # painted toggles that sit on the face (blush, tears, heart eyes, > <, fang...) become extra
    # expressions; symbols floating around the head have no surface to sit on
    for toggle, path in (made.get("toggles") or {}).items():
        if expressions.TOGGLES.get(toggle) and expressions.TOGGLES[toggle].region in ON_FACE and (model_dir / path).exists():
            image = Image.open(model_dir / path).convert("RGBA")
            if expressions.TOGGLES[toggle].flood:   # > < and spiral eyes replace the whole eye too
                image = cover_open_eyes(model_dir, image)
            wanted.append((f"toggle:{toggle}", image, False))
    have, layers = {n for n, _, _ in wanted}, None
    for toggle in DRAWN_MARKS:
        if f"toggle:{toggle}" in have:
            continue
        layers = expressions.load_layers(model_dir / "assets") if layers is None else layers
        region = expressions.TOGGLES[toggle].region
        mask = expressions.toggle_mask(layers, region) if layers else None
        if mask is not None and region == "gloom" and "front hair" in layers:
            mask &= ~(layers["front hair"] > 128)   # in 2D it sits under the bangs; on a mesh they are in front
        if mask is not None and mask.sum() > 30:
            image = (expressions.draw_blush(mask, expressions.eye_boxes(layers)) if toggle == "blush"
                     else expressions.draw_symbol(region, mask, toggle))
            wanted.append((f"toggle:{toggle}", image, False))

    img = cam.to_image(pos)
    eye_pos = np.array([0, 0, cam.f])
    # Facing by each triangle's own normal: Pixal3D's vertex normals are noisy around the eyes.
    t = pos[faces]
    facing = (np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]) * (eye_pos - t.mean(1))).sum(1) > 0
    # visible = within a hair of the front-most surface at that pixel (bangs over an eye hide it)
    grid = np.full((512, 512), -np.inf)
    g = np.clip(((img - [cam.x0, cam.y0]) / cam.side * 512).astype(int), 0, 511)
    np.maximum.at(grid, (g[:, 1], g[:, 0]), pos[:, 2])
    from_front = pos[:, 2] >= grid[g[:, 1], g[:, 0]] - 0.01
    # Two of three corners: triangles are large, and one corner touching a bang strand that
    # crosses the eye used to drop the whole triangle, leaving the open iris showing through.
    visible = facing & (from_front[faces].sum(1) >= 2)
    out = []
    for name, layer, shown in wanted:
        alpha = np.asarray(layer)[..., 3]
        ys, xs = np.nonzero(alpha > 8)
        if not len(xs):
            continue
        x1, x2, y1, y2 = xs.min() - 3, xs.max() + 4, ys.min() - 3, ys.max() + 4
        inside = (img[:, 0] >= x1) & (img[:, 0] <= x2) & (img[:, 1] >= y1) & (img[:, 1] <= y2)
        tri = faces[inside[faces].all(1) & visible]
        if not len(tri):
            continue
        crop = layer.crop((x1, y1, x2, y2))
        uv = (img - [x1, y1]) / [x2 - x1, y2 - y1]
        toggle = expressions.TOGGLES.get(name[7:]) if name.startswith("toggle:") else None
        label = toggle.label if toggle else name.upper()
        out.append({"name": name, "label": label, "image": crop, "faces": tri, "uv": uv, "shown": shown,
                    "mouth_toggle": bool(toggle and toggle.region == "mouth")})
    return out


# ---------------------------------------------------------------------------------------- build

def build(model_dir: Path, mesh_dir: Path, name: str = "Ka-myii character") -> dict:
    """Rig ``mesh_dir/mesh.glb`` (with its camera.json) into ``mesh_dir/avatar.vrm``. Returns a
    report: joints found, expression shapes, triangle count."""
    from src.core.outpaint import character_mask
    from src.core.pose import keypoints

    cam = Camera(json.loads((mesh_dir / "camera.json").read_text()))
    mesh = load_mesh(mesh_dir / "mesh.glb")
    pos, nrm, faces = mesh["pos"], mesh["nrm"], mesh["faces"]
    art = Image.open(model_dir / "base_image.png").convert("RGB")
    kp = keypoints(art, character_mask(art))
    if kp is None:
        raise RuntimeError("The pose model is not installed. Run: python scripts/install_3d.py")
    tri = pos[faces]
    j, radii, full_body = skeleton(kp, cam, tri, pos)
    canon = weld(pos)
    welded = canon[faces]
    labels = front_labels(model_dir, cam, pos, 0.3 * radii["UpperArm"])
    shown = labels != ""
    labels[canon[shown]] = labels[shown]   # a label on any copy of a seam vertex labels them all
    parts = spread_parts(welded, pos, labels)[canon]
    w, keep = skin_weights(pos, welded, j, radii, parts)
    w = w[canon]
    before, faces = faces, faces[keep]
    try:
        w, springs = hair_springs(pos, parts, w, j)
    except Exception:   # optional swing: an avatar without it beats no avatar
        logger.warning("Hair springs failed; the hair stays rigid", exc_info=True)
        springs = {"bones": [], "roots": [], "colliders": [], "joints": {}}
    try:
        w = bust_springs(pos, parts, w, j, springs)
    except Exception:
        logger.warning("Chest springs failed; the chest stays rigid", exc_info=True)
    pos, nrm, uv, w, faces = cap_holes(pos, nrm, mesh["uv"], w, faces, before, canon)
    expr = decals(model_dir, cam, pos, nrm, faces, kp)
    pos, nrm, j = t_pose(pos, nrm, w, j)
    j.update(springs.pop("joints"))

    # metres, feet on the floor (a bust: chin at a standing height), facing -Z as VRM 0.x does
    chin = j["head"]
    if full_body:
        scale = 1.3 / max(chin[1] - pos[:, 1].min(), 1e-6)
        lift_y = -pos[:, 1].min() * scale
    else:
        scale = 0.24 / max(j["headTop"][1] - j["head"][1], 1e-6)   # an anime head is ~24 cm
        lift_y = 1.3 - chin[1] * scale
    centre = np.array([j["hips"][0], 0, j["hips"][2]])

    def place(p):
        q = (p - centre) * scale
        return np.stack([-q[..., 0], q[..., 1] + lift_y, -q[..., 2]], -1)

    pos = place(pos)
    nrm = nrm * np.array([-1, 1, -1])
    joints = {k: place(v) for k, v in j.items()}
    # colliders keep swinging hair out of the head, neck and chest (offsets/radii in metres)
    tall = joints["headTop"][1] - joints["head"][1]
    width = np.linalg.norm(joints["leftUpperArm"] - joints["rightUpperArm"])
    springs["colliders"] = [("head", 0.45 * tall, 0.38 * tall), ("neck", 0.0, 0.25 * width),
                            ("chest", 0.0, 0.42 * width)]
    out = mesh_dir / "avatar.vrm"
    write_vrm(out, pos, nrm, uv, faces, w, joints, expr, mesh["image"], mesh["mime"], name,
              Image.open(mesh_dir / "crop.png"), springs)
    report = {"path": str(out), "rig_version": RIG_VERSION, "triangles": int(len(faces)), "full_body": full_body,
              "expressions": [e["name"] for e in expr], "hair_chains": len(springs["roots"]),
              "bust_springs": len(springs.get("bust_roots", [])),
              "height_m": round(float(pos[:, 1].max() - pos[:, 1].min()), 3)}
    (mesh_dir / "rig.json").write_text(json.dumps(report, indent=2))
    return report


def demo(model_dir: Path, out_dir: Path, name: str = "Ka-myii character") -> dict:
    """Demo mode (no GPU): a box mannequin in the VRM T-pose wearing the artwork on its front,
    written by the same VRM writer, so the Studio, Gallery and 3D stage can be exercised."""
    out_dir.mkdir(parents=True, exist_ok=True)
    j = {"hips": (0, .85, 0), "spine": (0, .95, 0), "chest": (0, 1.1, 0), "neck": (0, 1.3, 0), "head": (0, 1.36, 0),
         "headTop": (0, 1.62, 0)}
    for side, x in (("left", -1), ("right", 1)):   # VRM 0.x faces -Z: the character's left is -X
        j.update({f"{side}Shoulder": (x * .03, 1.27, 0), f"{side}UpperArm": (x * .13, 1.27, 0),
                  f"{side}LowerArm": (x * .37, 1.27, 0), f"{side}Hand": (x * .6, 1.27, 0),
                  f"{side}HandTip": (x * .7, 1.27, 0), f"{side}UpperLeg": (x * .09, .82, 0),
                  f"{side}LowerLeg": (x * .09, .46, 0), f"{side}Foot": (x * .09, .08, 0),
                  f"{side}Toes": (x * .09, .02, -.1)})
    j = {k: np.array(v, float) for k, v in j.items()}
    ends = {"hips": "spine", "spine": "chest", "chest": "neck", "neck": "head", "head": "headTop"}
    for side in ("left", "right"):
        ends.update({f"{side}{a}": f"{side}{b}" for a, b in (("Shoulder", "UpperArm"), ("UpperArm", "LowerArm"),
                     ("LowerArm", "Hand"), ("Hand", "HandTip"), ("UpperLeg", "LowerLeg"), ("LowerLeg", "Foot"),
                     ("Foot", "Toes"))})
    pos, faces, bone = [], [], []
    quad = np.array([[0, 1, 2], [0, 2, 3]])
    for b, e in ends.items():
        a, c = j[b], j[e]
        axis = (c - a) / np.linalg.norm(c - a)
        u = np.cross(axis, [0, 0, 1] if abs(axis[2]) < .9 else [1, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        t = .1 if b == "head" else .07 if b in ("hips", "spine", "chest") else .035
        ring = [s * u * t + r * v * t for s, r in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        box = np.array([a + o for o in ring] + [c + o for o in ring])
        sides = [[k, (k + 1) % 4, 4 + (k + 1) % 4, 4 + k] for k in range(4)] + [[3, 2, 1, 0], [4, 5, 6, 7]]
        tri = np.concatenate([np.array(q)[quad] for q in sides])
        normal = np.cross(box[tri[:, 1]] - box[tri[:, 0]], box[tri[:, 2]] - box[tri[:, 0]])
        inward = (normal * (box[tri].mean(1) - box.mean(0))).sum(1) < 0   # wind every face outwards
        tri[inward] = tri[inward][:, ::-1]
        faces.append(len(pos) + tri)
        pos += list(box)
        bone += [INDEX[b]] * 8
    pos, faces = np.array(pos), np.concatenate(faces)
    w = np.zeros((len(pos), len(BONES)))
    w[np.arange(len(pos)), bone] = 1
    lo, hi = pos.min(0), pos.max(0)
    uv = np.stack([(hi[0] - pos[:, 0]) / (hi[0] - lo[0]), (hi[1] - pos[:, 1]) / (hi[1] - lo[1])], 1)
    art = Image.open(model_dir / "base_image.png").convert("RGB")
    write_vrm(out_dir / "avatar.vrm", pos, vertex_normals(pos, faces), uv, faces, w, j, [], _png(art),
              "image/png", name, art)
    report = {"triangles": int(len(faces)), "full_body": True, "expressions": [], "height_m": 1.62, "demo": True}
    (out_dir / "rig.json").write_text(json.dumps(report, indent=2))
    return report


# ------------------------------------------------------------------------------------ VRM write

class _Writer:
    def __init__(self):
        self.gltf = {"asset": {"version": "2.0", "generator": "Ka-myii"}, "buffers": [], "bufferViews": [],
                     "accessors": []}
        self.bin = bytearray()

    def view(self, data: bytes, target: Optional[int] = None) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            v["target"] = target
        self.bin += data
        self.gltf["bufferViews"].append(v)
        return len(self.gltf["bufferViews"]) - 1

    def add(self, arr: np.ndarray, kind: str, target: Optional[int] = None, bounds: bool = False) -> int:
        ctype = {np.float32: 5126, np.uint32: 5125, np.uint16: 5123, np.uint8: 5121}[arr.dtype.type]
        a = {"bufferView": self.view(arr.tobytes(), target), "componentType": ctype,
             "count": int(len(arr)), "type": kind}
        if bounds:
            a["min"] = arr.min(0).tolist()
            a["max"] = arr.max(0).tolist()
        self.gltf["accessors"].append(a)
        return len(self.gltf["accessors"]) - 1

    def image(self, data: bytes, mime: str) -> int:
        self.gltf.setdefault("images", []).append({"bufferView": self.view(data), "mimeType": mime})
        self.gltf.setdefault("samplers", [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}])
        self.gltf.setdefault("textures", []).append({"sampler": 0, "source": len(self.gltf["images"]) - 1})
        return len(self.gltf["textures"]) - 1

    def save(self, path: Path):
        while len(self.bin) % 4:
            self.bin.append(0)
        self.gltf["buffers"] = [{"byteLength": len(self.bin)}]
        js = json.dumps(self.gltf, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 4)
        body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(self.bin), 0x004E4942) + self.bin
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)
        tmp.replace(path)


def _png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "PNG", optimize=True)
    return buf.getvalue()


OUTLINE_CM = 0.06   # MToon world-space outline width; 0 turns the anime line art off


def _mtoon(name: str, texture: int, transparent: bool) -> dict:
    """VRM 0.x MToon properties: lit colour from the texture, a soft toon shade and, on the opaque
    body, a thin dark outline like the line art of the drawing (decals get none)."""
    outline = 0 if transparent else OUTLINE_CM
    return {
        "name": name, "shader": "VRM/MToon", "renderQueue": 3000 if transparent else 2000,
        "floatProperties": {"_Cutoff": 0.5, "_BumpScale": 1, "_ShadeShift": -0.1, "_ShadeToony": 0.9,
                            "_ReceiveShadowRate": 1, "_ShadingGradeRate": 1, "_LightColorAttenuation": 0,
                            "_IndirectLightIntensity": 0.1, "_RimLightingMix": 0, "_RimFresnelPower": 1,
                            "_RimLift": 0, "_OutlineWidth": outline, "_OutlineScaledMaxDistance": 1,
                            "_OutlineLightingMix": 1, "_DebugMode": 0, "_BlendMode": 2 if transparent else 0,
                            "_OutlineWidthMode": 1 if outline else 0, "_OutlineColorMode": 0, "_CullMode": 2,
                            "_OutlineCullMode": 1, "_SrcBlend": 5 if transparent else 1,
                            "_DstBlend": 10 if transparent else 0, "_ZWrite": 0 if transparent else 1,
                            "_MToonVersion": 38},
        "vectorProperties": {"_Color": [1, 1, 1, 1], "_ShadeColor": [0.82, 0.78, 0.8, 1],
                             "_MainTex": [0, 0, 1, 1], "_ShadeTexture": [0, 0, 1, 1], "_RimColor": [0, 0, 0, 1],
                             "_EmissionColor": [0, 0, 0, 1], "_OutlineColor": [0.22, 0.16, 0.18, 1]},
        "textureProperties": {"_MainTex": texture, "_ShadeTexture": texture},
        "keywordMap": {"_ALPHABLEND_ON": True} if transparent else {"MTOON_OUTLINE_WIDTH_WORLD": True,
                                                                     "MTOON_OUTLINE_COLOR_FIXED": True},
        "tagMap": {"RenderType": "Transparent" if transparent else "Opaque"},
    }


def write_vrm(path: Path, pos, nrm, uv, faces, w, joints: dict, expr: List[dict], image: bytes, mime: str,
              name: str, thumbnail: Image.Image, springs: Optional[dict] = None):
    """``w`` has a column per BONES entry, then per extra (spring) bone in ``springs["bones"]``."""
    springs = springs or {"bones": [], "roots": [], "colliders": []}
    bones = BONES + springs["bones"]
    index = {b: i for i, (b, _) in enumerate(bones)}
    W = _Writer()
    g = W.gltf
    # skeleton nodes: local translation = joint - parent joint (VRM 0.x rest pose: no rotations)
    nodes = [{"name": "Root"}]
    for bone, parent in bones:
        p = joints[bone] - (joints[parent] if parent else 0)
        nodes.append({"name": bone, "translation": [float(x) for x in p]})
    for i, (bone, parent) in enumerate(bones):
        if parent:
            nodes[index[parent] + 1].setdefault("children", []).append(i + 1)
    g["nodes"] = nodes
    g["scenes"] = [{"nodes": [0]}]
    g["scene"] = 0

    joint_ids = np.argsort(-w, 1)[:, :4].astype(np.uint16)
    weights = np.take_along_axis(w, joint_ids.astype(np.int64), 1).astype(np.float32)
    weights /= np.maximum(weights.sum(1, keepdims=True), 1e-12)
    joint_ids[weights == 0] = 0   # glTF validators flag unused slots that name a joint
    inverse = np.zeros((len(bones), 16), np.float32)
    for i, (bone, _) in enumerate(bones):
        m = np.eye(4, dtype=np.float32)
        m[:3, 3] = -joints[bone]
        inverse[i] = m.T.reshape(-1)   # column-major
    g["skins"] = [{"joints": list(range(1, len(bones) + 1)), "skeleton": 1,
                   "inverseBindMatrices": W.add(inverse, "MAT4")}]

    body_tex = W.image(image, mime)
    attrs = {"POSITION": W.add(pos.astype(np.float32), "VEC3", 34962, True),
             "NORMAL": W.add(nrm.astype(np.float32), "VEC3", 34962),
             "TEXCOORD_0": W.add(uv.astype(np.float32), "VEC2", 34962),
             "JOINTS_0": W.add(joint_ids, "VEC4", 34962),
             "WEIGHTS_0": W.add(weights, "VEC4", 34962)}
    g["meshes"] = [{"name": "Body", "primitives": [
        {"attributes": attrs, "indices": W.add(faces.astype(np.uint32).reshape(-1), "SCALAR", 34963), "material": 0}]}]
    g["materials"] = [{"name": "Body", "pbrMetallicRoughness": {"baseColorTexture": {"index": body_tex},
                                                                "metallicFactor": 0, "roughnessFactor": 1}}]
    mtoon = [_mtoon("Body", body_tex, False)]
    groups = []
    face_mesh = None
    if expr:
        # one primitive (glTF wants every primitive of a mesh to share its morph targets), one atlas
        widths = [e["image"].width for e in expr]
        atlas = Image.new("RGBA", (sum(widths) + 4 * len(expr), max(e["image"].height for e in expr)))
        vp, vn, vu, vf, vw, vj, offsets = [], [], [], [], [], [], []
        x = 0
        base = 0
        for k, e in enumerate(expr):
            gap = DECAL_GAP * (1 + 0.25 * k)
            atlas.paste(e["image"], (x, 0))
            used = np.unique(e["faces"])
            remap = np.zeros(len(pos), np.int64)
            remap[used] = np.arange(len(used)) + base
            u = e["uv"][used]
            vu.append(np.stack([(x + u[:, 0] * e["image"].width) / atlas.width,
                                u[:, 1] * e["image"].height / atlas.height], 1))
            sign = 1 if e["shown"] else -1
            vp.append(pos[used] + sign * gap * nrm[used])
            vn.append(nrm[used])
            vf.append(remap[e["faces"]])
            vj.append(joint_ids[used])
            vw.append(weights[used])
            offsets.append((base, len(used), -2 * sign * gap * nrm[used]))
            base += len(used)
            x += e["image"].width + 4
        total = base
        targets = []
        for start, count, delta in offsets:
            d = np.zeros((total, 3), np.float32)
            d[start:start + count] = delta
            targets.append({"POSITION": W.add(d, "VEC3", 34962, True)})
        face_tex = W.image(_png(atlas), "image/png")
        g["materials"].append({"name": "Expressions", "alphaMode": "BLEND", "pbrMetallicRoughness": {
            "baseColorTexture": {"index": face_tex}, "metallicFactor": 0, "roughnessFactor": 1}})
        mtoon.append(_mtoon("Expressions", face_tex, True))
        g["meshes"].append({"name": "Face", "primitives": [{
            "attributes": {"POSITION": W.add(np.concatenate(vp).astype(np.float32), "VEC3", 34962, True),
                           "NORMAL": W.add(np.concatenate(vn).astype(np.float32), "VEC3", 34962),
                           "TEXCOORD_0": W.add(np.concatenate(vu).astype(np.float32), "VEC2", 34962),
                           "JOINTS_0": W.add(np.concatenate(vj), "VEC4", 34962),
                           "WEIGHTS_0": W.add(np.concatenate(vw), "VEC4", 34962)},
            "indices": W.add(np.concatenate(vf).astype(np.uint32).reshape(-1), "SCALAR", 34963),
            "material": 1, "targets": targets}],
            "extras": {"targetNames": [e["name"] for e in expr]}})
        face_mesh = 1
        names = [e["name"] for e in expr]
        presets = {"blink_l": "Blink_L", "blink_r": "Blink_R", "a": "A", "i": "I", "u": "U", "e": "E", "o": "O"}
        neutral_mouth = names.index("_mouth_neutral") if "_mouth_neutral" in names else next(
            (k for k, e in enumerate(expr) if e["name"] == "a" and e["shown"]), None)
        for k, key in enumerate(names):   # toggles are custom expressions ("unknown" preset)
            if key == "_mouth_neutral":
                continue   # implementation detail, not a user-selectable expression
            binds = [{"mesh": face_mesh, "index": k, "weight": 100}]
            if neutral_mouth is not None and neutral_mouth != k and (key in ("a", "i", "u", "e", "o") or expr[k].get("mouth_toggle")):
                binds.append({"mesh": face_mesh, "index": neutral_mouth, "weight": 100})
            groups.append({"name": presets.get(key, expr[k].get("label", key)),
                           "presetName": key if key in presets else "unknown", "isBinary": False,
                           "materialValues": [], "binds": binds})
        if "blink_l" in names and "blink_r" in names:
            groups.append({"name": "Blink", "presetName": "blink", "isBinary": False, "materialValues": [],
                           "binds": [{"mesh": face_mesh, "index": names.index(n), "weight": 100}
                                     for n in ("blink_l", "blink_r")]})
        # VRM's emotion presets, which VSeeFace and similar apps put on hotkeys, built from the
        # painted parts this character has; an emotion with none of its parts is left out.
        eyes = next((n for n in ("toggle:stars", "toggle:hearts") if n in names), None)
        for label, preset, parts in (("Joy", "joy", ("blink_l", "blink_r", "toggle:blush")),
                                     ("Angry", "angry", ("toggle:anger",)),
                                     ("Sorrow", "sorrow", ("toggle:tears", "toggle:gloom")),
                                     ("Fun", "fun", (eyes, "toggle:blush") if eyes else ())):
            binds = [{"mesh": face_mesh, "index": names.index(n), "weight": 100} for n in parts if n in names]
            if binds:
                groups.append({"name": label, "presetName": preset, "isBinary": False,
                               "materialValues": [], "binds": binds})
        if "a" in names:   # vowels the character has no painted mouth for open it partway
            for vowel, amount in (("i", 45), ("u", 35), ("e", 60), ("o", 80)):
                if vowel not in names:
                    a_index = names.index("a")
                    # When A itself is the closed neutral decal, the fallback must
                    # move it fully behind the face to reveal the original mouth.
                    binds = [{"mesh": face_mesh, "index": a_index,
                              "weight": 100 if neutral_mouth == a_index else amount}]
                    if neutral_mouth is not None and neutral_mouth != a_index:
                        binds.append({"mesh": face_mesh, "index": neutral_mouth, "weight": 100})
                    groups.append({"name": vowel.upper(), "presetName": vowel, "isBinary": False,
                                   "materialValues": [], "binds": binds})
    groups.append({"name": "Neutral", "presetName": "neutral", "isBinary": False, "binds": [], "materialValues": []})
    mesh_nodes = []
    for m in range(len(g["meshes"])):
        nodes.append({"name": g["meshes"][m]["name"], "mesh": m, "skin": 0})
        mesh_nodes.append(len(nodes) - 1)
    nodes[0]["children"] = [1]
    g["scenes"] = [{"nodes": [0] + mesh_nodes}]   # skinned meshes at the scene root, as glTF advises

    thumb = thumbnail.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    thumb_tex = W.image(_png(thumb), "image/png")
    head = INDEX["head"] + 1
    g["extensionsUsed"] = ["VRM"]
    g["extensions"] = {"VRM": {
        "exporterVersion": "Ka-myii-3D-0.1", "specVersion": "0.0",
        "meta": {"title": name, "version": "1", "author": "Ka-myii (generated)", "contactInformation": "",
                 "reference": "", "texture": thumb_tex, "allowedUserName": "OnlyAuthor",
                 "violentUssageName": "Disallow", "sexualUssageName": "Disallow",
                 "commercialUssageName": "Disallow", "otherPermissionUrl": "",
                 "licenseName": "Redistribution_Prohibited", "otherLicenseUrl": ""},
        "humanoid": {"humanBones": [{"bone": b, "node": i + 1, "useDefaultValues": True} for i, (b, _) in enumerate(BONES)],
                     "armStretch": 0.05, "legStretch": 0.05, "upperArmTwist": 0.5, "lowerArmTwist": 0.5,
                     "upperLegTwist": 0.5, "lowerLegTwist": 0.5, "feetSpacing": 0, "hasTranslationDoF": False},
        "firstPerson": {"firstPersonBone": head, "firstPersonBoneOffset": {"x": 0, "y": 0.06, "z": 0},
                        "meshAnnotations": [{"mesh": m, "firstPersonFlag": "Auto"} for m in range(len(g["meshes"]))],
                        "lookAtTypeName": "BlendShape",
                        **{k: {"curve": [0, 0, 0, 1, 1, 1, 1, 0], "xRange": 90, "yRange": 10}
                           for k in ("lookAtHorizontalInner", "lookAtHorizontalOuter",
                                     "lookAtVerticalDown", "lookAtVerticalUp")}},
        "blendShapeMaster": {"blendShapeGroups": groups},
        "secondaryAnimation": {
            "boneGroups": ([{"comment": "hair", "stiffiness": 1.0, "gravityPower": 0.15,
                            "gravityDir": {"x": 0, "y": -1, "z": 0}, "dragForce": 0.45, "center": -1,
                            "hitRadius": 0.02, "bones": [index[r] + 1 for r in springs["roots"]],
                            "colliderGroups": list(range(len(springs["colliders"])))}] if springs["roots"] else [])
                          + ([{"comment": "bust", "stiffiness": 1.2, "gravityPower": 0,
                               "gravityDir": {"x": 0, "y": -1, "z": 0}, "dragForce": 0.2, "center": -1,
                               "hitRadius": 0.02, "bones": [index[r] + 1 for r in springs["bust_roots"]],
                               "colliderGroups": []}] if springs.get("bust_roots") else []),
            "colliderGroups": [{"node": index[b] + 1, "colliders": [{"offset": {"x": 0, "y": float(y), "z": 0},
                                                                     "radius": float(r)}]}
                               for b, y, r in springs["colliders"]]},
        "materialProperties": mtoon,
    }}
    W.save(path)
