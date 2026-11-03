"""Ka-myii's engine node, copied into the private 3D engine's custom_nodes at every start.

KamyiiProjectImage: paint the sharp input artwork back onto the baked texture. Taken from
Atelier3D's AtelierProjectImage (github.com/KokiichiiCoding/Atelier3D, MIT, same owner); the
only change is that it also reports the camera fit as JSON, which Ka-myii's rig reuses so the
skeleton and the expression decals land where the artwork has them.

Pixal3D generates geometry pixel-aligned with its input crop under a known pinhole camera
(comfy/ldm/trellis2/model.py: camera on +Z looking at the origin, Y up, distance
0.5 / tan(fov / 2), horizontal fov from MoGe). The voxel-colour bake is softer than the
1024 px input, so for texels the camera sees head-on we take the photo's fine detail and keep
the bake's low frequencies (its albedo-like tone), which hides any lighting baked into the
photo and makes the blend seamless. A silhouette check guards the whole thing: if the mesh
seen from that camera does not match the input mask, the texture is returned unchanged.
"""
from __future__ import annotations

import json
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from typing_extensions import override

import comfy.model_management
from comfy_api.latest import IO, ComfyExtension
from comfy_extras import nodes_mesh_postprocess as mp
from comfy_extras.nodes_save_3d import get_mesh_batch_item

log = logging.getLogger("kamyii3d")
MIN_IOU = 0.85        # silhouette agreement needed before the photo is trusted
TIE = 0.02            # another camera must beat the derived one by this much
CHECK_RES = 256       # resolution of the silhouette check renders


def camera(azimuth_deg: float, fov_deg: float, distance_scale: float, device):
    """Eye, forward, right, up and normalised focal length of a Pixal3D orbit camera in the
    mesh frame (glTF: Y up, front view on +Z)."""
    az = math.radians(azimuth_deg)
    back = torch.tensor([math.sin(az), 0.0, math.cos(az)], device=device)
    right = torch.tensor([math.cos(az), 0.0, -math.sin(az)], device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    tan_half = math.tan(math.radians(fov_deg) / 2.0)
    eye = back * (distance_scale * 0.5 / tan_half)
    return eye, -back, right, up, 0.5 / tan_half


def project(points, eye, fwd, right, up, focal):
    """Mesh-frame points [K,3] -> image coords [K,2] in [0,1] (x right, y down) and distance."""
    rel = points - eye
    depth = (rel * fwd).sum(-1).clamp_min(1e-6)
    x = 0.5 + focal * (rel * right).sum(-1) / depth
    y = 0.5 - focal * (rel * up).sum(-1) / depth
    return torch.stack([x, y], -1), rel.norm(dim=-1)


def silhouette(tri, bvh, eye, fwd, right, up, focal, res):
    """Ray-cast the mesh from the camera: hit mask and distance per pixel ([res,res])."""
    dev = tri.device
    c = (torch.arange(res, device=dev, dtype=torch.float32) + 0.5) / res
    gy, gx = torch.meshgrid(c, c, indexing="ij")
    d = (right * ((gx - 0.5) / focal)[..., None] + up * ((0.5 - gy) / focal)[..., None] + fwd)
    d = F.normalize(d.reshape(-1, 3), dim=-1)
    t, _, hit = mp._closest_hit_rays_bvh(eye[None].expand_as(d), d, tri, bvh, tmin=1e-5, tmax=1e30)
    dist = torch.where(hit, t, torch.full_like(t, float("inf")))
    return hit.reshape(res, res), dist.reshape(res, res)


def iou(a, b):
    union = (a | b).sum().item()
    return (a & b).sum().item() / union if union else 0.0


def moments(m):
    """Centroid (x, y) and sqrt(area) of a boolean mask in [0,1] image units."""
    ys, xs = torch.nonzero(m, as_tuple=True)
    n = m.shape[0]
    if ys.numel() == 0:
        return None
    return ((xs.float().mean().item() + 0.5) / n, (ys.float().mean().item() + 0.5) / n,
            math.sqrt(ys.numel()) / n)


def similarity(xy, src, dst):
    """Map image coords [K,2] by the 2D similarity taking moments ``src`` onto ``dst``."""
    (sx, sy, ss), (dx, dy, ds) = src, dst
    k = ds / ss
    return torch.stack([dx + (xy[:, 0] - sx) * k, dy + (xy[:, 1] - sy) * k], -1)


def resize_hw(x, size, mode="bilinear"):
    """[H,W,C] or [H,W] -> square ``size``."""
    two_d = x.ndim == 2
    t = x[None, None] if two_d else x.permute(2, 0, 1)[None]
    kw = {} if mode == "nearest" else {"align_corners": False}
    t = F.interpolate(t.float(), size=(size, size), mode=mode, **kw)
    return t[0, 0] if two_d else t[0].permute(1, 2, 0)


def blur_masked(img, mask, sigma):
    """Gaussian blur of img [H,W,C] that only averages pixels inside mask [H,W] (no black halo)."""
    r = max(1, int(3 * sigma))
    k = torch.exp(-(torch.arange(-r, r + 1, device=img.device, dtype=torch.float32) ** 2) / (2 * sigma * sigma))
    k = k / k.sum()
    m = mask.float()[None, None]
    x = torch.cat([img.permute(2, 0, 1)[None] * m, m], 1)
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return (x[0, :-1] / x[0, -1:].clamp_min(1e-4)).permute(1, 2, 0)


def sample(img, xy):
    """Bilinear sample img [H,W,C] at xy [K,2] in [0,1]."""
    g = (xy * 2 - 1).view(1, 1, -1, 2)
    return F.grid_sample(img.permute(2, 0, 1)[None].float(), g, mode="bilinear",
                         padding_mode="border", align_corners=False)[0, :, 0].t()


def project_image(v, f, uv, base, image, mask, fov, azimuth=0.0, distance_scale=1.0,
                  strength=1.0, detail_sigma=12.0):
    """Core of KamyiiProjectImage on plain tensors: v [N,3], f [M,3], uv [N,2] (bake UVs),
    base [T,T,3], image [H,W,3] and mask [H,W] in [0,1]. Returns (new base [T,T,3], report)."""
    dev = comfy.model_management.get_torch_device()
    T = base.shape[0]
    v, f = v.to(dev).float(), f.to(dev).long()
    uv_np = mp._normalize_uvs_to_unit(uv.detach().cpu().numpy())   # the bake's exact UVs
    tri = v[f]
    bvh = mp._build_triangle_bvh(tri)
    R = image.shape[0]
    img = image.to(dev).float()[..., :3]
    msk = mask.to(dev).float() > 0.5

    # 1. which camera explains the input silhouette? (the Pixal3D front first). A small
    # residual scale/offset (pad factor, MoGe fov error) is corrected by a 2D similarity.
    small = resize_hw(msk.float(), CHECK_RES) > 0.5
    c = (torch.arange(CHECK_RES, device=dev, dtype=torch.float32) + 0.5) / CHECK_RES
    grid = torch.stack(torch.meshgrid(c, c, indexing="xy"), -1).reshape(-1, 2)
    # Score every hypothesis: a blobby shape can match a wrong one fairly well, so the first
    # one above the bar is not enough. The derived camera wins near-ties.
    best = None
    for az, vflip in ((azimuth, False), (azimuth + 180.0, False), (azimuth, True), (azimuth + 180.0, True)):
        cam = camera(az, fov, distance_scale, dev)
        target = small.flip(0) if vflip else small
        hit, _ = silhouette(tri, bvh, *cam, CHECK_RES)
        mh, mt = moments(hit), moments(target)
        if mh is None or mt is None:
            continue
        k = mt[2] / mh[2]
        if 0.8 < k < 1.25 and abs(mt[0] - mh[0]) < 0.1 and abs(mt[1] - mh[1]) < 0.1:
            warped = sample(target.float()[..., None], similarity(grid, mh, mt))[:, 0] > 0.5
            score = iou(hit, warped.reshape(CHECK_RES, CHECK_RES))
        else:
            score = 0.0
        if best is None or score > best[0] + TIE:
            best = (score, az, vflip, mh, mt)
    if best is None or best[0] < MIN_IOU:
        report = {"painted": 0.0, "score": 0.0 if best is None else best[0]}
        log.warning("KamyiiProjectImage skipped: silhouette match %.2f < %s", report["score"], MIN_IOU)
        return base, report
    score, az, vflip, mh, mt = best
    if vflip:
        img, msk = img.flip(0), msk.flip(0)
    cam = camera(az, fov, distance_scale, dev)
    eye, fwd, right, up, focal = cam
    k = mt[2] / mh[2]

    def to_photo(xy):
        return similarity(xy, mh, mt)

    # 2. the camera's view of the mesh: distance map for visibility, the bake for low frequencies
    hit, dist = silhouette(tri, bvh, *cam, R)
    near = -F.max_pool2d(-dist[None, None].nan_to_num(posinf=1e9), 3, 1, 1)[0, 0]  # 3x3 min: strict at edges
    rendered, _, _ = mp._render_view(tri, bvh, torch.from_numpy(uv_np).to(dev), f, base.to(dev).float(),
                                     eye, fwd, right, up, math.radians(fov), R, R)
    sigma = detail_sigma * R / 1024.0
    low_bake = blur_masked(rendered, hit, sigma)
    detail = img - blur_masked(img, msk, sigma)
    soft = msk.float()[None, None]
    erode = max(1, round(3 * R / 1024))
    soft = -F.max_pool2d(-soft, 2 * erode + 1, 1, erode)                     # stay off the cut-out edge
    soft = F.avg_pool2d(soft, 2 * erode + 1, 1, erode, count_include_pad=False)[0, 0][..., None]

    # 3. every covered texel: where is it, is it seen, how squarely?
    face_idx, bary, cov = mp._rasterize_uv_barycentric(f.cpu().numpy(), uv_np, T)
    idx = torch.nonzero(cov.reshape(-1), as_tuple=True)[0]
    fi = face_idx.reshape(-1)[idx]
    b = bary.reshape(-1, 3)[idx][:, :, None]
    pos = (b * v[f[fi]]).sum(1)
    vn = torch.from_numpy(mp._smooth_vertex_normals(v.cpu().numpy(), f.cpu().numpy())).to(dev)
    nrm = F.normalize((b * vn[f[fi]]).sum(1), dim=-1)
    xy, d = project(pos, *cam)
    facing = (nrm * F.normalize(eye - pos, dim=-1)).sum(-1)
    pix = d / (focal * R)                                                    # world size of a pixel there
    seen_at = sample(near[..., None], xy)[:, 0]
    visible = d <= seen_at + 0.004 + 3 * pix / facing.clamp_min(0.3)
    inside = (xy >= 0).all(-1) & (xy <= 1).all(-1)
    w = ((facing - 0.3) / 0.45).clamp(0, 1)
    w = w * w * (3 - 2 * w)                                                  # smoothstep: no grazing smear
    pxy = to_photo(xy)
    w = w * visible * inside * sample(soft, pxy)[:, 0] * float(strength)

    # 4. blend: photo detail on the bake's own low frequencies
    out = base.to(dev).float().reshape(-1, 3).clone()
    photo = (sample(detail, pxy) + sample(low_bake, xy)).clamp(0, 1)
    out[idx] = torch.lerp(out[idx], photo, w[:, None])
    out = out.reshape(T, T, 3)
    out = torch.from_numpy(mp._seam_fill(out.cpu().numpy(), cov.cpu().numpy()))  # refresh UV gutters
    # mesh-camera image coords -> artwork crop: similarity(xy, mesh_moments, photo_moments)
    report = {"painted": (w > 0.01).float().mean().item(), "score": score, "azimuth": az,
              "flipped": vflip, "mesh_moments": list(mh), "photo_moments": list(mt), "fov": fov}
    log.info("KamyiiProjectImage %s", report)
    return out, report


class KamyiiProjectImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="KamyiiProjectImage",
            display_name="Project Image Onto Texture (Ka-myii)",
            category="3d/texturing",
            description="Sharpen a baked base colour with the pixel-aligned input image (Pixal3D camera).",
            inputs=[
                IO.Mesh.Input("mesh", tooltip="The UV-unwrapped mesh the base colour was baked on."),
                IO.Image.Input("base_color"),
                IO.Image.Input("image", tooltip="The cropped input exactly as the conditioning saw it."),
                IO.Mask.Input("mask", tooltip="The subject mask, cropped like the image."),
                IO.Float.Input("fov", default=49.13, min=1.0, max=170.0, step=0.01,
                               tooltip="Horizontal FOV in degrees, the one the conditioning used."),
                IO.Float.Input("azimuth", default=0.0, min=-360.0, max=360.0, step=90.0,
                               tooltip="0 = front view; multi-view rigs: 90 left, 180 back, 270 right."),
                IO.Float.Input("distance_scale", default=1.0, min=0.5, max=2.0, step=0.01,
                               tooltip="1.0 for single-image Pixal3D, 1.1 for the multi-view rig."),
                IO.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.05),
                IO.Float.Input("detail_sigma", default=12.0, min=1.0, max=64.0, step=0.5,
                               tooltip="Pixels (at 1024) below which detail comes from the photo."),
            ],
            outputs=[IO.Image.Output(display_name="base_color")],
        )

    @classmethod
    def execute(cls, mesh, base_color, image, mask, fov, azimuth, distance_scale, strength,
                detail_sigma) -> IO.NodeOutput:
        if mesh.uvs is None:
            raise ValueError("KamyiiProjectImage: the mesh has no UVs (unwrap it first)")
        v, f, _, uvs, _ = get_mesh_batch_item(mesh, 0)
        uvs = uvs if uvs is not None else mesh.uvs[0, :v.shape[0]]
        T = base_color.shape[1]
        mp._prepare_gpu_mesh_processing(comfy.model_management.get_torch_device(),
                                        T * T * 256 + f.shape[0] * 512)
        out, report = project_image(v, f, uvs, base_color[0, ..., :3], image[0], mask[0], fov, azimuth,
                                    distance_scale, strength, detail_sigma)
        return IO.NodeOutput(out.float().clamp(0, 1).cpu()[None], ui={"text": (json.dumps(report),)})


class KamyiiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [KamyiiProjectImage]


async def comfy_entrypoint() -> KamyiiExtension:
    return KamyiiExtension()
