"""
Character artwork -> textured 3D mesh with Pixal3D (Tencent ARC; Comfy-Org's int8 repack).

Runs a private, pinned ComfyUI (``engines/comfy3d``, installed by scripts/install_3d.py) as a
child process for the length of one 3D job, like See-through: its torch stack stays out of
Ka-myii's venv and exiting returns every byte of VRAM and RAM to the next job. The graph is
Atelier3D's Pixal3D image-to-3D graph (Comfy-Org's official template) without its background
removal: Ka-myii frames the cut-out itself, so the crop and the mesh camera are known exactly
and rig3d.py can put joints and expression decals where the artwork has them.

Pixal3D's mesh frame is glTF (Y up, metres-ish unit cube); the input crop was seen by a pinhole
camera on +Z at distance 0.5 / tan(fov / 2) looking at the origin, fov = MoGe's estimate.
"""
from __future__ import annotations

import atexit
import json
import logging
import math
import os
import queue
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
NODES = ROOT / "scripts" / "comfy3d_nodes"
MODELS = {  # ComfyUI models sub-folder -> file; scripts/install_3d.py fetches them
    "diffusion_models": "pixal3d_int8_convrot.safetensors",
    "clip_vision": "dino_v3_L_naf_fp32.safetensors",
    "vae": "trellis_2_shape_vae_bf16.safetensors",
    "vae/": "trellis_2_texture_vae_bf16.safetensors",
    "geometry_estimation": "moge_2_vitl_normal_fp16.safetensors",
}
CROP, PAD = 1024, 1.1
# Sampler steps per quality, Atelier3D's presets (standard = Comfy's template, smaller upsample).
QUALITY = {
    "draft": dict(ss_steps=8, shape_steps=12, refine_steps=8, tex_steps=8, upsample=1024, remesh=512),
    "standard": dict(ss_steps=12, shape_steps=20, refine_steps=12, tex_steps=12, upsample=1024, remesh=768),
    "high": dict(ss_steps=12, shape_steps=20, refine_steps=12, tex_steps=12, upsample=1536, remesh=768),
}
# Share of the run per stage; KSampler tqdm bars (stdout) mark the boundaries.
STAGES = [("Preparing the artwork", 0.0), ("Blocking out the body", 0.12), ("Sculpting", 0.22),
          ("Refining detail", 0.45), ("Painting", 0.6), ("Baking the texture", 0.72)]
_TQDM = re.compile(r"^\s*\d+%\|.*?(\d+)/(\d+) \[")   # unnamed bars only: "MoGe inference: ..." is not a sampler
_live: set = set()   # Windows does not kill children with their parent
atexit.register(lambda: [p.kill() for p in list(_live)])


def _assert_port_free(port: int):
    """Reject *any* listener, including a non-Comfy service that returns 404 to our probe."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # Windows can otherwise allow overlapping binds under some reuse configurations.
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(("127.0.0.1", port))
    except OSError as exc:
        raise RuntimeError(f"Port {port} is already in use or unavailable. "
                           "Close the other service or set KAMYII_3D_PORT.") from exc


def engine_dir() -> Path:
    return Path(config.AVATAR3D["engine_dir"])


def engine_python() -> Path:
    return engine_dir() / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def available() -> bool:
    models = engine_dir() / "models"
    return (engine_python().exists() and (engine_dir() / "ComfyUI" / "main.py").exists()
            and all((models / folder.rstrip("/") / name).exists() for folder, name in MODELS.items()))


def frame(image: Image.Image, mask: np.ndarray, size: int = CROP, pad: float = PAD):
    """The cut-out framed like ComfyUI's ImageCropToMask: a square ``pad`` x the mask's
    bounding box, centred on it, subject on black. Returns (rgb, mask image, box) with
    box = (x0, y0, side) in artwork pixels: crop pixel = (artwork pixel - (x0, y0)) * size / side."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("No character found in the artwork")
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = max(x2 - x1, y2 - y1) * pad
    x0, y0 = (x1 + x2 - side) / 2, (y1 + y2 - side) / 2
    box = (x0, y0, x0 + side, y0 + side)
    source_alpha = Image.fromarray(mask.astype(np.uint8) * 255)
    alpha = source_alpha.crop(box).resize((size, size), Image.Resampling.BILINEAR)
    # Resample the cut-out as premultiplied RGBA: resizing the original RGB first
    # mixes its background into the silhouette, leaving a light fringe on hair
    # and clothing when the character is later composited against black.
    cutout = image.convert("RGB").crop(box).convert("RGBA")
    cutout.putalpha(source_alpha.crop(box))
    rgb = cutout.resize((size, size), Image.Resampling.LANCZOS).convert("RGB")
    rgb = Image.composite(rgb, Image.new("RGB", rgb.size), alpha)
    return rgb, alpha, (float(x0), float(y0), float(side))


def graph(image_name: str, mask_name: str, seed: int, quality: str = "standard",
          faces: int = 100_000, texture: int = 2048) -> tuple[dict, dict]:
    """ComfyUI API prompt for one framed image, and the ids of the nodes whose outputs matter."""
    if quality not in QUALITY:
        raise ValueError(f"quality must be one of {', '.join(QUALITY)}")
    q = QUALITY[quality]
    nodes: dict = {}

    def add(class_type, **inputs):
        nid = str(len(nodes) + 1)
        nodes[nid] = {"class_type": class_type, "inputs": inputs}
        return nid

    img = add("LoadImage", image=image_name)
    msk = add("LoadImageMask", image=mask_name, channel="red")
    dino = add("CLIPVisionLoader", clip_name=MODELS["clip_vision"])
    moge = add("LoadMoGeModel", model_name=MODELS["geometry_estimation"])
    geo = add("MoGeInference", moge_model=[moge, 0], image=[img, 0], resolution_level=9, fov_x_degrees=0.0,
              batch_size=4, force_projection=True, apply_mask=True, refine_steps=3)
    fov = add("MoGeGeometryToFOV", moge_geometry=[geo, 0], axis="horizontal", unit="degrees")
    fov_out = add("PreviewAny", source=[fov, 0])
    cond = add("Pixal3DConditioning", clip_vision_model=[dino, 0], image=[img, 0], camera_angle_x=[fov, 0])

    unet = add("UNETLoader", unet_name=MODELS["diffusion_models"], weight_dtype="default")
    shape_vae = add("VAELoader", vae_name=MODELS["vae"])
    # CFG schedules match the original Pixal3D / TRELLIS.2 pipeline defaults
    m = add("CFGOverride", model=[unet, 0], cfg=1.0, start_percent=0.667, end_percent=1.0)
    m = add("RescaleCFG", model=[m, 0], multiplier=0.7)
    m_struct = add("ModelSamplingSD3", model=[m, 0], shift=5.0)
    m = add("CFGOverride", model=[unet, 0], cfg=1.0, start_percent=0.769, end_percent=1.0)
    m_shape = add("RescaleCFG", model=[m, 0], multiplier=0.5)
    sampler = dict(cfg=7.5, sampler_name="euler", denoise=1.0)

    empty = add("EmptyTrellis2LatentStructure", batch_size=1)
    s = add("KSampler", model=[m_struct, 0], seed=seed, steps=q["ss_steps"], scheduler="normal",
            positive=[cond, 0], negative=[cond, 1], latent_image=[empty, 0], **sampler)
    voxel = add("VaeDecodeStructureTrellis2", samples=[s, 0], vae=[shape_vae, 0], resolution="32")
    st = add("Trellis2ShapeStage", positive=[cond, 0], negative=[cond, 1], voxel=[voxel, 0])
    shape = add("KSampler", model=[m_shape, 0], seed=seed + 1, steps=q["shape_steps"], scheduler="normal",
                positive=[st, 0], negative=[st, 1], latent_image=[st, 2], **sampler)
    up = add("Trellis2UpsampleStage", positive=[st, 0], negative=[st, 1], shape_latent=[shape, 0],
             vae=[shape_vae, 0], target_resolution=q["upsample"])
    fine = add("KSampler", model=[m_shape, 0], seed=seed + 1, steps=q["refine_steps"], scheduler="simple",
               positive=[up, 0], negative=[up, 1], latent_image=[up, 2], **sampler)
    hi = add("VaeDecodeShapeTrellis", samples=[fine, 0], vae=[shape_vae, 0])
    tex_vae = add("VAELoader", vae_name=MODELS["vae/"])
    ts = add("Trellis2TextureStage", positive=[up, 0], negative=[up, 1], shape_latent=[fine, 0])
    tex = add("KSampler", model=[unet, 0], seed=seed + 2, steps=q["tex_steps"], cfg=1.0, sampler_name="euler",
              scheduler="normal", positive=[ts, 0], negative=[ts, 1], latent_image=[ts, 2], denoise=1.0)
    colors = add("VaeDecodeTextureTrellis", samples=[tex, 0], vae=[tex_vae, 0], shape_subdivides=[hi, 1])

    remeshed = add("RemeshMesh", mesh=[hi, 0], resolution=q["remesh"], sign_mode="solid",
                   **{"sign_mode.fill": 20.0, "sign_mode.qef": False}, band=1.0, project_back=0.0,
                   fix_poles=False, smooth_iters=20, drop_small_components=0.01, precluster_max_verts=20_000_000)
    low = add("DecimateMesh", mesh=[remeshed, 0], target_face_count=faces, placement_mode="midpoint")
    low = add("MeshSmoothNormals", mesh=[low, 0], crease_angle=180.0)
    uv = add("UnwrapMesh", mesh=[low, 0], segmenter="pec", resolution=texture, padding=max(2, texture // 1024),
             weld_distance=0.0002)
    maps = add("BakeTextureFromVoxel", mesh=[uv, 0], voxel_colors=[colors, 0], texture_size=texture,
               reference_mesh=[hi, 0])
    # shortcut: no normal/AO bakes (MToon avatars shade flat colour), add them for PBR use.
    paint = add("KamyiiProjectImage", mesh=[uv, 0], base_color=[maps, 0], image=[img, 0], mask=[msk, 0],
                fov=[fov, 0], azimuth=0.0, distance_scale=1.0, strength=1.0, detail_sigma=12.0)
    final = add("ApplyTextureToMesh", mesh=[uv, 0], base_color=[paint, 0])
    final = add("MeshSmoothNormals", mesh=[final, 0], crease_angle=180.0)
    model = add("SaveGLB", mesh=[final, 0], filename_prefix="kamyii/mesh")
    return nodes, {"model": model, "fov": fov_out, "paint": paint}


class Engine:
    """The private ComfyUI, alive inside a ``with`` block. ``progress(frac, label)`` is called
    at least every second (it raises InterruptedError to cancel, which kills the engine)."""

    def __init__(self, progress: Optional[Callable[[float, str], None]] = None, log_path: Optional[Path] = None):
        self.url = f"http://127.0.0.1:{config.AVATAR3D['port']}"
        self.progress = progress or (lambda frac, label: None)
        self.log_path = log_path or ROOT / "logs" / "comfy3d.log"
        self.proc: Optional[subprocess.Popen] = None
        self.tail: list = []
        self.bound = threading.Event()
        self.lines: queue.Queue = queue.Queue()

    def _get(self, path: str, timeout: float = 5):
        import requests

        r = requests.get(self.url + path, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def __enter__(self):
        _assert_port_free(config.AVATAR3D["port"])
        data = engine_dir() / "data"
        for d in ("custom_nodes", "input", "output", "temp", "user"):
            (data / d).mkdir(parents=True, exist_ok=True)
        for pkg in NODES.iterdir():
            if pkg.is_dir():
                shutil.copytree(pkg, data / "custom_nodes" / pkg.name, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__"))
        extra_args = shlex.split(config.AVATAR3D["engine_args"])
        if "--dont-print-server" in extra_args:
            raise ValueError("Remove --dont-print-server from KAMYII_3D_ENGINE_ARGS; "
                             "Ka-myii needs the startup message to verify its own engine.")
        restricted = {"--listen", "--port", "--base-directory", "--models-directory"}
        if any(arg.split("=", 1)[0] in restricted for arg in extra_args):
            raise ValueError("KAMYII_3D_ENGINE_ARGS cannot override the private engine's "
                             "listen address, port or data directories.")
        cmd = [str(engine_python()), "-u", str(engine_dir() / "ComfyUI" / "main.py"), "--listen", "127.0.0.1",
               "--port", str(config.AVATAR3D["port"]), "--base-directory", str(data),
               "--models-directory", str(engine_dir() / "models"), "--disable-auto-launch",
               *extra_args]
        logger.info("3D engine: %s", " ".join(cmd))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(cmd, cwd=engine_dir() / "ComfyUI", stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        _live.add(self.proc)
        threading.Thread(target=self._pump, name="comfy3d-output", daemon=True).start()
        try:
            self._wait_ready()
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc):
        if self.proc:
            proc = self.proc
            self.proc = None
            try:
                if proc.poll() is None:
                    proc.kill()
            except OSError:
                # The child can exit between poll and kill; do not mask the real error.
                if proc.poll() is None:
                    raise
            finally:
                try:
                    proc.wait()
                finally:
                    _live.discard(proc)

    def _pump(self):
        with open(self.log_path, "a", encoding="utf-8", errors="replace") as log:
            buf = b""
            while True:
                chunk = self.proc.stdout.read1(4096) if self.proc else b""
                if not chunk:
                    return
                log.write(chunk.decode("utf-8", "replace"))
                log.flush()
                *lines, buf = re.split(rb"[\r\n]", buf + chunk)  # tqdm redraws with \r
                for raw in lines:
                    line = raw.decode("utf-8", "replace").strip()
                    if line:
                        # Keep proof of binding even if subsequent startup logs fill the tail.
                        if f"To see the GUI go to: {self.url}" in line:
                            self.bound.set()
                        self.tail[:] = (self.tail + [line])[-40:]
                        self.lines.put(line)

    def _died(self) -> RuntimeError:
        text = "\n".join(self.tail[-15:])
        if "cudaGetDeviceCount" in text or "GPU access blocked" in text:
            return RuntimeError("The 3D engine cannot reach the NVIDIA GPU (Windows blocks it after a "
                                "driver crash). Restart the PC and try again.")
        return RuntimeError(f"The 3D engine stopped (out of memory?). Last lines:\n{text}")

    def _wait_ready(self, timeout: float = 420):
        import requests

        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            self.progress(0.0, "Starting the 3D engine")
            if self.proc.poll() is not None:
                raise self._died()
            # ComfyUI prints this only *after* binding its socket. The preflight bind alone
            # has a race: another service could claim the port before the child does.
            if not self.bound.is_set():
                time.sleep(1)
                continue
            try:
                stats = self._get("/system_stats", 2)
                device = (stats.get("devices") or [{}])[0]
                if device.get("type") != "cuda":
                    raise RuntimeError("The 3D engine found no CUDA GPU; Pixal3D needs an NVIDIA card.")
                return
            except requests.RequestException:
                time.sleep(1)
        raise RuntimeError(f"The 3D engine did not start within {timeout:.0f} s; see {self.log_path}")

    def _interruptible(self, call, frac: float, label: str):
        """Keep cancellation responsive while a busy engine holds its HTTP response open."""
        result = queue.Queue(maxsize=1)

        def fetch():
            try:
                result.put((call(), None))
            except Exception as exc:
                result.put((None, exc))

        threading.Thread(target=fetch, name="comfy3d-history", daemon=True).start()
        while True:
            try:
                value, error = result.get(timeout=1)
                if error:
                    raise error
                return value
            except queue.Empty:
                self.progress(frac, label)
                if self.proc.poll() is not None:
                    raise self._died()

    def _history(self, pid: str, frac: float, label: str):
        return self._interruptible(lambda: self._get(f"/history/{pid}", 30), frac, label)

    def run(self, prompt: dict) -> dict:
        """Queue ``prompt``, follow it to the end and return its history outputs."""
        import requests

        r = self._interruptible(lambda: requests.post(self.url + "/prompt",
            json={"prompt": prompt, "client_id": uuid.uuid4().hex}, timeout=60), 0.0, "Queueing the 3D job")
        if r.status_code != 200:
            raise RuntimeError(describe_rejection(r))
        pid = r.json()["prompt_id"]
        bars = sum(1 for n in prompt.values() if n["class_type"] == "KSampler")
        done, last, reported, baking = 0, 0.0, 0.0, None
        while True:
            try:
                while True:
                    m = _TQDM.search(self.lines.get_nowait())
                    if m:
                        frac = int(m.group(1)) / max(1, int(m.group(2)))
                        done += frac < last   # the next sampler's bar started
                        last = frac
            except queue.Empty:
                pass
            # stage 0 until the first sampler bar, 1..bars while sampling, then baking
            stage = len(STAGES) - 1 if done + (last >= 1) >= bars else (done + 1 if done or last else 0)
            start = STAGES[stage][1]
            end = STAGES[stage + 1][1] if stage + 1 < len(STAGES) else 0.98
            inside = last if 0 < stage < len(STAGES) - 1 else 0.0
            if stage == len(STAGES) - 1:   # remesh/unwrap/bake print no steps: creep (~90 s measured)
                baking = baking or time.monotonic()
                inside = 1 - math.exp(-(time.monotonic() - baking) / 60)
            reported = max(reported, start + (end - start) * inside)
            self.progress(reported, STAGES[stage][0])
            if self.proc.poll() is not None:
                raise self._died()
            try:  # the server stalls for seconds while a node holds the GIL (loading, remeshing)
                hist = self._history(pid, reported, STAGES[stage][0]).get(pid)  # written once finished
            except requests.RequestException:
                continue
            if hist:
                if (hist.get("status") or {}).get("status_str") == "error":
                    raise RuntimeError(execution_error(hist["status"]))
                return hist.get("outputs", {})
            time.sleep(1)


def describe_rejection(r) -> str:
    """ComfyUI's validation error JSON as one readable line."""
    try:
        j = r.json()
    except ValueError:
        return f"The 3D engine rejected the job ({r.status_code}): {r.text[:300]}"
    msgs = [f"{err.get('class_type', nid)}: {e.get('message')} {e.get('details', '')}".strip()
            for nid, err in (j.get("node_errors") or {}).items() for e in err.get("errors", [])]
    head = (j.get("error") or {}).get("message", "The 3D engine rejected the job")
    return head + (" - " + "; ".join(msgs) if msgs else "")


def execution_error(status: dict) -> str:
    for kind, data in status.get("messages", []):
        if kind == "execution_error":
            msg = (data.get("exception_message") or "").strip()
            if "out of memory" in msg.lower() or "OutOfMemory" in (data.get("exception_type") or ""):
                return "The GPU ran out of memory while building the 3D model. Close other GPU apps or try Draft quality."
            return f"{data.get('node_type', 'A 3D node')} failed: {msg}"
    return "The 3D job failed (see logs/comfy3d.log)"


def _model_output(info: dict) -> Path:
    """Accept only the GLB produced in our fixed SaveGLB output folder."""
    if not isinstance(info, dict):
        raise RuntimeError("The 3D engine returned an invalid model location")
    name = info.get("filename")
    if (info.get("type", "output") != "output" or info.get("subfolder", "") != "kamyii"
            or not isinstance(name, str) or not name or "/" in name or "\\" in name
            or Path(name).suffix.lower() != ".glb"):
        raise RuntimeError("The 3D engine returned an invalid model location")
    output = (engine_dir() / "data" / "output").resolve()
    root = (output / "kamyii").resolve()
    source = (root / name).resolve()
    if root.parent != output or source.parent != root or not source.is_file():
        raise RuntimeError("The 3D engine returned an invalid or missing model")
    return source


def generate(image: Image.Image, mask: np.ndarray, out_dir: Path, seed: int, quality: str = "standard",
             faces: int = 100_000, texture: int = 2048,
             progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """Build ``out_dir/mesh.glb`` from the artwork. Also writes the framed crop and mask and
    ``camera.json`` (crop box, MoGe fov, the photo-paint camera fit). Returns the camera dict."""
    if not available():
        raise RuntimeError("The 3D engine is not installed. Run: python scripts/install_3d.py")
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb, alpha, box = frame(image, mask)
    rgb.save(out_dir / "crop.png")
    alpha.save(out_dir / "crop_mask.png")
    staged = engine_dir() / "data" / "input" / "kamyii"
    staged.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:12]
    staged_files = [staged / f"{tag}.png", staged / f"{tag}_mask.png"]
    try:
        shutil.copyfile(out_dir / "crop.png", staged_files[0])
        shutil.copyfile(out_dir / "crop_mask.png", staged_files[1])
        prompt, ids = graph(f"kamyii/{tag}.png", f"kamyii/{tag}_mask.png", seed, quality, faces, texture)
        with Engine(progress, out_dir / "engine.log") as engine:
            outputs = engine.run(prompt)
    finally:
        for f in staged_files:
            f.unlink(missing_ok=True)
    files = outputs.get(ids["model"], {}).get("3d") or []
    if not files:
        raise RuntimeError("The 3D engine finished without a model (see engine.log)")
    src = _model_output(files[0])
    shutil.move(str(src), out_dir / "mesh.glb")
    fov = float(outputs[ids["fov"]]["text"][0])
    paint = json.loads(outputs.get(ids["paint"], {}).get("text", ["{}"])[0])
    camera = {"fov": fov, "box": box, "crop": CROP, "paint": paint, "seed": seed, "quality": quality,
              "faces": faces, "texture": texture}
    (out_dir / "camera.json").write_text(json.dumps(camera, indent=2))
    return camera
