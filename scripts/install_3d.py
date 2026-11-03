"""
Install the optional 3D avatar engine into engines/comfy3d.

    python scripts/install_3d.py [--link-from <folder with ComfyUI-style models>]

The engine is a private, pinned ComfyUI (the same pin Atelier3D uses: its native Pixal3D nodes
are pure PyTorch, no CUDA extensions to compile) with its own virtualenv. Ka-myii starts it only
for a 3D job and stops it afterwards; it never touches a ComfyUI you may have installed.
Model files (~10 GB: Pixal3D int8, its DINOv3 encoder and VAEs, MoGe) are hard-linked from
--link-from (or a sibling Atelier3D checkout) when the same file is already on this drive,
otherwise downloaded from Hugging Face, plus DWPose (~134 MB) for the rig's joints.
Safe to re-run: finished steps are skipped.
Needs an NVIDIA GPU with 8 GB+ and a driver that supports CUDA 13.
"""
import argparse
import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engines" / "comfy3d"
COMFY_REF = "v0.42.1"
COMFY_REPO = "https://github.com/Comfy-Org/ComfyUI"
# comfy-kitchen only enables its int8 kernels on CUDA 13 builds of torch (see AGENTS.md)
TORCH = ["torch==2.14.1+cu130", "torchvision==0.29.1+cu130"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
HF = "https://huggingface.co"
# (ComfyUI models sub-folder, file, repo, path in repo, bytes) — Comfy-Org repacks, MIT
# (the DINOv3 encoder is under Meta's DINOv3 License).
MODELS = [
    ("diffusion_models", "pixal3d_int8_convrot.safetensors", "Comfy-Org/Pixal3D", "diffusion_models/pixal3d_int8_convrot.safetensors", 5584555824),
    ("clip_vision", "dino_v3_L_naf_fp32.safetensors", "Comfy-Org/Pixal3D", "clip_vision/dino_v3_L_naf_fp32.safetensors", 1215214176),
    ("vae", "trellis_2_shape_vae_bf16.safetensors", "Comfy-Org/Pixal3D", "vae/trellis_2_shape_vae_bf16.safetensors", 1095844024),
    ("vae", "trellis_2_texture_vae_bf16.safetensors", "Comfy-Org/Pixal3D", "vae/trellis_2_texture_vae_bf16.safetensors", 948461364),
    ("geometry_estimation", "moge_2_vitl_normal_fp16.safetensors", "Comfy-Org/MoGe", "geometry_estimation/moge_2_vitl_normal_fp16.safetensors", 661859924),
]
# Body keypoints for the rig (src/core/pose.py), into the shared Hugging Face cache; Apache-2.0.
DWPOSE = ("yzd-v/DWPose", "dw-ll_ucoco_384.onnx", "1a7144101628d69ee7a3768d1ee3a094070dc388")


def run(*cmd, **kw):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def find_uv():
    local = Path.home() / ".local" / "bin" / ("uv.exe" if os.name == "nt" else "uv")
    return shutil.which("uv") or (str(local) if local.is_file() else None)


def fetch_comfy():
    if (ENGINE / "ComfyUI" / "main.py").is_file():
        return
    ENGINE.mkdir(parents=True, exist_ok=True)
    if shutil.which("git"):
        run("git", "clone", "--depth", "1", "--branch", COMFY_REF, COMFY_REPO, ENGINE / "ComfyUI")
        return
    url = f"{COMFY_REPO}/archive/refs/tags/{COMFY_REF}.zip"   # no git: the release archive
    print(f"Downloading {url}...", flush=True)
    with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=120).read())) as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(ENGINE)
    (ENGINE / top).rename(ENGINE / "ComfyUI")


def build_venv():
    py = ENGINE / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    stamp = ENGINE / ".venv" / f".kamyii-{COMFY_REF}-{TORCH[0].replace('==', '-')}"
    if stamp.exists():
        return py
    uv = find_uv()
    if uv:  # hard-links wheels uv already has, so a second engine costs almost no disk
        os.environ.setdefault("UV_LINK_MODE", "hardlink")
        if not py.exists():
            run(uv, "venv", "--python", "3.12", ENGINE / ".venv")
        pip = [uv, "pip", "install", "--python", py]
        run(*pip, *TORCH, "--index-url", TORCH_INDEX, "--extra-index-url", "https://pypi.org/simple",
            "--index-strategy", "unsafe-best-match")
    else:
        if not py.exists():
            run(sys.executable, "-m", "venv", ENGINE / ".venv")
        pip = [py, "-m", "pip", "install", "--disable-pip-version-check"]
        run(*pip, *TORCH, "--extra-index-url", TORCH_INDEX)
    run(*pip, "-r", ENGINE / "ComfyUI" / "requirements.txt", "numba")
    stamp.write_text("ok")
    return py


def fetch_models(link_from):
    for folder, name, repo, path, size in MODELS:
        dest = ENGINE / "models" / folder / name
        if dest.is_file() and dest.stat().st_size == size:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = next((p for p in (Path(d) / folder / name for d in link_from)
                    if p.is_file() and p.stat().st_size == size), None)
        if src:
            dest.unlink(missing_ok=True)
            try:
                os.link(src, dest)
                print(f"linked {name} from {src.parent}", flush=True)
                continue
            except OSError as e:  # another drive: download instead of copying
                print(f"could not hard-link {src}: {e}", flush=True)
        part = dest.with_suffix(".part")
        have = part.stat().st_size if part.exists() else 0
        print(f"downloading {name} ({size / 1e9:.1f} GB)...", flush=True)
        req = urllib.request.Request(f"{HF}/{repo}/resolve/main/{path}", headers={"Range": f"bytes={have}-"})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if r.status == 206 else "wb") as f:
            shutil.copyfileobj(r, f, 8 << 20)
        if part.stat().st_size != size:
            raise SystemExit(f"{name}: download incomplete, run the installer again to resume")
        os.replace(part, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--link-from", action="append", default=[],
                    help="folder laid out like ComfyUI's models/ to hard-link files from")
    args = ap.parse_args()
    link_from = args.link_from + [str(ROOT.parent / "Atelier3D" / "models")]
    fetch_comfy()
    py = build_venv()
    fetch_models(link_from)
    for d in ("custom_nodes", "input", "output", "temp", "user"):
        (ENGINE / "data" / d).mkdir(parents=True, exist_ok=True)
    print("Fetching the DWPose body keypoint model (~134 MB)...", flush=True)
    repo, name, revision = DWPOSE
    run(py, "-c", f"from huggingface_hub import hf_hub_download; hf_hub_download({repo!r}, {name!r}, revision={revision!r})")
    run(py, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA GPU not visible'; "
                  "print('3D engine ready on', torch.cuda.get_device_name(0))")


if __name__ == "__main__":
    main()
