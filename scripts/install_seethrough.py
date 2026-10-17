"""
Install the See-through layer-decomposition engine into engines/see-through.

    python scripts/install_seethrough.py

Downloads a pinned See-through release (no git needed), creates its own virtualenv,
installs CUDA PyTorch + inference deps, and pre-fetches the 4-bit model weights
(~5.3 GB) so the first generation doesn't stall. Safe to re-run: finished steps are skipped.
Needs an NVIDIA GPU (8 GB+). ~9 GB of disk in total.
"""
import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engines" / "see-through"
COMMIT = "df019de5129d6c4b406587a14c3501669441a783"  # shitagaki-lab/see-through, 2026-10-06
TORCH = ["torch==2.8.0+cu128", "torchvision==0.23.0+cu128", "--index-url", "https://download.pytorch.org/whl/cu128"]
WEIGHTS = ["24yearsold/seethroughv0.0.2_layerdiff3d_nf4", "24yearsold/seethroughv0.0.1_marigold_nf4"]
# Character cut-out model the app reads from the shared HF cache (src/core/asset_separator.py ANIME_SEG).
ANIME_SEG = ("skytnt/anime-seg", "isnetis.onnx", "493cb60893f47441b26ec4fb9a306bce9e342982")


def run(*cmd, **kw):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    if not (ENGINE / "inference" / "scripts").exists():
        url = f"https://github.com/shitagaki-lab/see-through/archive/{COMMIT}.zip"
        print(f"Downloading See-through ({COMMIT[:7]})...", flush=True)
        data = urllib.request.urlopen(url, timeout=120).read()
        tmp = ENGINE.parent / "_see-through-download"
        shutil.rmtree(tmp, ignore_errors=True)
        zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
        ENGINE.parent.mkdir(parents=True, exist_ok=True)
        (tmp / f"see-through-{COMMIT}").rename(ENGINE)
        shutil.rmtree(tmp, ignore_errors=True)

    venv_python = ENGINE / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not venv_python.exists():
        run(sys.executable, "-m", "venv", ENGINE / ".venv")
    run(venv_python, "-m", "pip", "install", "--disable-pip-version-check", *TORCH)
    run(venv_python, "-m", "pip", "install", "--disable-pip-version-check", "-r", ROOT / "requirements-seethrough.txt", cwd=ROOT)

    print("Fetching 4-bit See-through weights (~5.3 GB)...", flush=True)
    run(venv_python, "-c", f"from huggingface_hub import snapshot_download\nfor r in {WEIGHTS!r}: snapshot_download(r)")
    print("Fetching the anime character segmentation model (~176 MB)...", flush=True)
    repo, name, revision = ANIME_SEG
    run(venv_python, "-c", f"from huggingface_hub import hf_hub_download\nhf_hub_download({repo!r}, {name!r}, revision={revision!r})")
    run(venv_python, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA GPU not visible'; print('See-through ready on', torch.cuda.get_device_name(0))")


if __name__ == "__main__":
    main()
