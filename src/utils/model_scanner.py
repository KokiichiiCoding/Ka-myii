"""
Model scanner — discovers locally installed checkpoints, LoRAs and VAEs and
merges them with the curated Hugging Face presets for the UI dropdowns.

This mirrors the convenience of Stable Diffusion WebUI / ComfyUI where you
simply drop a ``.safetensors`` file into a folder and it appears in the picker.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import config

logger = logging.getLogger(__name__)

# File extensions recognised as model weights.
CHECKPOINT_EXTS = {".safetensors", ".ckpt"}
LORA_EXTS = {".safetensors", ".pt"}
VAE_EXTS = {".safetensors", ".pt", ".ckpt"}


def _scan_dir(directory: Path, exts: set) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: p.name.lower(),
    )


def list_checkpoints() -> List[Dict]:
    """Return checkpoints available to the UI.

    Combines curated Hugging Face presets (always selectable; auto-downloaded
    on first use) with any local ``.safetensors`` / ``.ckpt`` files found in
    ``models/checkpoints``.
    """
    entries: List[Dict] = []
    seen_local_names = set()

    # Local checkpoints first — these are what power users actually want.
    for path in _scan_dir(config.CHECKPOINTS_DIR, CHECKPOINT_EXTS):
        seen_local_names.add(path.stem.lower())
        entries.append(
            {
                "id": str(path),
                "name": path.stem,
                "pipeline": "sdxl" if "xl" in path.stem.lower() else "auto",
                "source": "local",
                "tags": ["local"],
                "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
                "note": f"Local checkpoint: {path.name}",
            }
        )

    # Curated presets. A "local" preset is only offered if the matching file is
    # actually present; HF presets are always offered (auto-download).
    for preset in config.MODEL_PRESETS:
        if preset["source"] == "local":
            stem = preset["id"].lower()
            if not any(stem in name or name in stem for name in seen_local_names):
                # Still show it as a hint, but flag that the file is missing.
                entries.append({**preset, "available": False})
                continue
        entries.append({**preset, "available": True})

    return entries


def list_loras() -> List[Dict]:
    """Return LoRA adapters found in ``models/loras``."""
    loras: List[Dict] = []
    for path in _scan_dir(config.LORA_DIR, LORA_EXTS):
        loras.append(
            {
                "id": str(path),
                "name": path.stem,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
            }
        )
    return loras


def list_vaes() -> List[Dict]:
    """Return VAEs found in ``models/vae``."""
    vaes: List[Dict] = []
    for path in _scan_dir(config.VAE_DIR, VAE_EXTS):
        vaes.append({"id": str(path), "name": path.stem})
    return vaes


def list_samplers() -> List[str]:
    """Return the ordered list of sampler names for the UI."""
    return list(config.SAMPLERS.keys())


def resolve_checkpoint(identifier: str) -> Dict:
    """Resolve a checkpoint identifier (local path or HF id) into load info.

    Returns a dict with ``model_name``, ``custom_model_path`` and ``pipeline``
    suitable for constructing an :class:`ImageGenerator`.
    """
    if not identifier:
        return {
            "model_name": config.IMAGE_GENERATION["model_name"],
            "custom_model_path": config.IMAGE_GENERATION.get("custom_model_path"),
            "pipeline": config.IMAGE_GENERATION.get("pipeline", "auto"),
        }

    path = Path(identifier)
    if path.exists() and path.is_file():
        return {
            "model_name": path.stem,
            "custom_model_path": str(path),
            "pipeline": "sdxl" if "xl" in path.stem.lower() else "auto",
        }

    # Otherwise treat it as a Hugging Face repo id, matching a preset if known.
    for preset in config.MODEL_PRESETS:
        if preset["id"] == identifier:
            return {
                "model_name": identifier,
                "custom_model_path": None,
                "pipeline": preset.get("pipeline", "auto"),
            }

    return {"model_name": identifier, "custom_model_path": None, "pipeline": "auto"}
