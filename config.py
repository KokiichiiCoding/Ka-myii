"""
Configuration settings for Ka-myii — Automated VTuber Model Studio.

Centralises every tunable knob for the modern (2026) pipeline:
image generation (SDXL / Illustrious / NoobAI class models), asset
decomposition, PSD + Live2D packaging, and the web UI.

Environment variables override every important default so the app can be
configured without editing code (see ``.env.example``).
"""
import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Model asset sub-directories (Stable-Diffusion-WebUI style layout). Drop your
# downloaded checkpoints / LoRAs / VAEs / embeddings here and Ka-myii will pick
# them up automatically in the model dropdowns.
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LORA_DIR = MODELS_DIR / "loras"
VAE_DIR = MODELS_DIR / "vae"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"

for _directory in (
    OUTPUT_DIR,
    MODELS_DIR,
    CHECKPOINTS_DIR,
    LORA_DIR,
    VAE_DIR,
    EMBEDDINGS_DIR,
):
    _directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Flask / server settings
# ---------------------------------------------------------------------------
FLASK_ENV = os.getenv("FLASK_ENV", "development")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = _env_int("PORT", 5000)
DEBUG = FLASK_ENV == "development"


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------
# Default to a strong, openly-hosted anime SDXL model so a fresh install can
# auto-download something high quality. Power users should drop an
# Illustrious-XL / NoobAI-XL / Pony / Holodayo checkpoint into
# ``models/checkpoints`` and select it in the UI for best results.
DEFAULT_MODEL = os.getenv("KAMYII_MODEL_NAME", "cagliostrolab/animagine-xl-3.1")

IMAGE_GENERATION = {
    # Portrait orientation at SDXL-native resolution gives full-body, rig-ready
    # framing. These are the defaults the UI loads with.
    "default_width": _env_int("KAMYII_DEFAULT_WIDTH", 832),
    "default_height": _env_int("KAMYII_DEFAULT_HEIGHT", 1216),
    "default_steps": _env_int("KAMYII_DEFAULT_STEPS", 28),
    "default_guidance_scale": float(os.getenv("KAMYII_DEFAULT_CFG", "6.5")),
    "default_sampler": os.getenv("KAMYII_DEFAULT_SAMPLER", "DPM++ 2M Karras"),
    "model_name": DEFAULT_MODEL,
    # Supported values: "auto", "sd15", "sdxl". SDXL is the modern default.
    "pipeline": os.getenv("KAMYII_MODEL_PIPELINE", "auto"),
    # Optional path to a local .safetensors / .ckpt checkpoint (CivitAI etc.).
    "custom_model_path": os.getenv("KAMYII_CUSTOM_MODEL_PATH"),
    # Optional custom VAE (diffusers folder or .safetensors). Only used if set.
    "vae_path": os.getenv("KAMYII_CUSTOM_VAE_PATH"),
    # Optional original SD config file for legacy single-file checkpoints.
    "original_config_file": os.getenv("KAMYII_ORIGINAL_CONFIG"),
    # Enable the SDXL refiner pass for extra detail when available.
    "use_refiner": _env_bool("KAMYII_USE_REFINER", False),
    "refiner_model": os.getenv(
        "KAMYII_REFINER_MODEL", "stabilityai/stable-diffusion-xl-refiner-1.0"
    ),
    # CPU offload trades speed for lower VRAM — handy on 8GB cards.
    "enable_cpu_offload": _env_bool("KAMYII_CPU_OFFLOAD", False),
    "clip_skip": _env_int("KAMYII_CLIP_SKIP", 2),
}

# Curated 2026 model presets surfaced in the UI. ``source`` is either a
# Hugging Face repo id (auto-downloaded) or "local" (scanned from
# ``models/checkpoints``). These reflect the current state-of-the-art for
# anime / VTuber character art.
MODEL_PRESETS = [
    {
        "id": "cagliostrolab/animagine-xl-3.1",
        "name": "Animagine XL 3.1",
        "pipeline": "sdxl",
        "source": "huggingface",
        "tags": ["anime", "general", "auto-download"],
        "note": "Well-rounded anime SDXL, hosted on Hugging Face. Great default.",
    },
    {
        "id": "Illustrious-XL",
        "name": "Illustrious-XL (local)",
        "pipeline": "sdxl",
        "source": "local",
        "tags": ["anime", "illustration", "recommended"],
        "note": "Place Illustrious-XL .safetensors in models/checkpoints. Best line work.",
    },
    {
        "id": "NoobAI-XL",
        "name": "NoobAI-XL v-pred (local)",
        "pipeline": "sdxl",
        "source": "local",
        "tags": ["anime", "v-prediction", "recommended"],
        "note": "v-prediction anime model built on Illustrious. Superb detail.",
    },
    {
        "id": "Holodayo-XL",
        "name": "Holodayo XL 2.1 (local)",
        "pipeline": "sdxl",
        "source": "local",
        "tags": ["vtuber", "anime"],
        "note": "Tuned specifically for VTuber aesthetics. Place in models/checkpoints.",
    },
    {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "name": "SDXL Base 1.0",
        "pipeline": "sdxl",
        "source": "huggingface",
        "tags": ["general", "auto-download"],
        "note": "Vanilla SDXL base. Reliable fallback for non-anime styles.",
    },
]

# Diffusers scheduler map for the sampler dropdown. Each entry names the
# diffusers scheduler class and any config overrides applied on load.
SAMPLERS = {
    "Euler a": {"class": "EulerAncestralDiscreteScheduler", "config": {}},
    "Euler": {"class": "EulerDiscreteScheduler", "config": {}},
    "DPM++ 2M": {"class": "DPMSolverMultistepScheduler", "config": {}},
    "DPM++ 2M Karras": {
        "class": "DPMSolverMultistepScheduler",
        "config": {"use_karras_sigmas": True},
    },
    "DPM++ SDE Karras": {
        "class": "DPMSolverSinglestepScheduler",
        "config": {"use_karras_sigmas": True},
    },
    "UniPC": {"class": "UniPCMultistepScheduler", "config": {}},
    "DDIM": {"class": "DDIMScheduler", "config": {}},
    "Heun": {"class": "HeunDiscreteScheduler", "config": {}},
    "LMS": {"class": "LMSDiscreteScheduler", "config": {}},
}

# Style presets injected into prompts. Each is (positive suffix, negative suffix).
STYLE_PRESETS = {
    "anime": {
        "positive": "masterpiece, best quality, very aesthetic, absurdres, anime style, clean lineart",
        "negative": "",
    },
    "vtuber": {
        "positive": (
            "masterpiece, best quality, very aesthetic, absurdres, vtuber reference sheet, "
            "single character, full body, facing viewer, neutral pose, arms slightly apart, "
            "symmetrical, flat lighting, simple background, white background"
        ),
        "negative": "",
    },
    "live2d": {
        "positive": (
            "masterpiece, best quality, absurdres, character reference sheet, single character, "
            "full body, facing viewer, T-pose, symmetrical, even flat lighting, no shadows, "
            "simple background, white background, separated hair"
        ),
        "negative": "dramatic lighting, harsh shadows, motion blur, dutch angle",
    },
    "chibi": {
        "positive": "masterpiece, best quality, chibi, cute, deformed, simple shading, full body",
        "negative": "",
    },
    "realistic": {
        "positive": "masterpiece, best quality, highly detailed, semi-realistic, soft lighting",
        "negative": "flat colors",
    },
}

# Quality / negative scaffolding shared across styles. Tuned for Danbooru-tag
# SDXL anime models (Illustrious / NoobAI / Animagine / Pony).
QUALITY_PROMPT = "masterpiece, best quality, very aesthetic, absurdres, newest"
DEFAULT_NEGATIVE_PROMPT = (
    "lowres, worst quality, low quality, bad anatomy, bad hands, missing fingers, "
    "extra digits, fewer digits, cropped, jpeg artifacts, signature, watermark, "
    "username, blurry, text, error, multiple views, multiple girls, multiple boys, "
    "monochrome, greyscale, sketch, deformed, disfigured, mutated, extra limbs"
)


# ---------------------------------------------------------------------------
# Asset separation (decomposition into rig-ready layers)
# ---------------------------------------------------------------------------
# Ordered back-to-front. This is the draw order used in the PSD / Live2D export
# and mirrors the layer taxonomy used by modern decomposition tools
# (e.g. "See-through", SIGGRAPH 2026).
ASSET_SEPARATION = {
    "layers": [
        "background",
        "hair_back",
        "body",
        "clothing",
        "head",
        "ear_L",
        "ear_R",
        "face",
        "brow_L",
        "brow_R",
        "eye_L",
        "eye_R",
        "nose",
        "mouth",
        "hair_front",
        "accessories",
    ],
    "use_ai_segmentation": _env_bool("KAMYII_AI_SEGMENTATION", True),
    # Segmentation backend preference order. Whatever is installed first wins.
    # "sam2" -> Meta Segment Anything 2, "rembg" -> background matting,
    # "heuristic" -> always-available geometric fallback.
    "backend_priority": ["sam2", "rembg", "heuristic"],
    # Inpaint occluded regions (head behind hair, body under clothes) so each
    # exported layer is complete — the key idea behind See-through decomposition.
    "inpaint_occluded": _env_bool("KAMYII_INPAINT_OCCLUDED", True),
    "sam2_checkpoint": os.getenv("KAMYII_SAM2_CHECKPOINT"),
    "sam2_model_cfg": os.getenv("KAMYII_SAM2_CONFIG", "sam2_hiera_l.yaml"),
}


# ---------------------------------------------------------------------------
# Live2D / model assembly
# ---------------------------------------------------------------------------
MODEL_ASSEMBLY = {
    "format": "live2d",  # "live2d" packages a Cubism-import-ready project
    "texture_size": _env_int("KAMYII_TEXTURE_SIZE", 4096),
    "include_physics": True,
    "export_psd": True,  # Layered PSD is the primary rig-ready artifact
    # Standard Cubism parameter set generated into the .cdi3.json so the artist
    # only needs to bind deformers in Cubism 5's AI auto-rig.
    "standard_parameters": [
        {"Id": "ParamAngleX", "Min": -30, "Max": 30, "Default": 0, "Name": "Angle X"},
        {"Id": "ParamAngleY", "Min": -30, "Max": 30, "Default": 0, "Name": "Angle Y"},
        {"Id": "ParamAngleZ", "Min": -30, "Max": 30, "Default": 0, "Name": "Angle Z"},
        {"Id": "ParamEyeLOpen", "Min": 0, "Max": 1, "Default": 1, "Name": "Eye L Open"},
        {"Id": "ParamEyeROpen", "Min": 0, "Max": 1, "Default": 1, "Name": "Eye R Open"},
        {"Id": "ParamEyeBallX", "Min": -1, "Max": 1, "Default": 0, "Name": "Eyeball X"},
        {"Id": "ParamEyeBallY", "Min": -1, "Max": 1, "Default": 0, "Name": "Eyeball Y"},
        {"Id": "ParamBrowLY", "Min": -1, "Max": 1, "Default": 0, "Name": "Brow L Y"},
        {"Id": "ParamBrowRY", "Min": -1, "Max": 1, "Default": 0, "Name": "Brow R Y"},
        {"Id": "ParamMouthForm", "Min": -1, "Max": 1, "Default": 0, "Name": "Mouth Form"},
        {"Id": "ParamMouthOpenY", "Min": 0, "Max": 1, "Default": 0, "Name": "Mouth Open"},
        {"Id": "ParamBodyAngleX", "Min": -10, "Max": 10, "Default": 0, "Name": "Body Angle X"},
        {"Id": "ParamBodyAngleY", "Min": -10, "Max": 10, "Default": 0, "Name": "Body Angle Y"},
        {"Id": "ParamBodyAngleZ", "Min": -10, "Max": 10, "Default": 0, "Name": "Body Angle Z"},
        {"Id": "ParamBreath", "Min": 0, "Max": 1, "Default": 0, "Name": "Breath"},
        {"Id": "ParamHairFront", "Min": -1, "Max": 1, "Default": 0, "Name": "Hair Front"},
        {"Id": "ParamHairBack", "Min": -1, "Max": 1, "Default": 0, "Name": "Hair Back"},
    ],
}


# ---------------------------------------------------------------------------
# Pipeline / API
# ---------------------------------------------------------------------------
PIPELINE = {
    "save_intermediate_steps": True,
    "cleanup_temp_files": False,
}

API = {
    "max_concurrent_generations": _env_int("KAMYII_MAX_CONCURRENT", 2),
    "timeout": _env_int("KAMYII_TIMEOUT", 600),
}

# Application metadata
APP_NAME = "Ka-myii"
APP_VERSION = "2.0.0"
APP_TAGLINE = "Automated VTuber Model Studio"
