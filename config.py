"""
Configuration settings for Ka-myii
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Directories
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Flask settings
FLASK_ENV = os.getenv("FLASK_ENV", "development")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
DEBUG = FLASK_ENV == "development"

# Image Generation settings
IMAGE_GENERATION = {
    "default_width": 512,
    "default_height": 512,
    "default_steps": 30,
    "default_guidance_scale": 7.5,
    "model_name": "runwayml/stable-diffusion-v1-5",  # Can be changed to other models
}

# Asset Separation settings
ASSET_SEPARATION = {
    "layers": [
        "background",
        "body",
        "head",
        "eyes",
        "mouth",
        "hair_back",
        "hair_front",
        "accessories",
        "clothing"
    ],
    "use_ai_segmentation": True,
}

# Live2D/Model Assembly settings
MODEL_ASSEMBLY = {
    "format": "live2d",  # Could support other formats in future
    "texture_size": 2048,
    "include_physics": True,
}

# Pipeline settings
PIPELINE = {
    "save_intermediate_steps": True,
    "cleanup_temp_files": False,  # Keep for debugging
}

# Available pipeline presets that combine optional modules together.
# The UI exposes these so users can switch between a lightweight dummy
# flow, the default diffusers pipeline, and the new "ultimate" preset that
# layers SAM segmentation, ControlNet assisted expressions, and automatic
# physics helpers.
PIPELINE_PROFILES = {
    "lightweight": {
        "label": "Lightweight (Dummy)",
        "description": "Uses all dummy components for rapid UI previews without GPU dependencies.",
        "use_dummy": True,
        "segmentation": "basic",
        "controlnet": False,
        "auto_physics": False,
        "generate_expressions": False,
        "generate_accessories": False,
    },
    "standard": {
        "label": "Standard Diffusion",
        "description": "Stable Diffusion for the base image with rule-based layer separation and Live2D assembly.",
        "use_dummy": False,
        "segmentation": "basic",
        "controlnet": False,
        "auto_physics": True,
        "generate_expressions": True,
        "generate_accessories": False,
    },
    "enhanced": {
        "label": "Ultimate VTuber (SAM + ControlNet)",
        "description": (
            "High fidelity preset that fuses SAM segmentation, accessory generation, "
            "ControlNet expressions, and automatic physics."
        ),
        "use_dummy": False,
        "segmentation": "sam",
        "controlnet": True,
        "auto_physics": True,
        "generate_expressions": True,
        "generate_accessories": True,
    },
}

# Default expression and accessory shortcuts used when the UI requests
# automatic batches without providing explicit selections.
DEFAULT_EXPRESSIONS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "smug",
]

DEFAULT_ACCESSORIES = [
    "cat_ears",
    "halo",
    "glasses",
]

# API settings
API = {
    "max_concurrent_generations": 3,
    "timeout": 300,  # seconds
}
