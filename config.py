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

# API settings
API = {
    "max_concurrent_generations": 3,
    "timeout": 300,  # seconds
}
