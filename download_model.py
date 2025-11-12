#!/usr/bin/env python3
"""
Model Download Script
Downloads the Stable Diffusion model before first use
"""
import sys
import torch
from pathlib import Path

import config

def download_model():
    """Download the Stable Diffusion model"""
    print("=" * 60)
    print("Ka-myii Model Download Script")
    print("=" * 60)
    print()

    # Check PyTorch
    print("1. Checking PyTorch installation...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Check diffusers
    print("2. Checking diffusers library...")
    try:
        from diffusers import StableDiffusionPipeline
        print("   ✓ diffusers is installed")
    except ImportError:
        print("   ✗ diffusers not found!")
        print("   Installing diffusers...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "diffusers", "transformers", "accelerate"])
        from diffusers import StableDiffusionPipeline
        print("   ✓ diffusers installed")
    print()

    # Download model
    settings = config.IMAGE_GENERATION
    custom_checkpoint = settings.get("custom_model_path")
    model_name = settings.get("model_name", "runwayml/stable-diffusion-v1-5")

    if custom_checkpoint:
        print("3. Custom checkpoint configured")
        print(f"   Path: {custom_checkpoint}")
        print("   Skipping automatic download. Place the file at the path above.")
        print()
        return Path(custom_checkpoint).exists()

    print("3. Downloading Stable Diffusion model...")
    print(f"   Model: {model_name}")
    print("   Size: ~4-5 GB")
    print("   This may take 10-30 minutes depending on your internet speed...")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        print("   Downloading... (this will show progress)")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
        )
        print()
        print("   ✓ Model downloaded successfully!")
        print()

        # Test the model
        print("4. Testing model...")
        pipeline = pipeline.to(device)
        print(f"   ✓ Model loaded on {device}")
        print()

        print("=" * 60)
        print("SUCCESS! Model is ready to use.")
        print("=" * 60)
        print()
        print("You can now run the application:")
        print("  Windows: run.bat")
        print("  Linux/Mac: python app.py")
        print()

        return True

    except Exception as e:
        print()
        print("=" * 60)
        print("ERROR: Model download failed")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        print("Possible solutions:")
        print("1. Check your internet connection")
        print("2. Try again (the download may have been interrupted)")
        print("3. Use dummy mode for testing: python app.py --dummy")
        print("4. Manually download the model using:")
        print("   from diffusers import StableDiffusionPipeline")
        print("   pipeline = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')")
        print()
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
