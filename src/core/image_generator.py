"""
Image generation module using Stable Diffusion or similar models
"""
import torch
from pathlib import Path
from typing import Optional, Dict
import logging

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

from src.models.vtuber_model import GenerationRequest

logger = logging.getLogger(__name__)


class ImageGenerator:
    """
    Handles image generation using Stable Diffusion models
    """

    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5", device: Optional[str] = None):
        """
        Initialize the image generator

        Args:
            model_name: Hugging Face model identifier
            device: Device to run on (cuda/cpu). Auto-detected if None
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None

        logger.info(f"ImageGenerator initialized with model: {model_name}, device: {self.device}")

    def load_model(self):
        """Load the Stable Diffusion model"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers library not found. Install with: pip install diffusers transformers accelerate"
            )

        logger.info(f"Loading model: {self.model_name}")

        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,  # Disable safety checker for speed
            )

            # Use DPM++ scheduler for better quality
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )

            self.pipeline = self.pipeline.to(self.device)

            # Enable memory optimizations
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
                # Optionally enable xformers for even better memory efficiency
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {e}")

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(
        self,
        request: GenerationRequest,
        output_path: Path,
    ) -> Path:
        """
        Generate an image based on the request

        Args:
            request: Generation request with parameters
            output_path: Path to save the generated image

        Returns:
            Path to the generated image
        """
        if self.pipeline is None:
            self.load_model()

        logger.info(f"Generating image with prompt: {request.prompt[:50]}...")

        # Enhance prompt for VTuber/anime style
        enhanced_prompt = self._enhance_prompt(request.prompt, request.style)

        try:
            # Generate image
            result = self.pipeline(
                prompt=enhanced_prompt,
                negative_prompt=request.negative_prompt or self._get_default_negative_prompt(),
                width=request.width,
                height=request.height,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(request.seed) if request.seed else None,
            )

            image = result.images[0]

            # Save image
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)

            logger.info(f"Image generated successfully: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise

    def _enhance_prompt(self, prompt: str, style: str) -> str:
        """
        Enhance the prompt with style-specific keywords

        Args:
            prompt: Original prompt
            style: Style identifier (anime, realistic, etc.)

        Returns:
            Enhanced prompt
        """
        style_enhancements = {
            "anime": "anime style, high quality, detailed, clean lines, vibrant colors",
            "live2d": "live2d style, anime character, front view, clean background, character design, reference sheet",
            "vtuber": "vtuber model, anime style, character design, front facing, clean background, high quality",
            "realistic": "realistic, high quality, detailed",
        }

        enhancement = style_enhancements.get(style.lower(), style_enhancements["anime"])
        return f"{prompt}, {enhancement}"

    def _get_default_negative_prompt(self) -> str:
        """Get default negative prompt for quality"""
        return (
            "lowres, bad anatomy, bad hands, text, error, missing fingers, "
            "extra digit, fewer digits, cropped, worst quality, low quality, "
            "normal quality, jpeg artifacts, signature, watermark, username, blurry, "
            "multiple views, side view, back view"
        )

    def unload_model(self):
        """Unload the model to free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded")


# For systems without GPU or for testing
class DummyImageGenerator(ImageGenerator):
    """Dummy generator for testing without actual model"""

    def __init__(self):
        super().__init__()
        logger.info("Using DummyImageGenerator (no actual generation)")

    def load_model(self):
        logger.info("DummyImageGenerator: Model 'loaded' (no-op)")

    def generate(self, request: GenerationRequest, output_path: Path) -> Path:
        """Create a placeholder image"""
        from PIL import Image, ImageDraw, ImageFont

        logger.info("DummyImageGenerator: Creating placeholder image")

        # Create a placeholder image
        img = Image.new('RGB', (request.width, request.height), color='lightblue')
        draw = ImageDraw.Draw(img)

        # Add text
        text = f"Placeholder\n{request.prompt[:30]}"
        draw.text((10, 10), text, fill='black')

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

        return output_path
