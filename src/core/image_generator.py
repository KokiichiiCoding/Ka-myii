"""
Image generation module using Stable Diffusion or similar models
"""
import torch
from pathlib import Path
from typing import Optional, Dict
import logging
import uuid

import config

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL
    DIFFUSERS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    StableDiffusionPipeline = None  # type: ignore[assignment]
    StableDiffusionXLPipeline = None  # type: ignore[assignment]
    AutoencoderKL = None  # type: ignore[assignment]
    DPMSolverMultistepScheduler = None  # type: ignore[assignment]
    DIFFUSERS_AVAILABLE = False

from src.models.vtuber_model import GenerationRequest

# Import progress tracking
try:
    from src.utils.progress_tracker import get_progress_manager, DiffusionProgressCallback
    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImageGenerator:
    """
    Handles image generation using Stable Diffusion models
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        pipeline: Optional[str] = None,
        custom_model_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        original_config_file: Optional[str] = None,
    ):
        """
        Initialize the image generator

        Args:
            model_name: Hugging Face model identifier
            device: Device to run on (cuda/cpu). Auto-detected if None
        """
        settings = getattr(config, "IMAGE_GENERATION", {})

        self.model_name = model_name or settings.get("model_name", "runwayml/stable-diffusion-v1-5")
        self.pipeline_type = (pipeline or settings.get("pipeline", "auto") or "auto").lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        custom_path = custom_model_path or settings.get("custom_model_path")
        self.custom_model_path = Path(custom_path).expanduser() if custom_path else None
        vae_override = vae_path or settings.get("vae_path")
        self.vae_path = Path(vae_override).expanduser() if vae_override else None
        self.original_config_file = (
            Path(original_config_file).expanduser()
            if original_config_file
            else (
                Path(settings.get("original_config_file")).expanduser()
                if settings.get("original_config_file")
                else None
            )
        )
        self.pipeline = None

        logger.info(
            "ImageGenerator initialized with model: %s (pipeline=%s, custom=%s), device: %s",
            self.model_name,
            self.pipeline_type,
            self.custom_model_path if self.custom_model_path else "<huggingface>",
            self.device,
        )

    def load_model(self):
        """Load the Stable Diffusion model"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers library not found. Install with: pip install diffusers transformers accelerate"
            )

        pipeline_cls = self._determine_pipeline_class()

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        if self.custom_model_path and not self.custom_model_path.exists():
            raise FileNotFoundError(f"Custom model not found at {self.custom_model_path}")

        if self.vae_path and not self.vae_path.exists():
            raise FileNotFoundError(f"Custom VAE not found at {self.vae_path}")

        if self.original_config_file and not self.original_config_file.exists():
            raise FileNotFoundError(
                f"Original config file not found at {self.original_config_file}"
            )

        logger.info(
            "Loading model using %s from %s",
            pipeline_cls.__name__,
            self.custom_model_path if self.custom_model_path else self.model_name,
        )

        try:
            load_kwargs: Dict[str, object] = {"torch_dtype": dtype}
            if pipeline_cls is StableDiffusionPipeline:
                load_kwargs["safety_checker"] = None  # type: ignore[index]

            if self.custom_model_path:
                load_kwargs["original_config_file"] = (
                    str(self.original_config_file) if self.original_config_file else None
                )
                # Remove None values to avoid diffusers complaining
                load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}
                self.pipeline = pipeline_cls.from_single_file(
                    str(self.custom_model_path),
                    **load_kwargs,  # type: ignore[arg-type]
                )
            else:
                load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}
                self.pipeline = pipeline_cls.from_pretrained(
                    self.model_name,
                    **load_kwargs,  # type: ignore[arg-type]
                )

            if DPMSolverMultistepScheduler and hasattr(self.pipeline, "scheduler"):
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(  # type: ignore[assignment]
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

            if self.vae_path:
                if AutoencoderKL is None:
                    raise ImportError(
                        "diffusers AutoencoderKL not available. Update diffusers to load custom VAE."
                    )
                logger.info("Loading custom VAE from %s", self.vae_path)
                vae = AutoencoderKL.from_pretrained(
                    str(self.vae_path), torch_dtype=dtype
                )
                self.pipeline.vae = vae.to(self.device)

            logger.info(
                "Model loaded successfully (%s)",
                self.custom_model_path if self.custom_model_path else self.model_name,
            )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(
        self,
        request: GenerationRequest,
        output_path: Path,
        task_id: Optional[str] = None,
    ) -> Path:
        """
        Generate an image based on the request

        Args:
            request: Generation request with parameters
            output_path: Path to save the generated image
            task_id: Optional task ID for progress tracking

        Returns:
            Path to the generated image
        """
        if self.pipeline is None:
            self.load_model()

        logger.info(f"Generating image with prompt: {request.prompt[:50]}...")

        # Enhance prompt for VTuber/anime style
        enhanced_prompt = self._enhance_prompt(request.prompt, request.style)

        # Create progress tracker if available
        tracker = None
        callback = None
        if PROGRESS_AVAILABLE:
            try:
                manager = get_progress_manager()
                if manager:
                    if not task_id:
                        task_id = f"gen_{uuid.uuid4().hex[:8]}"
                    tracker = manager.create_tracker(task_id, request.steps)
                    callback = DiffusionProgressCallback(tracker, request.steps)
                    logger.info(f"Progress tracking enabled for task: {task_id}")
            except Exception as e:
                logger.warning(f"Could not initialize progress tracking: {e}")

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
                callback=callback if callback else None,
                callback_steps=1 if callback else None,
            )

            image = result.images[0]

            # Save image
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)

            # Mark as complete
            if tracker:
                tracker.complete("Image generated successfully")

            logger.info(f"Image generated successfully: {output_path}")
            return output_path

        except InterruptedError:
            # User cancelled generation
            if tracker:
                tracker.complete("Generation cancelled", status="cancelled")
            logger.info("Image generation cancelled by user")
            raise
        except Exception as e:
            # Mark as failed
            if tracker:
                tracker.complete(f"Generation failed: {str(e)}", status="failed")
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

    def _determine_pipeline_class(self):
        """Determine the diffusers pipeline class to use"""
        pipeline_choice = self.pipeline_type

        # Auto-detect XL models from paths/names if pipeline not explicitly set
        if pipeline_choice == "auto":
            source_name = (
                self.custom_model_path.name if self.custom_model_path else self.model_name
            )
            if source_name and "xl" in source_name.lower():
                pipeline_choice = "sdxl"
            else:
                pipeline_choice = "sd15"

        if pipeline_choice == "sdxl":
            if StableDiffusionXLPipeline is None:
                raise ImportError(
                    "StableDiffusionXLPipeline not available. Install diffusers>=0.19.0 for SDXL support."
                )
            return StableDiffusionXLPipeline

        # Default to SD 1.5 style pipeline
        if StableDiffusionPipeline is None:
            raise ImportError(
                "StableDiffusionPipeline not available. Install diffusers for Stable Diffusion support."
            )
        return StableDiffusionPipeline


# For systems without GPU or for testing
class DummyImageGenerator(ImageGenerator):
    """Dummy generator for testing without actual model"""

    def __init__(self):
        super().__init__()
        logger.info("Using DummyImageGenerator (no actual generation)")

    def load_model(self):
        logger.info("DummyImageGenerator: Model 'loaded' (no-op)")

    def generate(
        self, request: GenerationRequest, output_path: Path, task_id: Optional[str] = None
    ) -> Path:
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
