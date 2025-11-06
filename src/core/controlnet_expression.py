"""
Enhanced expression generation with ControlNet support
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np

from src.core.expression_generator import ExpressionGenerator, Expression

logger = logging.getLogger(__name__)

try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from diffusers import UniPCMultistepScheduler
    import torch
    CONTROLNET_AVAILABLE = True
except ImportError:
    CONTROLNET_AVAILABLE = False


class ControlNetExpressionGenerator(ExpressionGenerator):
    """
    Enhanced expression generator using ControlNet for better expression control
    """

    def __init__(self, use_controlnet: bool = True, device: Optional[str] = None):
        """
        Initialize the ControlNet expression generator

        Args:
            use_controlnet: Whether to use ControlNet for generation
            device: Device to run on (cuda/cpu)
        """
        super().__init__(use_ai_generation=use_controlnet)
        self.use_controlnet = use_controlnet and CONTROLNET_AVAILABLE
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None

        if self.use_controlnet:
            logger.info("ControlNetExpressionGenerator initialized with ControlNet support")
        else:
            logger.info("ControlNetExpressionGenerator initialized (ControlNet unavailable)")

    def load_controlnet_model(
        self,
        base_model: str = "runwayml/stable-diffusion-v1-5",
        controlnet_model: str = "lllyasviel/sd-controlnet-canny"
    ):
        """Load ControlNet model for expression generation"""
        if not CONTROLNET_AVAILABLE:
            logger.warning("ControlNet libraries not available")
            return

        try:
            logger.info("Loading ControlNet model...")

            # Load ControlNet
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

            # Load pipeline
            self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                base_model,
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None
            )

            # Optimize
            self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            self.pipeline = self.pipeline.to(self.device)

            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                except:
                    pass

            logger.info("ControlNet model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load ControlNet: {e}")
            self.use_controlnet = False

    def generate_expression_with_controlnet(
        self,
        base_image: Image.Image,
        expression_name: str,
        prompt: str,
        output_path: Path,
        num_inference_steps: int = 20,
        controlnet_conditioning_scale: float = 0.5
    ) -> Path:
        """
        Generate expression variant using ControlNet

        Args:
            base_image: Base character image
            expression_name: Name of expression to generate
            prompt: Text prompt for the expression
            output_path: Where to save result
            num_inference_steps: Number of denoising steps
            controlnet_conditioning_scale: How much to follow the control

        Returns:
            Path to generated image
        """
        if not self.use_controlnet or self.pipeline is None:
            logger.warning("ControlNet not available, using base generator")
            return output_path

        try:
            # Prepare control image (canny edge detection)
            import cv2
            control_image = self._prepare_control_image(base_image)

            # Expression-specific prompts
            expression_prompts = {
                Expression.HAPPY: "smiling, happy expression, joyful eyes",
                Expression.SAD: "sad expression, teary eyes, frowning",
                Expression.ANGRY: "angry expression, furrowed brows, intense eyes",
                Expression.CRY: "crying, tears streaming, sad expression",
                Expression.SMUG: "smug smile, confident expression, half-closed eyes",
                Expression.HEART_EYES: "heart-shaped eyes, loving expression, blushing",
                Expression.SURPRISED: "surprised expression, wide eyes, open mouth",
                Expression.FRUSTRATED: "frustrated expression, annoyed, furrowed brows",
            }

            full_prompt = f"anime character portrait, {expression_prompts.get(expression_name, prompt)}, high quality, detailed"

            # Generate
            result = self.pipeline(
                prompt=full_prompt,
                image=control_image,
                num_inference_steps=num_inference_steps,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
            )

            generated_image = result.images[0]
            generated_image.save(output_path)

            logger.info(f"Generated expression with ControlNet: {expression_name}")
            return output_path

        except Exception as e:
            logger.error(f"ControlNet generation failed: {e}")
            raise

    def _prepare_control_image(self, image: Image.Image) -> Image.Image:
        """Prepare control image for ControlNet (Canny edge detection)"""
        try:
            import cv2

            # Convert to numpy array
            img_array = np.array(image)

            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Canny edge detection
            edges = cv2.Canny(gray, 100, 200)

            # Convert back to PIL
            control_image = Image.fromarray(edges)

            return control_image

        except Exception as e:
            logger.error(f"Failed to prepare control image: {e}")
            # Return original image as fallback
            return image.convert('L')

    def generate_expression_variations(
        self,
        base_image_path: Path,
        expression_name: str,
        output_dir: Path,
        num_variations: int = 3
    ) -> List[Path]:
        """
        Generate multiple variations of an expression

        Args:
            base_image_path: Path to base image
            expression_name: Expression to generate
            output_dir: Output directory
            num_variations: Number of variations to create

        Returns:
            List of paths to generated variations
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        variations = []

        base_image = Image.open(base_image_path)

        for i in range(num_variations):
            output_path = output_dir / f"{expression_name}_var{i+1}.png"

            if self.use_controlnet and self.pipeline is not None:
                # Use ControlNet
                self.generate_expression_with_controlnet(
                    base_image,
                    expression_name,
                    f"variation {i+1}",
                    output_path
                )
            else:
                # Use base generator
                assets = self._generate_single_expression(
                    base_image,
                    expression_name,
                    output_dir
                )
                if assets:
                    # Composite the expression
                    self.create_expression_preview(assets, base_image, output_path)

            variations.append(output_path)

        logger.info(f"Generated {len(variations)} variations for {expression_name}")
        return variations
