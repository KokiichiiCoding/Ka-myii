"""
Asset separation module for splitting images into layers
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

from src.models.vtuber_model import Asset

logger = logging.getLogger(__name__)


class AssetSeparator:
    """
    Separates a character image into layered assets for Live2D
    """

    def __init__(self, use_ai_segmentation: bool = True):
        """
        Initialize the asset separator

        Args:
            use_ai_segmentation: Whether to use AI-based segmentation
        """
        self.use_ai_segmentation = use_ai_segmentation
        logger.info(f"AssetSeparator initialized (AI segmentation: {use_ai_segmentation})")

    def separate(
        self,
        image_path: Path,
        output_dir: Path,
        layers: Optional[List[str]] = None
    ) -> List[Asset]:
        """
        Separate an image into multiple layers

        Args:
            image_path: Path to the source image
            output_dir: Directory to save separated assets
            layers: List of layer types to generate

        Returns:
            List of generated assets
        """
        logger.info(f"Separating image: {image_path}")

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Load image
        image = Image.open(image_path)

        # Default layers if not specified
        if layers is None:
            layers = [
                "background",
                "body",
                "head",
                "eyes",
                "mouth",
                "hair_back",
                "hair_front"
            ]

        assets = []

        # Step 1: Remove background
        character_image = self._remove_background(image)

        # Step 2: Generate base layers
        assets.extend(self._generate_base_layers(character_image, output_dir, layers))

        # Step 3: Generate facial feature layers (eyes, mouth)
        if "eyes" in layers or "mouth" in layers:
            assets.extend(self._generate_facial_features(character_image, output_dir, layers))

        logger.info(f"Generated {len(assets)} assets")
        return assets

    def _remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove background from image

        Args:
            image: Source image

        Returns:
            Image with transparent background
        """
        logger.info("Removing background")

        if REMBG_AVAILABLE and self.use_ai_segmentation:
            try:
                # Use rembg for AI-based background removal
                result = remove(image)
                logger.info("Background removed using AI")
                return result
            except Exception as e:
                logger.warning(f"AI background removal failed: {e}, falling back to simple method")

        # Fallback: Simple alpha channel
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        logger.info("Background removal skipped (using original image)")
        return image

    def _generate_base_layers(
        self,
        image: Image.Image,
        output_dir: Path,
        layers: List[str]
    ) -> List[Asset]:
        """
        Generate base layers (body, head, hair, etc.)

        Args:
            image: Source image with transparent background
            output_dir: Output directory
            layers: List of layers to generate

        Returns:
            List of generated assets
        """
        assets = []

        # For now, create the basic structure
        # In a full implementation, this would use segmentation models
        # to intelligently separate different parts

        base_layers = {
            "background": self._create_background_layer,
            "body": self._create_body_layer,
            "head": self._create_head_layer,
            "hair_back": self._create_hair_layer,
            "hair_front": self._create_hair_layer,
            "clothing": self._create_body_layer,
            "accessories": self._create_accessory_layer,
        }

        for layer_name in layers:
            if layer_name in base_layers:
                try:
                    layer_image = base_layers[layer_name](image, layer_name)
                    layer_path = output_dir / f"{layer_name}.png"
                    layer_image.save(layer_path)

                    assets.append(Asset(
                        layer_type=layer_name,
                        file_path=layer_path,
                        metadata={"generated": True}
                    ))
                    logger.info(f"Generated layer: {layer_name}")
                except Exception as e:
                    logger.warning(f"Failed to generate layer {layer_name}: {e}")

        return assets

    def _generate_facial_features(
        self,
        image: Image.Image,
        output_dir: Path,
        layers: List[str]
    ) -> List[Asset]:
        """
        Generate facial feature layers (eyes, mouth, expressions)

        Args:
            image: Source image
            output_dir: Output directory
            layers: List of layers to generate

        Returns:
            List of generated assets
        """
        assets = []

        # For eyes and mouth, we need to detect facial features
        # This is a placeholder - would use face detection in production

        if "eyes" in layers:
            eyes_variants = ["eyes_open", "eyes_closed", "eyes_happy"]
            for variant in eyes_variants:
                layer_path = output_dir / f"{variant}.png"
                eyes_image = self._create_eyes_layer(image, variant)
                eyes_image.save(layer_path)

                assets.append(Asset(
                    layer_type="eyes",
                    file_path=layer_path,
                    metadata={"variant": variant}
                ))

        if "mouth" in layers:
            mouth_variants = ["mouth_closed", "mouth_open", "mouth_smile"]
            for variant in mouth_variants:
                layer_path = output_dir / f"{variant}.png"
                mouth_image = self._create_mouth_layer(image, variant)
                mouth_image.save(layer_path)

                assets.append(Asset(
                    layer_type="mouth",
                    file_path=layer_path,
                    metadata={"variant": variant}
                ))

        return assets

    # Layer creation methods (placeholders for now)
    def _create_background_layer(self, image: Image.Image, name: str) -> Image.Image:
        """Create a background layer"""
        # Create a simple colored background
        bg = Image.new('RGBA', image.size, (255, 255, 255, 0))
        return bg

    def _create_body_layer(self, image: Image.Image, name: str) -> Image.Image:
        """Create body layer"""
        # In production, would segment the body region
        return image.copy()

    def _create_head_layer(self, image: Image.Image, name: str) -> Image.Image:
        """Create head layer"""
        # In production, would segment the head region
        return image.copy()

    def _create_hair_layer(self, image: Image.Image, name: str) -> Image.Image:
        """Create hair layer"""
        # In production, would segment hair region
        return image.copy()

    def _create_accessory_layer(self, image: Image.Image, name: str) -> Image.Image:
        """Create accessory layer"""
        return Image.new('RGBA', image.size, (0, 0, 0, 0))

    def _create_eyes_layer(self, image: Image.Image, variant: str) -> Image.Image:
        """Create eyes layer with variant"""
        # Placeholder - would detect and extract eye region
        return Image.new('RGBA', image.size, (0, 0, 0, 0))

    def _create_mouth_layer(self, image: Image.Image, variant: str) -> Image.Image:
        """Create mouth layer with variant"""
        # Placeholder - would detect and extract mouth region
        return Image.new('RGBA', image.size, (0, 0, 0, 0))


class DummyAssetSeparator(AssetSeparator):
    """Dummy separator for testing"""

    def separate(
        self,
        image_path: Path,
        output_dir: Path,
        layers: Optional[List[str]] = None
    ) -> List[Asset]:
        """Create dummy assets"""
        logger.info("DummyAssetSeparator: Creating placeholder assets")

        output_dir.mkdir(parents=True, exist_ok=True)

        if layers is None:
            layers = ["body", "head", "eyes", "mouth"]

        assets = []
        image = Image.open(image_path)

        for layer in layers:
            layer_path = output_dir / f"{layer}.png"
            # Just copy the original image as placeholder
            image.save(layer_path)

            assets.append(Asset(
                layer_type=layer,
                file_path=layer_path,
                metadata={"dummy": True}
            ))

        return assets
