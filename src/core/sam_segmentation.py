"""
Segment Anything Model (SAM) integration for superior asset segmentation
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np

from src.models.vtuber_model import Asset

logger = logging.getLogger(__name__)

try:
    # Try to import SAM
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class SAMSegmentator:
    """
    Advanced segmentation using Segment Anything Model (SAM)
    """

    def __init__(self, model_type: str = "vit_h", checkpoint_path: Optional[Path] = None):
        """
        Initialize SAM segmentator

        Args:
            model_type: SAM model type (vit_h, vit_l, vit_b)
            checkpoint_path: Path to SAM checkpoint file
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.sam_model = None
        self.mask_generator = None

        if SAM_AVAILABLE and checkpoint_path and checkpoint_path.exists():
            self._load_model()
        else:
            logger.warning("SAM not available or checkpoint not found")

    def _load_model(self):
        """Load SAM model"""
        try:
            logger.info(f"Loading SAM model: {self.model_type}")

            self.sam_model = sam_model_registry[self.model_type](
                checkpoint=str(self.checkpoint_path)
            )

            self.mask_generator = SamAutomaticMaskGenerator(self.sam_model)

            logger.info("SAM model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            raise

    def segment_image(
        self,
        image_path: Path,
        output_dir: Path,
        min_mask_region_area: int = 100
    ) -> List[Asset]:
        """
        Segment image into parts using SAM

        Args:
            image_path: Input image
            output_dir: Output directory for segments
            min_mask_region_area: Minimum area for a mask to be considered

        Returns:
            List of segmented assets
        """
        if not SAM_AVAILABLE or self.mask_generator is None:
            logger.warning("SAM not available, using fallback segmentation")
            return self._fallback_segmentation(image_path, output_dir)

        logger.info(f"Segmenting image with SAM: {image_path}")

        # Load image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)

        # Generate masks
        masks = self.mask_generator.generate(image_array)

        # Sort masks by area (largest first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        # Create assets from masks
        assets = []
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, mask_data in enumerate(masks[:20]):  # Limit to top 20 masks
            if mask_data['area'] < min_mask_region_area:
                continue

            # Extract mask
            mask = mask_data['segmentation']

            # Create masked image
            masked_image = self._apply_mask(image_array, mask)

            # Determine layer type based on position and size
            layer_type = self._classify_segment(
                mask,
                mask_data['bbox'],
                image_array.shape,
                i
            )

            # Save
            output_path = output_dir / f"{layer_type}_{i}.png"
            Image.fromarray(masked_image).save(output_path)

            assets.append(Asset(
                layer_type=layer_type,
                file_path=output_path,
                metadata={
                    "segmentation_method": "SAM",
                    "area": int(mask_data['area']),
                    "bbox": mask_data['bbox'],
                    "predicted_iou": float(mask_data.get('predicted_iou', 0))
                }
            ))

        logger.info(f"Generated {len(assets)} segments with SAM")
        return assets

    def _apply_mask(
        self,
        image_array: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Apply mask to image with transparency"""
        # Create RGBA image
        h, w = image_array.shape[:2]
        masked = np.zeros((h, w, 4), dtype=np.uint8)

        # Copy RGB channels
        masked[:, :, :3] = image_array

        # Set alpha channel based on mask
        masked[:, :, 3] = mask.astype(np.uint8) * 255

        return masked

    def _classify_segment(
        self,
        mask: np.ndarray,
        bbox: List[int],
        image_shape: Tuple,
        index: int
    ) -> str:
        """
        Classify segment into layer type based on position and size

        Args:
            mask: Segmentation mask
            bbox: Bounding box [x, y, w, h]
            image_shape: Shape of original image
            index: Segment index

        Returns:
            Layer type name
        """
        h, w = image_shape[:2]
        x, y, bw, bh = bbox

        # Calculate relative position
        center_y = (y + bh / 2) / h
        center_x = (x + bw / 2) / w
        area_ratio = (bw * bh) / (w * h)

        # Classification heuristics
        if index == 0 and area_ratio > 0.5:
            return "body"
        elif center_y < 0.4 and center_x > 0.4 and center_x < 0.6:
            return "head"
        elif center_y < 0.3:
            if center_x < 0.3 or center_x > 0.7:
                return "hair"
            else:
                return "hair_front"
        elif center_y > 0.3 and center_y < 0.5:
            if bw < w * 0.15:
                return "eyes"
            else:
                return "face_detail"
        elif center_y > 0.5 and center_y < 0.65:
            if bw < w * 0.2:
                return "mouth"
            else:
                return "clothing_upper"
        elif center_y > 0.65:
            return "clothing_lower"
        else:
            return f"part_{index}"

    def _fallback_segmentation(
        self,
        image_path: Path,
        output_dir: Path
    ) -> List[Asset]:
        """Fallback segmentation when SAM is not available"""
        from src.core.asset_separator import AssetSeparator

        logger.info("Using fallback asset separator")
        separator = AssetSeparator()
        return separator.separate(image_path, output_dir)

    def segment_by_labels(
        self,
        image_path: Path,
        output_dir: Path,
        target_labels: List[str]
    ) -> Dict[str, Path]:
        """
        Segment specific parts by label

        Args:
            image_path: Input image
            output_dir: Output directory
            target_labels: List of labels to segment (e.g., ['hair', 'face', 'body'])

        Returns:
            Dictionary mapping labels to output paths
        """
        # This would use a more advanced model with semantic segmentation
        # For now, use the automatic segmentation and try to match labels

        assets = self.segment_image(image_path, output_dir)

        result = {}
        for label in target_labels:
            # Find best matching asset
            matching_assets = [a for a in assets if label in a.layer_type.lower()]
            if matching_assets:
                result[label] = matching_assets[0].file_path

        return result


class BRIASegmentator:
    """
    Alternative segmentation using BRIA AI
    """

    def __init__(self):
        """Initialize BRIA segmentator"""
        logger.info("BRIASegmentator initialized")

    def remove_background(
        self,
        image_path: Path,
        output_path: Path,
        alpha_matting: bool = True
    ) -> Path:
        """
        Remove background using BRIA AI

        Args:
            image_path: Input image
            output_path: Output path
            alpha_matting: Whether to use alpha matting for better edges

        Returns:
            Path to output image
        """
        try:
            # Try using rembg which can use BRIA models
            from rembg import remove, new_session

            with open(image_path, 'rb') as i:
                input_data = i.read()

                # Create session with BRIA model if available
                try:
                    session = new_session("u2net")  # or "bria" if available
                    output_data = remove(input_data, session=session, alpha_matting=alpha_matting)
                except:
                    output_data = remove(input_data, alpha_matting=alpha_matting)

            with open(output_path, 'wb') as o:
                o.write(output_data)

            logger.info(f"Background removed: {output_path}")
            return output_path

        except ImportError:
            logger.warning("rembg not available for background removal")
            # Fallback: just copy the image
            from shutil import copy2
            copy2(image_path, output_path)
            return output_path
