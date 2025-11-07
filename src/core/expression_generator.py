"""
Expression generation module for VTuber emotions
Generates different facial expressions and emotion variants
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from src.models.vtuber_model import Asset

logger = logging.getLogger(__name__)


class Expression:
    """Represents a facial expression"""

    # Standard VTuber expressions
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    FEARFUL = "fearful"

    # Popular VTuber-specific expressions
    CRY = "cry"
    FRUSTRATED = "frustrated"
    SMUG = "smug"
    HEART_EYES = "heart_eyes"
    BLUSH = "blush"
    SHOCKED = "shocked"
    SLEEPY = "sleepy"
    CONFUSED = "confused"
    EMBARRASSED = "embarrassed"
    DETERMINED = "determined"
    POUTY = "pouty"
    WINK = "wink"
    WORRIED = "worried"
    EXCITED = "excited"

    # Eye states
    EYES_OPEN = "eyes_open"
    EYES_CLOSED = "eyes_closed"
    EYES_HALF = "eyes_half"
    EYES_HAPPY = "eyes_happy"
    EYES_ANGRY = "eyes_angry"
    EYES_SAD = "eyes_sad"
    EYES_SURPRISED = "eyes_surprised"
    EYES_HEART = "eyes_heart"
    EYES_SPARKLE = "eyes_sparkle"
    EYES_CRYING = "eyes_crying"

    # Mouth states
    MOUTH_NEUTRAL = "mouth_neutral"
    MOUTH_SMILE = "mouth_smile"
    MOUTH_OPEN = "mouth_open"
    MOUTH_FROWN = "mouth_frown"
    MOUTH_POUT = "mouth_pout"
    MOUTH_SMUG = "mouth_smug"
    MOUTH_WORRIED = "mouth_worried"
    MOUTH_SHOCKED = "mouth_shocked"

    # Eyebrow states
    BROW_NEUTRAL = "brow_neutral"
    BROW_RAISED = "brow_raised"
    BROW_FURROWED = "brow_furrowed"
    BROW_SAD = "brow_sad"
    BROW_ANGRY = "brow_angry"


class ExpressionSet:
    """A complete set of expressions for a character"""

    def __init__(self, base_image_path: Path):
        self.base_image_path = base_image_path
        self.expressions: Dict[str, List[Asset]] = {}

    def add_expression(self, expression_name: str, assets: List[Asset]):
        """Add an expression to the set"""
        self.expressions[expression_name] = assets

    def get_expression(self, expression_name: str) -> Optional[List[Asset]]:
        """Get assets for a specific expression"""
        return self.expressions.get(expression_name)

    def list_expressions(self) -> List[str]:
        """List all available expressions"""
        return list(self.expressions.keys())


class ExpressionGenerator:
    """
    Generates facial expressions and emotion variants for VTuber models
    """

    # Expression definitions with component states
    EXPRESSION_DEFINITIONS = {
        Expression.NEUTRAL: {
            "eyes": Expression.EYES_OPEN,
            "mouth": Expression.MOUTH_NEUTRAL,
            "brows": Expression.BROW_NEUTRAL,
            "blush": 0.0,
            "tears": 0.0,
        },
        Expression.HAPPY: {
            "eyes": Expression.EYES_HAPPY,
            "mouth": Expression.MOUTH_SMILE,
            "brows": Expression.BROW_RAISED,
            "blush": 0.2,
            "tears": 0.0,
        },
        Expression.SAD: {
            "eyes": Expression.EYES_SAD,
            "mouth": Expression.MOUTH_FROWN,
            "brows": Expression.BROW_SAD,
            "blush": 0.0,
            "tears": 0.0,
        },
        Expression.ANGRY: {
            "eyes": Expression.EYES_ANGRY,
            "mouth": Expression.MOUTH_FROWN,
            "brows": Expression.BROW_ANGRY,
            "blush": 0.3,
            "tears": 0.0,
        },
        Expression.SURPRISED: {
            "eyes": Expression.EYES_SURPRISED,
            "mouth": Expression.MOUTH_SHOCKED,
            "brows": Expression.BROW_RAISED,
            "blush": 0.1,
            "tears": 0.0,
        },
        Expression.CRY: {
            "eyes": Expression.EYES_CRYING,
            "mouth": Expression.MOUTH_FROWN,
            "brows": Expression.BROW_SAD,
            "blush": 0.2,
            "tears": 1.0,
        },
        Expression.FRUSTRATED: {
            "eyes": Expression.EYES_ANGRY,
            "mouth": Expression.MOUTH_POUT,
            "brows": Expression.BROW_FURROWED,
            "blush": 0.4,
            "tears": 0.0,
        },
        Expression.SMUG: {
            "eyes": Expression.EYES_HALF,
            "mouth": Expression.MOUTH_SMUG,
            "brows": Expression.BROW_RAISED,
            "blush": 0.0,
            "tears": 0.0,
        },
        Expression.HEART_EYES: {
            "eyes": Expression.EYES_HEART,
            "mouth": Expression.MOUTH_SMILE,
            "brows": Expression.BROW_NEUTRAL,
            "blush": 0.6,
            "tears": 0.0,
        },
        Expression.BLUSH: {
            "eyes": Expression.EYES_OPEN,
            "mouth": Expression.MOUTH_NEUTRAL,
            "brows": Expression.BROW_NEUTRAL,
            "blush": 0.8,
            "tears": 0.0,
        },
        Expression.SHOCKED: {
            "eyes": Expression.EYES_SURPRISED,
            "mouth": Expression.MOUTH_SHOCKED,
            "brows": Expression.BROW_RAISED,
            "blush": 0.0,
            "tears": 0.0,
        },
        Expression.SLEEPY: {
            "eyes": Expression.EYES_HALF,
            "mouth": Expression.MOUTH_NEUTRAL,
            "brows": Expression.BROW_NEUTRAL,
            "blush": 0.0,
            "tears": 0.0,
        },
        Expression.EMBARRASSED: {
            "eyes": Expression.EYES_CLOSED,
            "mouth": Expression.MOUTH_SMILE,
            "brows": Expression.BROW_SAD,
            "blush": 0.9,
            "tears": 0.0,
        },
        Expression.DETERMINED: {
            "eyes": Expression.EYES_OPEN,
            "mouth": Expression.MOUTH_NEUTRAL,
            "brows": Expression.BROW_FURROWED,
            "blush": 0.0,
            "tears": 0.0,
        },
        Expression.POUTY: {
            "eyes": Expression.EYES_HALF,
            "mouth": Expression.MOUTH_POUT,
            "brows": Expression.BROW_FURROWED,
            "blush": 0.3,
            "tears": 0.0,
        },
        Expression.WINK: {
            "eyes": Expression.EYES_HALF,
            "mouth": Expression.MOUTH_SMILE,
            "brows": Expression.BROW_NEUTRAL,
            "blush": 0.1,
            "tears": 0.0,
        },
        Expression.WORRIED: {
            "eyes": Expression.EYES_SAD,
            "mouth": Expression.MOUTH_WORRIED,
            "brows": Expression.BROW_SAD,
            "blush": 0.0,
            "tears": 0.0,
        },
        Expression.EXCITED: {
            "eyes": Expression.EYES_SPARKLE,
            "mouth": Expression.MOUTH_SMILE,
            "brows": Expression.BROW_RAISED,
            "blush": 0.3,
            "tears": 0.0,
        },
    }

    def __init__(self, use_ai_generation: bool = False):
        """
        Initialize the expression generator

        Args:
            use_ai_generation: Whether to use AI for expression generation
        """
        self.use_ai_generation = use_ai_generation
        logger.info(f"ExpressionGenerator initialized (AI: {use_ai_generation})")

    def generate_expression_set(
        self,
        base_image_path: Path,
        output_dir: Path,
        expressions: Optional[List[str]] = None
    ) -> ExpressionSet:
        """
        Generate a complete set of expressions

        Args:
            base_image_path: Path to the base character image
            output_dir: Directory to save expressions
            expressions: List of expression names to generate (None = all)

        Returns:
            ExpressionSet with all generated expressions
        """
        logger.info(f"Generating expression set for: {base_image_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Use all expressions if not specified
        if expressions is None:
            expressions = list(self.EXPRESSION_DEFINITIONS.keys())

        expression_set = ExpressionSet(base_image_path)
        base_image = Image.open(base_image_path)

        for expression_name in expressions:
            if expression_name not in self.EXPRESSION_DEFINITIONS:
                logger.warning(f"Unknown expression: {expression_name}")
                continue

            logger.info(f"Generating expression: {expression_name}")

            assets = self._generate_single_expression(
                base_image,
                expression_name,
                output_dir
            )

            expression_set.add_expression(expression_name, assets)

        logger.info(f"Generated {len(expressions)} expressions")
        return expression_set

    def _generate_single_expression(
        self,
        base_image: Image.Image,
        expression_name: str,
        output_dir: Path
    ) -> List[Asset]:
        """Generate assets for a single expression"""

        expression_def = self.EXPRESSION_DEFINITIONS[expression_name]
        assets = []

        # Create expression directory
        expr_dir = output_dir / expression_name
        expr_dir.mkdir(exist_ok=True)

        # Generate eyes
        eyes_asset = self._generate_eyes(
            base_image,
            expression_def["eyes"],
            expr_dir / f"eyes_{expression_name}.png"
        )
        if eyes_asset:
            assets.append(eyes_asset)

        # Generate mouth
        mouth_asset = self._generate_mouth(
            base_image,
            expression_def["mouth"],
            expr_dir / f"mouth_{expression_name}.png"
        )
        if mouth_asset:
            assets.append(mouth_asset)

        # Generate eyebrows
        brow_asset = self._generate_eyebrows(
            base_image,
            expression_def["brows"],
            expr_dir / f"brows_{expression_name}.png"
        )
        if brow_asset:
            assets.append(brow_asset)

        # Generate blush if needed
        if expression_def["blush"] > 0:
            blush_asset = self._generate_blush(
                base_image,
                expression_def["blush"],
                expr_dir / f"blush_{expression_name}.png"
            )
            if blush_asset:
                assets.append(blush_asset)

        # Generate tears if needed
        if expression_def["tears"] > 0:
            tears_asset = self._generate_tears(
                base_image,
                expression_def["tears"],
                expr_dir / f"tears_{expression_name}.png"
            )
            if tears_asset:
                assets.append(tears_asset)

        return assets

    def _generate_eyes(
        self,
        base_image: Image.Image,
        eye_state: str,
        output_path: Path
    ) -> Optional[Asset]:
        """Generate eyes for a specific state"""

        # Create a new transparent image
        eyes = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(eyes)

        # Calculate eye positions (center of image, approximate)
        width, height = base_image.size
        eye_y = int(height * 0.4)
        left_eye_x = int(width * 0.35)
        right_eye_x = int(width * 0.65)
        eye_size = int(width * 0.08)

        # Draw eyes based on state
        if "closed" in eye_state:
            # Closed eyes - horizontal lines
            for eye_x in [left_eye_x, right_eye_x]:
                draw.line(
                    [(eye_x - eye_size, eye_y), (eye_x + eye_size, eye_y)],
                    fill=(0, 0, 0, 255),
                    width=3
                )
        elif "half" in eye_state or "sleepy" in eye_state:
            # Half-closed eyes
            for eye_x in [left_eye_x, right_eye_x]:
                draw.ellipse(
                    [eye_x - eye_size, eye_y, eye_x + eye_size, eye_y + eye_size],
                    fill=(255, 255, 255, 255),
                    outline=(0, 0, 0, 255),
                    width=2
                )
        elif "heart" in eye_state:
            # Heart-shaped eyes
            for eye_x in [left_eye_x, right_eye_x]:
                self._draw_heart(draw, eye_x, eye_y, eye_size, (255, 100, 150, 255))
        elif "sparkle" in eye_state:
            # Sparkly eyes
            for eye_x in [left_eye_x, right_eye_x]:
                draw.ellipse(
                    [eye_x - eye_size, eye_y - eye_size, eye_x + eye_size, eye_y + eye_size],
                    fill=(255, 255, 255, 255),
                    outline=(0, 0, 0, 255),
                    width=2
                )
                # Add sparkle
                self._draw_sparkle(draw, eye_x, eye_y, eye_size // 2)
        else:
            # Default open eyes
            for eye_x in [left_eye_x, right_eye_x]:
                draw.ellipse(
                    [eye_x - eye_size, eye_y - eye_size, eye_x + eye_size, eye_y + eye_size],
                    fill=(255, 255, 255, 255),
                    outline=(0, 0, 0, 255),
                    width=2
                )
                # Pupil
                pupil_size = eye_size // 2
                draw.ellipse(
                    [eye_x - pupil_size, eye_y - pupil_size, eye_x + pupil_size, eye_y + pupil_size],
                    fill=(50, 50, 50, 255)
                )

        eyes.save(output_path)

        return Asset(
            layer_type=f"eyes_{eye_state}",
            file_path=output_path,
            metadata={"expression_component": "eyes", "state": eye_state}
        )

    def _generate_mouth(
        self,
        base_image: Image.Image,
        mouth_state: str,
        output_path: Path
    ) -> Optional[Asset]:
        """Generate mouth for a specific state"""

        mouth = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(mouth)

        width, height = base_image.size
        mouth_x = width // 2
        mouth_y = int(height * 0.6)
        mouth_width = int(width * 0.15)

        if "smile" in mouth_state:
            # Smiling mouth - upward arc
            draw.arc(
                [mouth_x - mouth_width, mouth_y - 10, mouth_x + mouth_width, mouth_y + 20],
                start=0, end=180,
                fill=(0, 0, 0, 255),
                width=3
            )
        elif "frown" in mouth_state:
            # Frowning mouth - downward arc
            draw.arc(
                [mouth_x - mouth_width, mouth_y - 20, mouth_x + mouth_width, mouth_y + 10],
                start=180, end=360,
                fill=(0, 0, 0, 255),
                width=3
            )
        elif "open" in mouth_state or "shocked" in mouth_state:
            # Open mouth - ellipse
            draw.ellipse(
                [mouth_x - mouth_width//2, mouth_y - 10, mouth_x + mouth_width//2, mouth_y + 10],
                fill=(50, 0, 0, 255),
                outline=(0, 0, 0, 255),
                width=2
            )
        elif "pout" in mouth_state:
            # Pouty mouth - small circle
            draw.ellipse(
                [mouth_x - mouth_width//3, mouth_y - 5, mouth_x + mouth_width//3, mouth_y + 5],
                fill=(255, 150, 150, 255),
                outline=(0, 0, 0, 255),
                width=2
            )
        elif "smug" in mouth_state:
            # Smug smile - slight upward curve on one side
            draw.arc(
                [mouth_x - mouth_width, mouth_y, mouth_x + mouth_width//2, mouth_y + 15],
                start=0, end=180,
                fill=(0, 0, 0, 255),
                width=2
            )
        else:
            # Neutral mouth - simple line
            draw.line(
                [(mouth_x - mouth_width, mouth_y), (mouth_x + mouth_width, mouth_y)],
                fill=(0, 0, 0, 255),
                width=2
            )

        mouth.save(output_path)

        return Asset(
            layer_type=f"mouth_{mouth_state}",
            file_path=output_path,
            metadata={"expression_component": "mouth", "state": mouth_state}
        )

    def _generate_eyebrows(
        self,
        base_image: Image.Image,
        brow_state: str,
        output_path: Path
    ) -> Optional[Asset]:
        """Generate eyebrows for a specific state"""

        brows = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(brows)

        width, height = base_image.size
        brow_y = int(height * 0.35)
        left_brow_x = int(width * 0.35)
        right_brow_x = int(width * 0.65)
        brow_width = int(width * 0.08)

        for brow_x in [left_brow_x, right_brow_x]:
            if "raised" in brow_state:
                # Raised eyebrows - slight upward curve
                y_offset = -5
                draw.arc(
                    [brow_x - brow_width, brow_y + y_offset - 5, brow_x + brow_width, brow_y + y_offset + 5],
                    start=180, end=360,
                    fill=(0, 0, 0, 255),
                    width=3
                )
            elif "furrowed" in brow_state or "angry" in brow_state:
                # Furrowed/angry eyebrows - angled down toward center
                angle = 20 if brow_x < width // 2 else -20
                draw.line(
                    [(brow_x - brow_width, brow_y), (brow_x + brow_width, brow_y - angle)],
                    fill=(0, 0, 0, 255),
                    width=3
                )
            elif "sad" in brow_state:
                # Sad eyebrows - angled up toward center
                angle = -15 if brow_x < width // 2 else 15
                draw.line(
                    [(brow_x - brow_width, brow_y), (brow_x + brow_width, brow_y + angle)],
                    fill=(0, 0, 0, 255),
                    width=3
                )
            else:
                # Neutral eyebrows - straight line
                draw.line(
                    [(brow_x - brow_width, brow_y), (brow_x + brow_width, brow_y)],
                    fill=(0, 0, 0, 255),
                    width=3
                )

        brows.save(output_path)

        return Asset(
            layer_type=f"brows_{brow_state}",
            file_path=output_path,
            metadata={"expression_component": "brows", "state": brow_state}
        )

    def _generate_blush(
        self,
        base_image: Image.Image,
        intensity: float,
        output_path: Path
    ) -> Optional[Asset]:
        """Generate blush overlay"""

        blush = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(blush)

        width, height = base_image.size
        blush_y = int(height * 0.5)
        left_blush_x = int(width * 0.3)
        right_blush_x = int(width * 0.7)
        blush_size = int(width * 0.1)

        alpha = int(255 * intensity * 0.5)  # Max 50% opacity

        for blush_x in [left_blush_x, right_blush_x]:
            draw.ellipse(
                [blush_x - blush_size, blush_y - blush_size//2,
                 blush_x + blush_size, blush_y + blush_size//2],
                fill=(255, 150, 150, alpha)
            )

        blush.save(output_path)

        return Asset(
            layer_type="blush",
            file_path=output_path,
            metadata={"expression_component": "blush", "intensity": intensity}
        )

    def _generate_tears(
        self,
        base_image: Image.Image,
        intensity: float,
        output_path: Path
    ) -> Optional[Asset]:
        """Generate tears"""

        tears = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(tears)

        width, height = base_image.size
        eye_y = int(height * 0.45)
        left_eye_x = int(width * 0.35)
        right_eye_x = int(width * 0.65)

        alpha = int(255 * intensity)

        for eye_x in [left_eye_x, right_eye_x]:
            # Draw teardrop
            tear_y = eye_y + int(height * 0.05)
            draw.ellipse(
                [eye_x - 5, tear_y, eye_x + 5, tear_y + 15],
                fill=(150, 200, 255, alpha),
                outline=(100, 150, 255, alpha),
                width=1
            )
            # Highlight
            draw.ellipse(
                [eye_x - 2, tear_y + 2, eye_x + 1, tear_y + 5],
                fill=(255, 255, 255, int(alpha * 0.8))
            )

        tears.save(output_path)

        return Asset(
            layer_type="tears",
            file_path=output_path,
            metadata={"expression_component": "tears", "intensity": intensity}
        )

    def _draw_heart(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int, int]):
        """Draw a heart shape"""
        # Simple heart approximation using circles and polygon
        draw.ellipse([x - size, y - size//2, x, y + size//2], fill=color)
        draw.ellipse([x, y - size//2, x + size, y + size//2], fill=color)
        draw.polygon([(x - size, y), (x, y + size), (x + size, y)], fill=color)

    def _draw_sparkle(self, draw: ImageDraw.Draw, x: int, y: int, size: int):
        """Draw a sparkle effect"""
        # Draw a star-like sparkle
        color = (255, 255, 255, 255)
        draw.line([(x - size, y), (x + size, y)], fill=color, width=2)
        draw.line([(x, y - size), (x, y + size)], fill=color, width=2)
        half = size // 2
        draw.line([(x - half, y - half), (x + half, y + half)], fill=color, width=1)
        draw.line([(x - half, y + half), (x + half, y - half)], fill=color, width=1)

    def create_expression_preview(
        self,
        expression_assets: List[Asset],
        base_image: Image.Image,
        output_path: Path
    ) -> Path:
        """
        Create a preview image of an expression by compositing all assets

        Args:
            expression_assets: List of assets for the expression
            base_image: Base character image
            output_path: Where to save the preview

        Returns:
            Path to the preview image
        """
        # Start with base image
        preview = base_image.copy()

        # Layer order for proper composition
        layer_order = ["brows", "eyes", "blush", "mouth", "tears"]

        # Composite assets in order
        for layer_type in layer_order:
            for asset in expression_assets:
                if asset.layer_type.startswith(layer_type):
                    try:
                        asset_img = Image.open(asset.file_path)
                        preview = Image.alpha_composite(preview.convert('RGBA'), asset_img.convert('RGBA'))
                    except Exception as e:
                        logger.warning(f"Failed to composite {asset.layer_type}: {e}")

        preview.save(output_path)
        return output_path
