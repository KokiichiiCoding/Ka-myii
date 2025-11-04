"""
Accessory generation module for VTuber models
Generates various accessories like hats, glasses, ribbons, etc.
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw
import random

from src.models.vtuber_model import Asset

logger = logging.getLogger(__name__)


class AccessoryType:
    """Types of accessories"""
    # Head accessories
    CAT_EARS = "cat_ears"
    BUNNY_EARS = "bunny_ears"
    HALO = "halo"
    CROWN = "crown"
    HAT = "hat"
    HEADBAND = "headband"
    HAIR_BOW = "hair_bow"
    HORNS = "horns"
    ANTENNA = "antenna"

    # Face accessories
    GLASSES = "glasses"
    MONOCLE = "monocle"
    EYE_PATCH = "eye_patch"
    MASK = "mask"

    # Body accessories
    NECKLACE = "necklace"
    CHOKER = "choker"
    BOW_TIE = "bow_tie"
    SCARF = "scarf"

    # Special effects
    SPARKLES = "sparkles"
    HEARTS = "hearts"
    STARS = "stars"
    FLOWERS = "flowers"
    BUTTERFLIES = "butterflies"


class Accessory:
    """Represents an accessory"""

    def __init__(
        self,
        accessory_type: str,
        name: str,
        color: Tuple[int, int, int, int],
        position: str = "auto",
        size_scale: float = 1.0
    ):
        self.accessory_type = accessory_type
        self.name = name
        self.color = color
        self.position = position
        self.size_scale = size_scale


class AccessoryGenerator:
    """
    Generates accessories for VTuber models
    """

    def __init__(self):
        logger.info("AccessoryGenerator initialized")

    def generate_accessory(
        self,
        base_image: Image.Image,
        accessory: Accessory,
        output_path: Path
    ) -> Asset:
        """
        Generate a single accessory

        Args:
            base_image: Base character image for size reference
            accessory: Accessory specification
            output_path: Where to save the accessory

        Returns:
            Asset object for the accessory
        """
        width, height = base_image.size

        # Create transparent image for accessory
        acc_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(acc_img)

        # Generate based on type
        if accessory.accessory_type == AccessoryType.CAT_EARS:
            self._draw_cat_ears(draw, width, height, accessory.color, accessory.size_scale)
        elif accessory.accessory_type == AccessoryType.BUNNY_EARS:
            self._draw_bunny_ears(draw, width, height, accessory.color, accessory.size_scale)
        elif accessory.accessory_type == AccessoryType.HALO:
            self._draw_halo(draw, width, height, accessory.color, accessory.size_scale)
        elif accessory.accessory_type == AccessoryType.CROWN:
            self._draw_crown(draw, width, height, accessory.color, accessory.size_scale)
        elif accessory.accessory_type == AccessoryType.GLASSES:
            self._draw_glasses(draw, width, height, accessory.color, accessory.size_scale)
        elif accessory.accessory_type == AccessoryType.HAIR_BOW:
            self._draw_hair_bow(draw, width, height, accessory.color, accessory.size_scale)
        elif accessory.accessory_type == AccessoryType.HORNS:
            self._draw_horns(draw, width, height, accessory.color, accessory.size_scale)
        elif accessory.accessory_type == AccessoryType.HEADBAND:
            self._draw_headband(draw, width, height, accessory.color, accessory.size_scale)
        elif accessory.accessory_type == AccessoryType.CHOKER:
            self._draw_choker(draw, width, height, accessory.color, accessory.size_scale)
        elif accessory.accessory_type == AccessoryType.BOW_TIE:
            self._draw_bow_tie(draw, width, height, accessory.color, accessory.size_scale)
        elif accessory.accessory_type == AccessoryType.SPARKLES:
            self._draw_sparkles(draw, width, height, accessory.color, accessory.size_scale)
        elif accessory.accessory_type == AccessoryType.HEARTS:
            self._draw_hearts(draw, width, height, accessory.color, accessory.size_scale)
        elif accessory.accessory_type == AccessoryType.STARS:
            self._draw_stars(draw, width, height, accessory.color, accessory.size_scale)
        else:
            logger.warning(f"Unknown accessory type: {accessory.accessory_type}")

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        acc_img.save(output_path)

        return Asset(
            layer_type=f"accessory_{accessory.accessory_type}",
            file_path=output_path,
            metadata={
                "accessory_type": accessory.accessory_type,
                "name": accessory.name,
                "color": accessory.color,
                "size_scale": accessory.size_scale
            }
        )

    def generate_accessory_set(
        self,
        base_image_path: Path,
        accessories: List[Accessory],
        output_dir: Path
    ) -> List[Asset]:
        """Generate multiple accessories"""
        base_image = Image.open(base_image_path)
        assets = []

        for i, accessory in enumerate(accessories):
            output_path = output_dir / f"{accessory.accessory_type}_{i}.png"
            asset = self.generate_accessory(base_image, accessory, output_path)
            assets.append(asset)

        logger.info(f"Generated {len(assets)} accessories")
        return assets

    # Drawing methods for different accessory types

    def _draw_cat_ears(self, draw: ImageDraw.Draw, w: int, h: int, color: Tuple, scale: float):
        """Draw cat ears"""
        ear_h = int(h * 0.15 * scale)
        ear_w = int(w * 0.08 * scale)
        ear_y = int(h * 0.15)

        # Left ear
        left_x = int(w * 0.3)
        draw.polygon(
            [(left_x, ear_y), (left_x - ear_w, ear_y - ear_h), (left_x + ear_w, ear_y - ear_h)],
            fill=color,
            outline=(0, 0, 0, 255),
            width=2
        )

        # Right ear
        right_x = int(w * 0.7)
        draw.polygon(
            [(right_x, ear_y), (right_x - ear_w, ear_y - ear_h), (right_x + ear_w, ear_y - ear_h)],
            fill=color,
            outline=(0, 0, 0, 255),
            width=2
        )

        # Inner ear detail
        inner_color = (255, 200, 200, 200)
        for ear_x in [left_x, right_x]:
            draw.polygon(
                [(ear_x, ear_y - ear_h//4), (ear_x - ear_w//2, ear_y - ear_h//2),
                 (ear_x + ear_w//2, ear_y - ear_h//2)],
                fill=inner_color
            )

    def _draw_bunny_ears(self, draw: ImageDraw.Draw, w: int, h: int, color: Tuple, scale: float):
        """Draw bunny ears"""
        ear_h = int(h * 0.25 * scale)
        ear_w = int(w * 0.06 * scale)
        ear_y = int(h * 0.1)

        for ear_x in [int(w * 0.35), int(w * 0.65)]:
            # Outer ear
            draw.ellipse(
                [ear_x - ear_w, ear_y - ear_h, ear_x + ear_w, ear_y],
                fill=color,
                outline=(0, 0, 0, 255),
                width=2
            )
            # Inner ear
            inner_color = (255, 200, 220, 200)
            draw.ellipse(
                [ear_x - ear_w//2, ear_y - ear_h + ear_h//4,
                 ear_x + ear_w//2, ear_y - ear_h//4],
                fill=inner_color
            )

    def _draw_halo(self, draw: ImageDraw.Draw, w: int, h: int, color: Tuple, scale: float):
        """Draw halo"""
        halo_y = int(h * 0.08)
        halo_x = w // 2
        halo_radius = int(w * 0.15 * scale)

        # Outer glow
        for i in range(3):
            alpha = max(50, color[3] - i * 30)
            glow_color = (color[0], color[1], color[2], alpha)
            draw.ellipse(
                [halo_x - halo_radius - i*2, halo_y - 10 - i*2,
                 halo_x + halo_radius + i*2, halo_y + 10 + i*2],
                outline=glow_color,
                width=2
            )

        # Main halo
        draw.ellipse(
            [halo_x - halo_radius, halo_y - 10, halo_x + halo_radius, halo_y + 10],
            fill=None,
            outline=color,
            width=4
        )

    def _draw_crown(self, draw: ImageDraw.Draw, w: int, h: int, color: Tuple, scale: float):
        """Draw crown"""
        crown_y = int(h * 0.12)
        crown_w = int(w * 0.25 * scale)
        crown_h = int(h * 0.08 * scale)
        center_x = w // 2

        # Crown base
        points = [
            (center_x - crown_w, crown_y),
            (center_x - crown_w * 0.6, crown_y - crown_h),
            (center_x - crown_w * 0.3, crown_y - crown_h * 0.6),
            (center_x, crown_y - crown_h),
            (center_x + crown_w * 0.3, crown_y - crown_h * 0.6),
            (center_x + crown_w * 0.6, crown_y - crown_h),
            (center_x + crown_w, crown_y),
        ]
        draw.polygon(points, fill=color, outline=(0, 0, 0, 255), width=2)

        # Jewels
        jewel_color = (255, 0, 0, 255)
        for x in [center_x, center_x - crown_w * 0.6, center_x + crown_w * 0.6]:
            draw.ellipse([x - 5, crown_y - crown_h - 5, x + 5, crown_y - crown_h + 5],
                        fill=jewel_color)

    def _draw_glasses(self, draw: ImageDraw.Draw, w: int, h: int, color: Tuple, scale: float):
        """Draw glasses"""
        eye_y = int(h * 0.4)
        lens_w = int(w * 0.12 * scale)
        lens_h = int(h * 0.08 * scale)

        # Left lens
        left_x = int(w * 0.35)
        draw.ellipse(
            [left_x - lens_w, eye_y - lens_h, left_x + lens_w, eye_y + lens_h],
            outline=color,
            width=3
        )

        # Right lens
        right_x = int(w * 0.65)
        draw.ellipse(
            [right_x - lens_w, eye_y - lens_h, right_x + lens_w, eye_y + lens_h],
            outline=color,
            width=3
        )

        # Bridge
        draw.line([(left_x + lens_w, eye_y), (right_x - lens_w, eye_y)],
                 fill=color, width=3)

    def _draw_hair_bow(self, draw: ImageDraw.Draw, w: int, h: int, color: Tuple, scale: float):
        """Draw hair bow"""
        bow_x = int(w * 0.75)
        bow_y = int(h * 0.25)
        bow_size = int(w * 0.08 * scale)

        # Left side
        draw.ellipse(
            [bow_x - bow_size*2, bow_y - bow_size, bow_x - bow_size//2, bow_y + bow_size],
            fill=color,
            outline=(0, 0, 0, 255),
            width=2
        )

        # Right side
        draw.ellipse(
            [bow_x + bow_size//2, bow_y - bow_size, bow_x + bow_size*2, bow_y + bow_size],
            fill=color,
            outline=(0, 0, 0, 255),
            width=2
        )

        # Center knot
        draw.ellipse(
            [bow_x - bow_size//2, bow_y - bow_size//2, bow_x + bow_size//2, bow_y + bow_size//2],
            fill=color,
            outline=(0, 0, 0, 255),
            width=2
        )

    def _draw_horns(self, draw: ImageDraw.Draw, w: int, h: int, color: Tuple, scale: float):
        """Draw horns"""
        horn_h = int(h * 0.15 * scale)
        horn_w = int(w * 0.04 * scale)
        horn_y = int(h * 0.18)

        for horn_x in [int(w * 0.32), int(w * 0.68)]:
            # Horn
            points = [
                (horn_x - horn_w, horn_y),
                (horn_x, horn_y - horn_h),
                (horn_x + horn_w, horn_y)
            ]
            draw.polygon(points, fill=color, outline=(0, 0, 0, 255), width=2)

            # Highlight
            highlight = (min(255, color[0] + 50), min(255, color[1] + 50),
                        min(255, color[2] + 50), color[3])
            draw.line([(horn_x - horn_w//2, horn_y - horn_h//4),
                      (horn_x, horn_y - horn_h)],
                     fill=highlight, width=2)

    def _draw_headband(self, draw: ImageDraw.Draw, w: int, h: int, color: Tuple, scale: float):
        """Draw headband"""
        band_y = int(h * 0.22)
        band_h = int(h * 0.03 * scale)

        # Main band
        draw.ellipse(
            [int(w * 0.2), band_y - band_h, int(w * 0.8), band_y + band_h],
            fill=color,
            outline=(0, 0, 0, 255),
            width=2
        )

    def _draw_choker(self, draw: ImageDraw.Draw, w: int, h: int, color: Tuple, scale: float):
        """Draw choker necklace"""
        choker_y = int(h * 0.68)
        choker_h = int(h * 0.02 * scale)

        draw.rectangle(
            [int(w * 0.35), choker_y - choker_h, int(w * 0.65), choker_y + choker_h],
            fill=color,
            outline=(0, 0, 0, 255),
            width=1
        )

        # Pendant
        pendant_x = w // 2
        draw.ellipse(
            [pendant_x - 8, choker_y + choker_h, pendant_x + 8, choker_y + choker_h + 16],
            fill=(255, 0, 0, 255),
            outline=(0, 0, 0, 255),
            width=1
        )

    def _draw_bow_tie(self, draw: ImageDraw.Draw, w: int, h: int, color: Tuple, scale: float):
        """Draw bow tie"""
        bow_x = w // 2
        bow_y = int(h * 0.7)
        bow_w = int(w * 0.08 * scale)
        bow_h = int(h * 0.04 * scale)

        # Left side
        draw.polygon(
            [(bow_x - bow_w*2, bow_y), (bow_x - bow_w//2, bow_y - bow_h),
             (bow_x - bow_w//2, bow_y + bow_h)],
            fill=color,
            outline=(0, 0, 0, 255),
            width=2
        )

        # Right side
        draw.polygon(
            [(bow_x + bow_w*2, bow_y), (bow_x + bow_w//2, bow_y - bow_h),
             (bow_x + bow_w//2, bow_y + bow_h)],
            fill=color,
            outline=(0, 0, 0, 255),
            width=2
        )

        # Center
        draw.rectangle(
            [bow_x - bow_w//2, bow_y - bow_h, bow_x + bow_w//2, bow_y + bow_h],
            fill=color,
            outline=(0, 0, 0, 255),
            width=2
        )

    def _draw_sparkles(self, draw: ImageDraw.Draw, w: int, h: int, color: Tuple, scale: float):
        """Draw sparkle effects"""
        num_sparkles = int(8 * scale)

        for _ in range(num_sparkles):
            x = random.randint(0, w)
            y = random.randint(0, h)
            size = random.randint(5, int(15 * scale))

            # Draw star shape
            self._draw_star(draw, x, y, size, color)

    def _draw_hearts(self, draw: ImageDraw.Draw, w: int, h: int, color: Tuple, scale: float):
        """Draw floating hearts"""
        num_hearts = int(6 * scale)

        for _ in range(num_hearts):
            x = random.randint(int(w * 0.1), int(w * 0.9))
            y = random.randint(int(h * 0.1), int(h * 0.9))
            size = random.randint(10, int(25 * scale))

            self._draw_heart(draw, x, y, size, color)

    def _draw_stars(self, draw: ImageDraw.Draw, w: int, h: int, color: Tuple, scale: float):
        """Draw star effects"""
        num_stars = int(10 * scale)

        for _ in range(num_stars):
            x = random.randint(0, w)
            y = random.randint(0, h)
            size = random.randint(8, int(20 * scale))

            self._draw_star(draw, x, y, size, color)

    def _draw_star(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple):
        """Helper to draw a star"""
        # Simple 4-point star
        draw.line([(x - size, y), (x + size, y)], fill=color, width=2)
        draw.line([(x, y - size), (x, y + size)], fill=color, width=2)
        half = size // 2
        draw.line([(x - half, y - half), (x + half, y + half)], fill=color, width=1)
        draw.line([(x - half, y + half), (x + half, y - half)], fill=color, width=1)

    def _draw_heart(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple):
        """Helper to draw a heart"""
        # Simple heart using circles and triangle
        draw.ellipse([x - size, y - size//2, x, y + size//2], fill=color)
        draw.ellipse([x, y - size//2, x + size, y + size//2], fill=color)
        draw.polygon([(x - size, y), (x, y + size), (x + size, y)], fill=color)

    def get_preset_accessories(self, preset_name: str) -> List[Accessory]:
        """
        Get predefined accessory sets

        Presets: cute, cool, elegant, fantasy, casual
        """
        presets = {
            "cute": [
                Accessory(AccessoryType.CAT_EARS, "Pink Cat Ears",
                         (255, 182, 193, 255), size_scale=1.0),
                Accessory(AccessoryType.HAIR_BOW, "Red Bow",
                         (255, 100, 100, 255), size_scale=0.8),
                Accessory(AccessoryType.HEARTS, "Heart Effects",
                         (255, 150, 200, 200), size_scale=1.0),
            ],
            "cool": [
                Accessory(AccessoryType.GLASSES, "Black Glasses",
                         (0, 0, 0, 255), size_scale=1.0),
                Accessory(AccessoryType.CHOKER, "Black Choker",
                         (20, 20, 20, 255), size_scale=1.0),
            ],
            "elegant": [
                Accessory(AccessoryType.CROWN, "Gold Crown",
                         (255, 215, 0, 255), size_scale=0.9),
                Accessory(AccessoryType.CHOKER, "Pearl Choker",
                         (240, 240, 240, 255), size_scale=0.8),
                Accessory(AccessoryType.SPARKLES, "Sparkle Effects",
                         (255, 255, 255, 200), size_scale=0.8),
            ],
            "fantasy": [
                Accessory(AccessoryType.HALO, "Angel Halo",
                         (255, 255, 150, 200), size_scale=1.0),
                Accessory(AccessoryType.HORNS, "Devil Horns",
                         (150, 50, 50, 255), size_scale=1.0),
                Accessory(AccessoryType.STARS, "Magic Stars",
                         (200, 150, 255, 200), size_scale=1.0),
            ],
            "casual": [
                Accessory(AccessoryType.HEADBAND, "Headband",
                         (100, 150, 255, 255), size_scale=1.0),
                Accessory(AccessoryType.BOW_TIE, "Bow Tie",
                         (255, 100, 100, 255), size_scale=0.9),
            ],
        }

        return presets.get(preset_name, [])
