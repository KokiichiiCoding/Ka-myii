"""
Image editing module for modifying generated VTuber assets
"""
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
import io
import base64

logger = logging.getLogger(__name__)


class ImageEditor:
    """
    Image editing tools for VTuber model assets
    """

    def __init__(self):
        logger.info("ImageEditor initialized")

    def adjust_brightness(
        self,
        image_path: Path,
        factor: float,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Adjust image brightness

        Args:
            image_path: Input image path
            factor: Brightness factor (1.0 = no change, >1.0 = brighter, <1.0 = darker)
            output_path: Output path (None = overwrite)

        Returns:
            Path to adjusted image
        """
        output_path = output_path or image_path

        img = Image.open(image_path)
        enhancer = ImageEnhance.Brightness(img)
        adjusted = enhancer.enhance(factor)
        adjusted.save(output_path)

        logger.info(f"Adjusted brightness by {factor}x: {output_path}")
        return output_path

    def adjust_contrast(
        self,
        image_path: Path,
        factor: float,
        output_path: Optional[Path] = None
    ) -> Path:
        """Adjust image contrast"""
        output_path = output_path or image_path

        img = Image.open(image_path)
        enhancer = ImageEnhance.Contrast(img)
        adjusted = enhancer.enhance(factor)
        adjusted.save(output_path)

        logger.info(f"Adjusted contrast by {factor}x: {output_path}")
        return output_path

    def adjust_saturation(
        self,
        image_path: Path,
        factor: float,
        output_path: Optional[Path] = None
    ) -> Path:
        """Adjust color saturation"""
        output_path = output_path or image_path

        img = Image.open(image_path)
        enhancer = ImageEnhance.Color(img)
        adjusted = enhancer.enhance(factor)
        adjusted.save(output_path)

        logger.info(f"Adjusted saturation by {factor}x: {output_path}")
        return output_path

    def adjust_sharpness(
        self,
        image_path: Path,
        factor: float,
        output_path: Optional[Path] = None
    ) -> Path:
        """Adjust image sharpness"""
        output_path = output_path or image_path

        img = Image.open(image_path)
        enhancer = ImageEnhance.Sharpness(img)
        adjusted = enhancer.enhance(factor)
        adjusted.save(output_path)

        logger.info(f"Adjusted sharpness by {factor}x: {output_path}")
        return output_path

    def resize(
        self,
        image_path: Path,
        width: int,
        height: int,
        output_path: Optional[Path] = None,
        maintain_aspect: bool = True
    ) -> Path:
        """Resize image"""
        output_path = output_path or image_path

        img = Image.open(image_path)

        if maintain_aspect:
            img.thumbnail((width, height), Image.Resampling.LANCZOS)
        else:
            img = img.resize((width, height), Image.Resampling.LANCZOS)

        img.save(output_path)

        logger.info(f"Resized to {width}x{height}: {output_path}")
        return output_path

    def crop(
        self,
        image_path: Path,
        left: int,
        top: int,
        right: int,
        bottom: int,
        output_path: Optional[Path] = None
    ) -> Path:
        """Crop image to specified bounds"""
        output_path = output_path or image_path

        img = Image.open(image_path)
        cropped = img.crop((left, top, right, bottom))
        cropped.save(output_path)

        logger.info(f"Cropped image: {output_path}")
        return output_path

    def rotate(
        self,
        image_path: Path,
        angle: float,
        output_path: Optional[Path] = None,
        expand: bool = True
    ) -> Path:
        """Rotate image by angle in degrees"""
        output_path = output_path or image_path

        img = Image.open(image_path)
        rotated = img.rotate(angle, expand=expand, fillcolor=(0, 0, 0, 0))
        rotated.save(output_path)

        logger.info(f"Rotated {angle} degrees: {output_path}")
        return output_path

    def flip(
        self,
        image_path: Path,
        direction: str,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Flip image

        Args:
            direction: 'horizontal' or 'vertical'
        """
        output_path = output_path or image_path

        img = Image.open(image_path)

        if direction.lower() == 'horizontal':
            flipped = ImageOps.mirror(img)
        elif direction.lower() == 'vertical':
            flipped = ImageOps.flip(img)
        else:
            raise ValueError(f"Invalid direction: {direction}")

        flipped.save(output_path)

        logger.info(f"Flipped {direction}: {output_path}")
        return output_path

    def apply_filter(
        self,
        image_path: Path,
        filter_type: str,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Apply filter to image

        Filter types: blur, sharpen, smooth, edge_enhance, emboss, contour
        """
        output_path = output_path or image_path

        img = Image.open(image_path)

        filters = {
            'blur': ImageFilter.BLUR,
            'sharpen': ImageFilter.SHARPEN,
            'smooth': ImageFilter.SMOOTH,
            'edge_enhance': ImageFilter.EDGE_ENHANCE,
            'emboss': ImageFilter.EMBOSS,
            'contour': ImageFilter.CONTOUR,
            'detail': ImageFilter.DETAIL,
        }

        if filter_type not in filters:
            raise ValueError(f"Unknown filter: {filter_type}")

        filtered = img.filter(filters[filter_type])
        filtered.save(output_path)

        logger.info(f"Applied {filter_type} filter: {output_path}")
        return output_path

    def add_border(
        self,
        image_path: Path,
        border_width: int,
        color: Tuple[int, int, int, int],
        output_path: Optional[Path] = None
    ) -> Path:
        """Add a colored border around the image"""
        output_path = output_path or image_path

        img = Image.open(image_path)
        bordered = ImageOps.expand(img, border=border_width, fill=color)
        bordered.save(output_path)

        logger.info(f"Added border: {output_path}")
        return output_path

    def composite_images(
        self,
        base_image_path: Path,
        overlay_image_path: Path,
        position: Tuple[int, int],
        output_path: Path,
        opacity: float = 1.0
    ) -> Path:
        """
        Composite overlay image onto base image

        Args:
            base_image_path: Base image
            overlay_image_path: Image to overlay
            position: (x, y) position for overlay
            output_path: Output path
            opacity: Overlay opacity (0.0 to 1.0)

        Returns:
            Path to composited image
        """
        base = Image.open(base_image_path).convert('RGBA')
        overlay = Image.open(overlay_image_path).convert('RGBA')

        # Adjust overlay opacity if needed
        if opacity < 1.0:
            alpha = overlay.split()[3]
            alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
            overlay.putalpha(alpha)

        # Composite
        base.paste(overlay, position, overlay)
        base.save(output_path)

        logger.info(f"Composited images: {output_path}")
        return output_path

    def draw_on_image(
        self,
        image_path: Path,
        draw_data: List[Dict],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Draw on image using draw data

        Draw data format:
        [
            {
                "type": "line",
                "coords": [(x1, y1), (x2, y2)],
                "color": (r, g, b, a),
                "width": 3
            },
            {
                "type": "rectangle",
                "coords": [(x1, y1), (x2, y2)],
                "color": (r, g, b, a),
                "fill": True
            },
            ...
        ]
        """
        output_path = output_path or image_path

        img = Image.open(image_path).convert('RGBA')
        draw = ImageDraw.Draw(img)

        for item in draw_data:
            draw_type = item.get("type")
            coords = item.get("coords", [])
            color = tuple(item.get("color", [0, 0, 0, 255]))
            width = item.get("width", 1)

            if draw_type == "line":
                draw.line(coords, fill=color, width=width)
            elif draw_type == "rectangle":
                if item.get("fill", False):
                    draw.rectangle(coords, fill=color, outline=color, width=width)
                else:
                    draw.rectangle(coords, outline=color, width=width)
            elif draw_type == "ellipse":
                if item.get("fill", False):
                    draw.ellipse(coords, fill=color, outline=color, width=width)
                else:
                    draw.ellipse(coords, outline=color, width=width)
            elif draw_type == "polygon":
                if item.get("fill", False):
                    draw.polygon(coords, fill=color, outline=color)
                else:
                    draw.polygon(coords, outline=color)
            elif draw_type == "text":
                text = item.get("text", "")
                draw.text(coords[0] if coords else (0, 0), text, fill=color)

        img.save(output_path)

        logger.info(f"Drew on image: {output_path}")
        return output_path

    def auto_enhance(
        self,
        image_path: Path,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Automatically enhance image quality
        """
        output_path = output_path or image_path

        img = Image.open(image_path)

        # Auto contrast
        img = ImageOps.autocontrast(img)

        # Slight sharpening
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)

        # Slight color enhancement
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.1)

        img.save(output_path)

        logger.info(f"Auto-enhanced image: {output_path}")
        return output_path

    def remove_background(
        self,
        image_path: Path,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Remove background from image (requires rembg)
        """
        output_path = output_path or image_path

        try:
            from rembg import remove

            with open(image_path, 'rb') as i:
                input_data = i.read()
                output_data = remove(input_data)

            with open(output_path, 'wb') as o:
                o.write(output_data)

            logger.info(f"Removed background: {output_path}")
            return output_path

        except ImportError:
            logger.warning("rembg not installed, skipping background removal")
            # Just return original
            if output_path != image_path:
                img = Image.open(image_path)
                img.save(output_path)
            return output_path

    def image_to_base64(self, image_path: Path) -> str:
        """Convert image to base64 string for web display"""
        with open(image_path, 'rb') as f:
            img_data = f.read()
            return base64.b64encode(img_data).decode('utf-8')

    def base64_to_image(self, base64_str: str, output_path: Path) -> Path:
        """Convert base64 string to image file"""
        img_data = base64.b64decode(base64_str)
        with open(output_path, 'wb') as f:
            f.write(img_data)
        return output_path

    def create_thumbnail(
        self,
        image_path: Path,
        output_path: Path,
        size: Tuple[int, int] = (256, 256)
    ) -> Path:
        """Create a thumbnail of the image"""
        img = Image.open(image_path)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        img.save(output_path)

        logger.info(f"Created thumbnail: {output_path}")
        return output_path

    def batch_edit(
        self,
        image_paths: List[Path],
        operations: List[Dict],
        output_dir: Path
    ) -> List[Path]:
        """
        Apply multiple operations to multiple images

        Args:
            image_paths: List of images to process
            operations: List of operations to apply
            output_dir: Output directory

        Operations format:
        [
            {"type": "brightness", "factor": 1.2},
            {"type": "contrast", "factor": 1.1},
            {"type": "resize", "width": 512, "height": 512},
            ...
        ]

        Returns:
            List of output paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []

        for img_path in image_paths:
            output_path = output_dir / img_path.name
            current_path = img_path

            # Apply each operation in sequence
            for i, op in enumerate(operations):
                op_type = op.get("type")
                temp_path = output_dir / f"temp_{i}_{img_path.name}"

                if op_type == "brightness":
                    current_path = self.adjust_brightness(current_path, op["factor"], temp_path)
                elif op_type == "contrast":
                    current_path = self.adjust_contrast(current_path, op["factor"], temp_path)
                elif op_type == "saturation":
                    current_path = self.adjust_saturation(current_path, op["factor"], temp_path)
                elif op_type == "sharpness":
                    current_path = self.adjust_sharpness(current_path, op["factor"], temp_path)
                elif op_type == "resize":
                    current_path = self.resize(
                        current_path,
                        op["width"],
                        op["height"],
                        temp_path,
                        op.get("maintain_aspect", True)
                    )
                elif op_type == "filter":
                    current_path = self.apply_filter(current_path, op["filter_type"], temp_path)
                elif op_type == "rotate":
                    current_path = self.rotate(current_path, op["angle"], temp_path)

            # Move final result to output
            final_img = Image.open(current_path)
            final_img.save(output_path)

            # Clean up temp files
            for i in range(len(operations)):
                temp_path = output_dir / f"temp_{i}_{img_path.name}"
                if temp_path.exists():
                    temp_path.unlink()

            output_paths.append(output_path)
            logger.info(f"Batch processed: {output_path}")

        return output_paths
