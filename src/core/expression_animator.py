"""
Expression Animation Generator
Creates GIF/WebM previews of expression animations
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image, ImageSequence
import io

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class ExpressionAnimator:
    """
    Creates animated previews of expressions
    """

    def __init__(self):
        logger.info("ExpressionAnimator initialized")

    def create_blink_animation(
        self,
        base_image_path: Path,
        eyes_open_path: Path,
        eyes_closed_path: Path,
        output_path: Path,
        duration: float = 2.0,
        fps: int = 30
    ) -> Path:
        """
        Create blink animation GIF

        Args:
            base_image_path: Base character image
            eyes_open_path: Eyes open layer
            eyes_closed_path: Eyes closed layer
            output_path: Output GIF path
            duration: Total animation duration in seconds
            fps: Frames per second

        Returns:
            Path to generated GIF
        """
        logger.info("Creating blink animation")

        frames = []
        total_frames = int(duration * fps)

        # Load images
        base = Image.open(base_image_path).convert('RGBA')
        eyes_open = Image.open(eyes_open_path).convert('RGBA')
        eyes_closed = Image.open(eyes_closed_path).convert('RGBA')

        # Blink pattern: open -> closing -> closed -> opening -> open
        blink_start = int(total_frames * 0.4)
        blink_end = int(total_frames * 0.6)
        blink_duration = blink_end - blink_start

        for i in range(total_frames):
            frame = base.copy()

            if i < blink_start or i > blink_end:
                # Eyes open
                frame = Image.alpha_composite(frame, eyes_open)
            elif i == blink_start + blink_duration // 2:
                # Fully closed
                frame = Image.alpha_composite(frame, eyes_closed)
            else:
                # Transitioning
                if i < blink_start + blink_duration // 2:
                    # Closing
                    progress = (i - blink_start) / (blink_duration // 2)
                else:
                    # Opening
                    progress = 1 - (i - (blink_start + blink_duration // 2)) / (blink_duration // 2)

                # Blend between open and closed
                blended = self._blend_images(eyes_open, eyes_closed, progress)
                frame = Image.alpha_composite(frame, blended)

            frames.append(frame)

        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // fps,
            loop=0
        )

        logger.info(f"Blink animation saved: {output_path}")
        return output_path

    def create_talking_animation(
        self,
        base_image_path: Path,
        mouth_closed_path: Path,
        mouth_open_path: Path,
        output_path: Path,
        duration: float = 1.5,
        fps: int = 30
    ) -> Path:
        """Create talking/mouth movement animation"""
        logger.info("Creating talking animation")

        frames = []
        total_frames = int(duration * fps)

        # Load images
        base = Image.open(base_image_path).convert('RGBA')
        mouth_closed = Image.open(mouth_closed_path).convert('RGBA')
        mouth_open = Image.open(mouth_open_path).convert('RGBA')

        # Talking pattern: closed -> open -> closed (repeat)
        cycle_frames = total_frames // 3

        for i in range(total_frames):
            frame = base.copy()
            cycle_pos = i % cycle_frames
            progress = cycle_pos / cycle_frames

            if progress < 0.5:
                # Opening
                blend_progress = progress * 2
                blended = self._blend_images(mouth_closed, mouth_open, blend_progress)
            else:
                # Closing
                blend_progress = (1 - progress) * 2
                blended = self._blend_images(mouth_closed, mouth_open, blend_progress)

            frame = Image.alpha_composite(frame, blended)
            frames.append(frame)

        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // fps,
            loop=0
        )

        logger.info(f"Talking animation saved: {output_path}")
        return output_path

    def create_expression_transition(
        self,
        expression1_path: Path,
        expression2_path: Path,
        output_path: Path,
        duration: float = 1.0,
        fps: int = 30
    ) -> Path:
        """
        Create smooth transition between two expressions

        Args:
            expression1_path: First expression image
            expression2_path: Second expression image
            output_path: Output GIF path
            duration: Transition duration in seconds
            fps: Frames per second

        Returns:
            Path to generated GIF
        """
        logger.info(f"Creating expression transition animation")

        frames = []
        total_frames = int(duration * fps)

        # Load expressions
        expr1 = Image.open(expression1_path).convert('RGBA')
        expr2 = Image.open(expression2_path).convert('RGBA')

        for i in range(total_frames):
            progress = i / total_frames

            # Smooth easing
            eased_progress = self._ease_in_out(progress)

            # Blend expressions
            frame = self._blend_images(expr1, expr2, eased_progress)
            frames.append(frame)

        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // fps,
            loop=0
        )

        logger.info(f"Transition animation saved: {output_path}")
        return output_path

    def create_expression_showcase(
        self,
        base_image_path: Path,
        expression_paths: List[Tuple[str, Path]],
        output_path: Path,
        hold_duration: float = 1.0,
        transition_duration: float = 0.5,
        fps: int = 30
    ) -> Path:
        """
        Create showcase animation cycling through all expressions

        Args:
            base_image_path: Base character image
            expression_paths: List of (name, path) tuples for expressions
            output_path: Output GIF path
            hold_duration: How long to hold each expression
            transition_duration: Duration of transitions
            fps: Frames per second

        Returns:
            Path to generated GIF
        """
        logger.info(f"Creating expression showcase with {len(expression_paths)} expressions")

        frames = []
        hold_frames = int(hold_duration * fps)
        transition_frames = int(transition_duration * fps)

        # Load all expressions
        expressions = [Image.open(path).convert('RGBA') for _, path in expression_paths]

        for i, expr in enumerate(expressions):
            # Hold current expression
            for _ in range(hold_frames):
                frames.append(expr.copy())

            # Transition to next
            next_expr = expressions[(i + 1) % len(expressions)]

            for j in range(transition_frames):
                progress = j / transition_frames
                eased_progress = self._ease_in_out(progress)
                blended = self._blend_images(expr, next_expr, eased_progress)
                frames.append(blended)

        # Save as GIF
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000 // fps,
                loop=0
            )

        logger.info(f"Showcase animation saved: {output_path}")
        return output_path

    def create_webm_video(
        self,
        frames: List[Image.Image],
        output_path: Path,
        fps: int = 30,
        quality: int = 90
    ) -> Path:
        """
        Create WebM video from frames (if CV2 available)

        Args:
            frames: List of PIL Images
            output_path: Output WebM path
            fps: Frames per second
            quality: Video quality (0-100)

        Returns:
            Path to generated video
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, cannot create WebM")
            # Fallback to GIF
            gif_path = output_path.with_suffix('.gif')
            if frames:
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=1000 // fps,
                    loop=0
                )
            return gif_path

        logger.info("Creating WebM video")

        # Convert frames to numpy arrays
        frame_arrays = []
        for frame in frames:
            # Convert RGBA to BGR for OpenCV
            if frame.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', frame.size, (255, 255, 255))
                background.paste(frame, mask=frame.split()[3])
                frame = background

            frame_array = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            frame_arrays.append(frame_array)

        # Get dimensions
        height, width = frame_arrays[0].shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Write frames
        for frame_array in frame_arrays:
            out.write(frame_array)

        out.release()

        logger.info(f"WebM video saved: {output_path}")
        return output_path

    def _blend_images(
        self,
        img1: Image.Image,
        img2: Image.Image,
        alpha: float
    ) -> Image.Image:
        """
        Blend two images with alpha blending

        Args:
            img1: First image
            img2: Second image
            alpha: Blend factor (0=img1, 1=img2)

        Returns:
            Blended image
        """
        return Image.blend(img1, img2, alpha)

    def _ease_in_out(self, t: float) -> float:
        """Smooth easing function"""
        return t * t * (3 - 2 * t)

    def optimize_gif(
        self,
        input_path: Path,
        output_path: Path,
        max_colors: int = 256,
        max_size: Optional[Tuple[int, int]] = None
    ) -> Path:
        """
        Optimize GIF file size

        Args:
            input_path: Input GIF
            output_path: Output optimized GIF
            max_colors: Maximum colors in palette
            max_size: Maximum dimensions (width, height)

        Returns:
            Path to optimized GIF
        """
        img = Image.open(input_path)

        frames = []
        for frame in ImageSequence.Iterator(img):
            frame = frame.convert('RGBA')

            # Resize if needed
            if max_size:
                frame.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Reduce colors
            frame = frame.convert('P', palette=Image.ADAPTIVE, colors=max_colors)

            frames.append(frame)

        # Save optimized
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=img.info.get('duration', 100),
                loop=img.info.get('loop', 0),
                optimize=True
            )

        logger.info(f"Optimized GIF saved: {output_path}")
        return output_path
