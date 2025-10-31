"""
Utility helper functions
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """
    Setup logging configuration

    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def save_json(data: Dict[str, Any], file_path: Path):
    """
    Save data to JSON file

    Args:
        data: Data to save
        file_path: Path to save to
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.debug(f"Saved JSON to {file_path}")


def load_json(file_path: Path) -> Dict[str, Any]:
    """
    Load data from JSON file

    Args:
        file_path: Path to load from

    Returns:
        Loaded data
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    logger.debug(f"Loaded JSON from {file_path}")
    return data


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    import re
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]

    return filename


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    size_bytes = file_path.stat().st_size
    return size_bytes / (1024 * 1024)


def create_unique_id() -> str:
    """Generate a unique ID"""
    import uuid
    return str(uuid.uuid4())


def timestamp() -> str:
    """Get current timestamp as ISO format string"""
    return datetime.now().isoformat()


class ProgressTracker:
    """Simple progress tracking utility"""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()

    def step(self, message: str = ""):
        """Increment progress by one step"""
        self.current_step += 1
        progress = self.current_step / self.total_steps
        elapsed = (datetime.now() - self.start_time).total_seconds()

        logger.info(f"[{progress * 100:.0f}%] {message}")

    def complete(self):
        """Mark as complete"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"Completed in {format_time(elapsed)}")
