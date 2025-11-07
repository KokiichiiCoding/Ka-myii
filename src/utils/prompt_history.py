"""
AI Prompt History System
Stores and manages generation prompts with thumbnails for re-use
"""
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class PromptHistory:
    """
    Manages prompt history with thumbnails and parameters
    """

    def __init__(self, history_dir: Path):
        """
        Initialize prompt history

        Args:
            history_dir: Directory to store history
        """
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.history_dir / "prompt_history.json"
        self.thumbnails_dir = self.history_dir / "thumbnails"
        self.thumbnails_dir.mkdir(exist_ok=True)

        self._load_history()

    def _load_history(self):
        """Load history from file"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = []

    def _save_history(self):
        """Save history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def add_entry(
        self,
        prompt: str,
        negative_prompt: str,
        parameters: Dict,
        model_id: str,
        thumbnail_path: Optional[Path] = None
    ) -> str:
        """
        Add a new history entry

        Args:
            prompt: Generation prompt
            negative_prompt: Negative prompt
            parameters: Generation parameters
            model_id: Generated model ID
            thumbnail_path: Path to thumbnail image

        Returns:
            Entry ID
        """
        # Generate entry ID
        entry_id = hashlib.md5(
            f"{prompt}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        # Copy thumbnail if provided
        thumbnail_name = None
        if thumbnail_path and thumbnail_path.exists():
            thumbnail_name = f"{entry_id}.png"
            thumbnail_dest = self.thumbnails_dir / thumbnail_name

            from PIL import Image
            img = Image.open(thumbnail_path)
            img.thumbnail((256, 256))
            img.save(thumbnail_dest)

        # Create entry
        entry = {
            "id": entry_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "parameters": parameters,
            "model_id": model_id,
            "thumbnail": thumbnail_name,
            "timestamp": datetime.now().isoformat(),
            "used_count": 0
        }

        self.history.append(entry)
        self._save_history()

        logger.info(f"Added prompt history entry: {entry_id}")
        return entry_id

    def get_entry(self, entry_id: str) -> Optional[Dict]:
        """Get a specific history entry"""
        for entry in self.history:
            if entry["id"] == entry_id:
                return entry
        return None

    def get_recent(self, limit: int = 20) -> List[Dict]:
        """Get recent history entries"""
        return sorted(
            self.history,
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]

    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Search history by prompt text

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching entries
        """
        query_lower = query.lower()
        results = [
            entry for entry in self.history
            if query_lower in entry["prompt"].lower()
            or query_lower in entry.get("negative_prompt", "").lower()
        ]

        return sorted(
            results,
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]

    def increment_usage(self, entry_id: str):
        """Increment usage count for an entry"""
        for entry in self.history:
            if entry["id"] == entry_id:
                entry["used_count"] = entry.get("used_count", 0) + 1
                self._save_history()
                break

    def get_popular(self, limit: int = 10) -> List[Dict]:
        """Get most frequently used prompts"""
        return sorted(
            self.history,
            key=lambda x: x.get("used_count", 0),
            reverse=True
        )[:limit]

    def get_by_parameters(
        self,
        style: Optional[str] = None,
        min_steps: Optional[int] = None,
        max_steps: Optional[int] = None
    ) -> List[Dict]:
        """Filter history by parameters"""
        results = self.history

        if style:
            results = [e for e in results if e["parameters"].get("style") == style]

        if min_steps is not None:
            results = [e for e in results if e["parameters"].get("steps", 0) >= min_steps]

        if max_steps is not None:
            results = [e for e in results if e["parameters"].get("steps", 0) <= max_steps]

        return sorted(results, key=lambda x: x["timestamp"], reverse=True)

    def delete_entry(self, entry_id: str) -> bool:
        """Delete a history entry"""
        for i, entry in enumerate(self.history):
            if entry["id"] == entry_id:
                # Delete thumbnail if exists
                if entry.get("thumbnail"):
                    thumb_path = self.thumbnails_dir / entry["thumbnail"]
                    if thumb_path.exists():
                        thumb_path.unlink()

                del self.history[i]
                self._save_history()
                logger.info(f"Deleted prompt history entry: {entry_id}")
                return True

        return False

    def export_favorites(self, output_path: Path, entry_ids: List[str]):
        """Export favorite prompts to JSON file"""
        favorites = [
            entry for entry in self.history
            if entry["id"] in entry_ids
        ]

        with open(output_path, 'w') as f:
            json.dump(favorites, f, indent=2)

        logger.info(f"Exported {len(favorites)} favorites to {output_path}")

    def import_prompts(self, import_path: Path):
        """Import prompts from JSON file"""
        with open(import_path, 'r') as f:
            imported = json.load(f)

        # Add imported entries (avoid duplicates by prompt+timestamp)
        existing_keys = {
            (e["prompt"], e["timestamp"]) for e in self.history
        }

        added = 0
        for entry in imported:
            key = (entry["prompt"], entry["timestamp"])
            if key not in existing_keys:
                self.history.append(entry)
                added += 1

        self._save_history()
        logger.info(f"Imported {added} new prompts")

    def get_statistics(self) -> Dict:
        """Get history statistics"""
        if not self.history:
            return {
                "total_entries": 0,
                "unique_prompts": 0,
                "total_usage": 0,
                "avg_usage": 0,
                "styles": {}
            }

        unique_prompts = len(set(e["prompt"] for e in self.history))
        total_usage = sum(e.get("used_count", 0) for e in self.history)

        # Count by style
        styles = {}
        for entry in self.history:
            style = entry["parameters"].get("style", "unknown")
            styles[style] = styles.get(style, 0) + 1

        return {
            "total_entries": len(self.history),
            "unique_prompts": unique_prompts,
            "total_usage": total_usage,
            "avg_usage": total_usage / len(self.history) if self.history else 0,
            "styles": styles,
            "oldest_entry": min(e["timestamp"] for e in self.history) if self.history else None,
            "newest_entry": max(e["timestamp"] for e in self.history) if self.history else None
        }
