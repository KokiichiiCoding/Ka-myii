"""
Model Versioning System
Auto-versioning for models with reproducibility tracking
"""
import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


class ModelVersion:
    """
    Represents a model version with full reproducibility info
    """

    def __init__(
        self,
        model_id: str,
        version: str,
        config: Dict,
        seed: Optional[int] = None
    ):
        self.model_id = model_id
        self.version = version
        self.config = config
        self.seed = seed
        self.created_at = datetime.now().isoformat()
        self.hash = self._generate_hash()

    def _generate_hash(self) -> str:
        """Generate version hash for reproducibility"""
        # Create hash from config + seed
        hash_data = json.dumps(self.config, sort_keys=True) + str(self.seed)
        return hashlib.sha256(hash_data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "hash": self.hash,
            "config": self.config,
            "seed": self.seed,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        """Create from dictionary"""
        version = cls(
            model_id=data["model_id"],
            version=data["version"],
            config=data["config"],
            seed=data.get("seed")
        )
        version.created_at = data.get("created_at", version.created_at)
        return version


class ModelVersionManager:
    """
    Manages model versions and enables reproducibility
    """

    def __init__(self, versions_dir: Path):
        """
        Initialize version manager

        Args:
            versions_dir: Directory to store version info
        """
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelVersionManager initialized at {versions_dir}")

    def create_version(
        self,
        model_id: str,
        config: Dict,
        seed: Optional[int] = None,
        model_dir: Optional[Path] = None
    ) -> ModelVersion:
        """
        Create a new model version

        Args:
            model_id: Model ID
            config: Generation configuration
            seed: Random seed used
            model_dir: Model directory to snapshot

        Returns:
            Created ModelVersion
        """
        # Get next version number
        existing_versions = self.get_versions(model_id)
        version_num = len(existing_versions) + 1
        version_str = f"v{version_num}"

        # Create version
        version = ModelVersion(model_id, version_str, config, seed)

        # Save version info
        version_file = self.versions_dir / f"{model_id}_{version.hash}.json"
        with open(version_file, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)

        # Create snapshot if model_dir provided
        if model_dir and model_dir.exists():
            self._create_snapshot(model_id, version_str, model_dir)

        logger.info(f"Created version {version_str} for model {model_id}")
        return version

    def get_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """Get a specific version"""
        version_files = self.versions_dir.glob(f"{model_id}_*.json")

        for version_file in version_files:
            with open(version_file, 'r') as f:
                data = json.load(f)
                if data["version"] == version:
                    return ModelVersion.from_dict(data)

        return None

    def get_versions(self, model_id: str) -> list[ModelVersion]:
        """Get all versions of a model"""
        versions = []
        version_files = self.versions_dir.glob(f"{model_id}_*.json")

        for version_file in version_files:
            with open(version_file, 'r') as f:
                data = json.load(f)
                versions.append(ModelVersion.from_dict(data))

        return sorted(versions, key=lambda v: v.created_at)

    def get_latest_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get the latest version of a model"""
        versions = self.get_versions(model_id)
        return versions[-1] if versions else None

    def find_by_hash(self, version_hash: str) -> Optional[ModelVersion]:
        """Find version by hash"""
        for version_file in self.versions_dir.glob(f"*_{version_hash}.json"):
            with open(version_file, 'r') as f:
                data = json.load(f)
                return ModelVersion.from_dict(data)

        return None

    def compare_versions(
        self,
        model_id: str,
        version1: str,
        version2: str
    ) -> Dict:
        """
        Compare two versions

        Returns:
            Dictionary with comparison results
        """
        v1 = self.get_version(model_id, version1)
        v2 = self.get_version(model_id, version2)

        if not v1 or not v2:
            return {"error": "One or both versions not found"}

        # Find differences
        differences = {}

        # Compare configs
        for key in set(list(v1.config.keys()) + list(v2.config.keys())):
            val1 = v1.config.get(key)
            val2 = v2.config.get(key)

            if val1 != val2:
                differences[key] = {
                    version1: val1,
                    version2: val2
                }

        return {
            "version1": version1,
            "version2": version2,
            "differences": differences,
            "seed_changed": v1.seed != v2.seed
        }

    def _create_snapshot(
        self,
        model_id: str,
        version: str,
        model_dir: Path
    ):
        """Create a snapshot of model files"""
        snapshot_dir = self.versions_dir / "snapshots" / model_id / version
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Copy model files
        for file_path in model_dir.rglob('*'):
            if file_path.is_file():
                rel_path = file_path.relative_to(model_dir)
                dest_path = snapshot_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest_path)

        logger.info(f"Created snapshot for {model_id} {version}")

    def restore_version(
        self,
        model_id: str,
        version: str,
        restore_dir: Path
    ) -> bool:
        """
        Restore a model from a version snapshot

        Args:
            model_id: Model ID
            version: Version to restore
            restore_dir: Directory to restore to

        Returns:
            True if successful
        """
        snapshot_dir = self.versions_dir / "snapshots" / model_id / version

        if not snapshot_dir.exists():
            logger.error(f"Snapshot not found for {model_id} {version}")
            return False

        # Copy snapshot to restore directory
        if restore_dir.exists():
            shutil.rmtree(restore_dir)

        shutil.copytree(snapshot_dir, restore_dir)

        logger.info(f"Restored {model_id} {version} to {restore_dir}")
        return True

    def export_version_history(
        self,
        model_id: str,
        output_path: Path
    ):
        """Export version history to JSON"""
        versions = self.get_versions(model_id)

        history = {
            "model_id": model_id,
            "total_versions": len(versions),
            "versions": [v.to_dict() for v in versions]
        }

        with open(output_path, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"Exported version history to {output_path}")

    def get_reproducibility_info(
        self,
        model_id: str,
        version: str
    ) -> Optional[Dict]:
        """
        Get complete reproducibility information

        Returns:
            Dictionary with all info needed to reproduce the model
        """
        version_obj = self.get_version(model_id, version)

        if not version_obj:
            return None

        return {
            "model_id": model_id,
            "version": version,
            "hash": version_obj.hash,
            "seed": version_obj.seed,
            "config": version_obj.config,
            "created_at": version_obj.created_at,
            "instructions": {
                "description": "Use this information to reproduce the exact same model",
                "steps": [
                    "Use the same seed value",
                    "Apply the exact config parameters",
                    "Ensure same model version is used"
                ]
            }
        }

    def cleanup_old_snapshots(
        self,
        model_id: str,
        keep_latest: int = 5
    ):
        """
        Clean up old version snapshots, keeping only the latest N

        Args:
            model_id: Model ID
            keep_latest: Number of latest snapshots to keep
        """
        versions = self.get_versions(model_id)

        if len(versions) <= keep_latest:
            return

        # Remove old snapshots
        versions_to_remove = versions[:-keep_latest]

        for version in versions_to_remove:
            snapshot_dir = self.versions_dir / "snapshots" / model_id / version.version

            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
                logger.info(f"Removed snapshot for {model_id} {version.version}")
