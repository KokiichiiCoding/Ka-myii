"""
Data models for VTuber generation
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional
import uuid


class GenerationStatus(Enum):
    """Status of model generation"""
    PENDING = "pending"
    IMAGE_GENERATION = "image_generation"
    ASSET_SEPARATION = "asset_separation"
    MODEL_ASSEMBLY = "model_assembly"
    RIGGING = "rigging"
    COMPLETED = "completed"
    FAILED = "failed"


class AssetLayer(Enum):
    """Types of asset layers"""
    BACKGROUND = "background"
    BODY = "body"
    HEAD = "head"
    EYES = "eyes"
    MOUTH = "mouth"
    HAIR_BACK = "hair_back"
    HAIR_FRONT = "hair_front"
    ACCESSORIES = "accessories"
    CLOTHING = "clothing"


@dataclass
class GenerationRequest:
    """Request for generating a VTuber model"""
    prompt: str
    negative_prompt: str = ""
    style: str = "anime"
    width: int = 512
    height: int = 512
    steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None

    # Advanced options
    include_rigging: bool = False
    custom_layers: Optional[List[str]] = None
    reference_image: Optional[str] = None


@dataclass
class Asset:
    """Represents a single asset/layer"""
    layer_type: str
    file_path: Path
    metadata: Dict = field(default_factory=dict)


@dataclass
class VTuberModel:
    """Complete VTuber model data"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    status: GenerationStatus = GenerationStatus.PENDING

    # Generation parameters
    request: Optional[GenerationRequest] = None

    # Generated assets
    base_image_path: Optional[Path] = None
    assets: List[Asset] = field(default_factory=list)
    final_model_path: Optional[Path] = None

    # Metadata
    generation_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "base_image_path": str(self.base_image_path) if self.base_image_path else None,
            "assets": [
                {
                    "layer_type": asset.layer_type,
                    "file_path": str(asset.file_path),
                    "metadata": asset.metadata
                }
                for asset in self.assets
            ],
            "final_model_path": str(self.final_model_path) if self.final_model_path else None,
            "generation_time": self.generation_time,
            "error_message": self.error_message,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'VTuberModel':
        """Create from dictionary"""
        model = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            status=GenerationStatus(data.get("status", "pending")),
        )

        if data.get("base_image_path"):
            model.base_image_path = Path(data["base_image_path"])

        if data.get("final_model_path"):
            model.final_model_path = Path(data["final_model_path"])

        model.assets = [
            Asset(
                layer_type=asset_data["layer_type"],
                file_path=Path(asset_data["file_path"]),
                metadata=asset_data.get("metadata", {})
            )
            for asset_data in data.get("assets", [])
        ]

        model.generation_time = data.get("generation_time", 0.0)
        model.error_message = data.get("error_message")
        model.metadata = data.get("metadata", {})

        return model
