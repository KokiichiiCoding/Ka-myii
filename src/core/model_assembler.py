"""
Model assembly module for creating Live2D compatible model files
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import shutil

from src.models.vtuber_model import Asset

logger = logging.getLogger(__name__)


class ModelAssembler:
    """
    Assembles separated assets into a Live2D compatible model
    """

    def __init__(self, texture_size: int = 2048):
        """
        Initialize the model assembler

        Args:
            texture_size: Size of the texture atlas
        """
        self.texture_size = texture_size
        logger.info(f"ModelAssembler initialized (texture size: {texture_size})")

    def assemble(
        self,
        assets: List[Asset],
        output_dir: Path,
        model_name: str = "vtuber_model"
    ) -> Path:
        """
        Assemble assets into a Live2D model

        Args:
            assets: List of assets to assemble
            output_dir: Output directory for the model
            model_name: Name of the model

        Returns:
            Path to the assembled model directory
        """
        logger.info(f"Assembling model: {model_name}")

        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create model structure
        textures_dir = model_dir / "textures"
        textures_dir.mkdir(exist_ok=True)

        # Copy assets to textures directory
        texture_mapping = {}
        for i, asset in enumerate(assets):
            texture_name = f"{asset.layer_type}.png"
            texture_path = textures_dir / texture_name
            shutil.copy(asset.file_path, texture_path)
            texture_mapping[asset.layer_type] = texture_name
            logger.info(f"Copied asset: {asset.layer_type}")

        # Generate model metadata
        model_json = self._generate_model_json(model_name, texture_mapping, assets)

        # Save model JSON
        model_json_path = model_dir / f"{model_name}.model3.json"
        with open(model_json_path, 'w') as f:
            json.dump(model_json, f, indent=2)

        logger.info(f"Model assembled: {model_dir}")

        # Generate physics if needed
        physics_json = self._generate_physics_json(model_name, assets)
        physics_json_path = model_dir / f"{model_name}.physics3.json"
        with open(physics_json_path, 'w') as f:
            json.dump(physics_json, f, indent=2)

        # Generate motion definitions
        self._generate_motion_definitions(model_dir, model_name)

        return model_dir

    def _generate_model_json(
        self,
        model_name: str,
        texture_mapping: Dict[str, str],
        assets: List[Asset]
    ) -> Dict:
        """
        Generate Live2D model3.json file

        Args:
            model_name: Model name
            texture_mapping: Mapping of layer types to texture files
            assets: List of assets

        Returns:
            Model JSON dictionary
        """
        # This is a simplified Live2D model3.json structure
        # A full implementation would need more detailed configuration

        model_json = {
            "Version": 3,
            "FileReferences": {
                "Moc": f"{model_name}.moc3",
                "Textures": [f"textures/{tex}" for tex in texture_mapping.values()],
                "Physics": f"{model_name}.physics3.json",
                "DisplayInfo": f"{model_name}.cdi3.json"
            },
            "Groups": self._generate_layer_groups(assets),
            "HitAreas": [
                {"Name": "Head", "Id": "HitAreaHead"},
                {"Name": "Body", "Id": "HitAreaBody"}
            ]
        }

        return model_json

    def _generate_layer_groups(self, assets: List[Asset]) -> List[Dict]:
        """Generate layer groupings"""
        groups = []

        # Group by layer type
        layer_groups = {}
        for asset in assets:
            layer_type = asset.layer_type
            if layer_type not in layer_groups:
                layer_groups[layer_type] = []
            layer_groups[layer_type].append(asset)

        # Create group definitions
        for group_name, group_assets in layer_groups.items():
            groups.append({
                "Target": "Parameter",
                "Name": group_name,
                "Ids": [f"Part{group_name.capitalize()}"]
            })

        return groups

    def _generate_physics_json(self, model_name: str, assets: List[Asset]) -> Dict:
        """
        Generate Live2D physics3.json file

        Args:
            model_name: Model name
            assets: List of assets

        Returns:
            Physics JSON dictionary
        """
        # Simplified physics configuration
        physics_json = {
            "Version": 3,
            "Meta": {
                "PhysicsSettingCount": 2,
                "TotalInputCount": 4,
                "TotalOutputCount": 4,
                "VertexCount": 8
            },
            "PhysicsSettings": [
                {
                    "Id": "PhysicsSetting1",
                    "Input": [
                        {"Source": {"Target": "Parameter", "Id": "ParamAngleX"}, "Weight": 60, "Type": "X"}
                    ],
                    "Output": [
                        {"Destination": {"Target": "Parameter", "Id": "ParamHairFront"}, "VertexIndex": 1, "Scale": 10, "Weight": 100, "Type": "Angle"}
                    ],
                    "Vertices": [
                        {"Position": {"X": 0, "Y": 0}, "Mobility": 1, "Delay": 1, "Acceleration": 1, "Radius": 0}
                    ],
                    "Normalization": {
                        "Position": {"Minimum": -10, "Default": 0, "Maximum": 10},
                        "Angle": {"Minimum": -10, "Default": 0, "Maximum": 10}
                    }
                }
            ]
        }

        return physics_json

    def _generate_motion_definitions(self, model_dir: Path, model_name: str):
        """
        Generate motion definition files

        Args:
            model_dir: Model directory
            model_name: Model name
        """
        motions_dir = model_dir / "motions"
        motions_dir.mkdir(exist_ok=True)

        # Create idle motion placeholder
        idle_motion = {
            "Version": 3,
            "Meta": {
                "Duration": 2.0,
                "Fps": 30.0,
                "Loop": True,
                "CurveCount": 2,
                "TotalSegmentCount": 4,
                "TotalPointCount": 8
            },
            "Curves": []
        }

        idle_motion_path = motions_dir / "idle.motion3.json"
        with open(idle_motion_path, 'w') as f:
            json.dump(idle_motion, f, indent=2)

        logger.info("Generated motion definitions")

    def create_preview_image(
        self,
        assets: List[Asset],
        output_path: Path,
        size: tuple = (512, 512)
    ):
        """
        Create a preview image of the model

        Args:
            assets: List of assets
            output_path: Path to save preview
            size: Size of preview image
        """
        from PIL import Image

        # Create composite image
        preview = Image.new('RGBA', size, (0, 0, 0, 0))

        # Layer assets in order
        layer_order = ["background", "body", "hair_back", "head", "eyes", "mouth", "hair_front", "accessories"]

        for layer_name in layer_order:
            for asset in assets:
                if asset.layer_type == layer_name or asset.layer_type.startswith(layer_name):
                    try:
                        layer_img = Image.open(asset.file_path)
                        layer_img = layer_img.resize(size, Image.Resampling.LANCZOS)
                        preview = Image.alpha_composite(preview, layer_img.convert('RGBA'))
                    except Exception as e:
                        logger.warning(f"Failed to add layer {asset.layer_type} to preview: {e}")

        preview.save(output_path)
        logger.info(f"Preview image created: {output_path}")


class DummyModelAssembler(ModelAssembler):
    """Dummy assembler for testing"""

    def assemble(
        self,
        assets: List[Asset],
        output_dir: Path,
        model_name: str = "vtuber_model"
    ) -> Path:
        """Create dummy model structure"""
        logger.info("DummyModelAssembler: Creating placeholder model")

        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Just create a simple marker file
        marker = model_dir / "model.json"
        marker.write_text(json.dumps({"name": model_name, "assets": len(assets)}))

        return model_dir
