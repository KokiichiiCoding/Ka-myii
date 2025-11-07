"""
Automatic physics setup for Live2D models
Generates physics3.json based on detected asset types
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json

from src.models.vtuber_model import Asset

logger = logging.getLogger(__name__)


class AutoPhysicsGenerator:
    """
    Automatically generates physics configurations for Live2D models
    """

    # Physics parameters for different asset types
    PHYSICS_PRESETS = {
        "hair_back": {
            "input_weight": 60,
            "input_type": "X",
            "output_scale": 10,
            "output_weight": 100,
            "output_type": "Angle",
            "mobility": 0.95,
            "delay": 0.9,
            "acceleration": 0.9,
            "radius": 10
        },
        "hair_front": {
            "input_weight": 50,
            "input_type": "X",
            "output_scale": 8,
            "output_weight": 80,
            "output_type": "Angle",
            "mobility": 0.90,
            "delay": 0.85,
            "acceleration": 0.85,
            "radius": 8
        },
        "hair_side": {
            "input_weight": 55,
            "input_type": "X",
            "output_scale": 9,
            "output_weight": 90,
            "output_type": "Angle",
            "mobility": 0.92,
            "delay": 0.88,
            "acceleration": 0.88,
            "radius": 9
        },
        "clothing": {
            "input_weight": 40,
            "input_type": "Y",
            "output_scale": 6,
            "output_weight": 70,
            "output_type": "Angle",
            "mobility": 0.85,
            "delay": 0.80,
            "acceleration": 0.80,
            "radius": 6
        },
        "sleeve": {
            "input_weight": 45,
            "input_type": "X",
            "output_scale": 7,
            "output_weight": 75,
            "output_type": "Angle",
            "mobility": 0.88,
            "delay": 0.82,
            "acceleration": 0.82,
            "radius": 7
        },
        "accessory": {
            "input_weight": 30,
            "input_type": "X",
            "output_scale": 5,
            "output_weight": 60,
            "output_type": "Angle",
            "mobility": 0.80,
            "delay": 0.75,
            "acceleration": 0.75,
            "radius": 5
        },
        "ribbon": {
            "input_weight": 35,
            "input_type": "X",
            "output_scale": 12,
            "output_weight": 85,
            "output_type": "Angle",
            "mobility": 0.92,
            "delay": 0.90,
            "acceleration": 0.90,
            "radius": 8
        },
        "skirt": {
            "input_weight": 50,
            "input_type": "Y",
            "output_scale": 8,
            "output_weight": 80,
            "output_type": "Angle",
            "mobility": 0.90,
            "delay": 0.85,
            "acceleration": 0.85,
            "radius": 10
        }
    }

    def __init__(self):
        logger.info("AutoPhysicsGenerator initialized")

    def generate_physics_json(
        self,
        assets: List[Asset],
        model_name: str,
        output_path: Path
    ) -> Path:
        """
        Generate physics3.json file based on assets

        Args:
            assets: List of model assets
            model_name: Name of the model
            output_path: Where to save physics3.json

        Returns:
            Path to generated physics file
        """
        logger.info(f"Generating physics for {len(assets)} assets")

        # Detect physics-enabled assets
        physics_settings = []
        setting_id = 1

        for asset in assets:
            asset_type = self._classify_asset_for_physics(asset)

            if asset_type:
                preset = self.PHYSICS_PRESETS.get(asset_type)

                if preset:
                    setting = self._create_physics_setting(
                        setting_id,
                        asset.layer_type,
                        asset_type,
                        preset
                    )
                    physics_settings.append(setting)
                    setting_id += 1

        # Build physics JSON structure
        physics_json = {
            "Version": 3,
            "Meta": {
                "PhysicsSettingCount": len(physics_settings),
                "TotalInputCount": len(physics_settings),
                "TotalOutputCount": len(physics_settings),
                "VertexCount": len(physics_settings) * 2
            },
            "PhysicsSettings": physics_settings
        }

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(physics_json, f, indent=2)

        logger.info(f"Physics file generated: {output_path}")
        logger.info(f"Created {len(physics_settings)} physics settings")

        return output_path

    def _classify_asset_for_physics(self, asset: Asset) -> Optional[str]:
        """
        Classify asset type for physics application

        Args:
            asset: Asset to classify

        Returns:
            Physics type or None if not physics-enabled
        """
        layer_type = asset.layer_type.lower()

        # Keyword matching
        if "hair" in layer_type:
            if "back" in layer_type:
                return "hair_back"
            elif "front" in layer_type:
                return "hair_front"
            elif "side" in layer_type:
                return "hair_side"
            else:
                return "hair_back"  # Default for unspecified hair

        elif "clothing" in layer_type or "cloth" in layer_type:
            return "clothing"

        elif "sleeve" in layer_type:
            return "sleeve"

        elif "skirt" in layer_type:
            return "skirt"

        elif "ribbon" in layer_type or "bow" in layer_type:
            return "ribbon"

        elif any(keyword in layer_type for keyword in ["ear", "tail", "wing", "horn"]):
            return "accessory"

        # No physics for this asset
        return None

    def _create_physics_setting(
        self,
        setting_id: int,
        part_name: str,
        physics_type: str,
        preset: Dict
    ) -> Dict:
        """
        Create a physics setting dictionary

        Args:
            setting_id: Unique setting ID
            part_name: Name of the part
            physics_type: Type of physics to apply
            preset: Physics preset parameters

        Returns:
            Physics setting dictionary
        """
        return {
            "Id": f"PhysicsSetting{setting_id}",
            "Input": [
                {
                    "Source": {
                        "Target": "Parameter",
                        "Id": "ParamAngleX"
                    },
                    "Weight": preset["input_weight"],
                    "Type": preset["input_type"],
                    "Reflect": False
                }
            ],
            "Output": [
                {
                    "Destination": {
                        "Target": "Parameter",
                        "Id": f"Param{part_name.replace('_', '').title()}"
                    },
                    "VertexIndex": 1,
                    "Scale": preset["output_scale"],
                    "Weight": preset["output_weight"],
                    "Type": preset["output_type"],
                    "Reflect": False
                }
            ],
            "Vertices": [
                {
                    "Position": {"X": 0, "Y": 0},
                    "Mobility": preset["mobility"],
                    "Delay": preset["delay"],
                    "Acceleration": preset["acceleration"],
                    "Radius": preset["radius"]
                },
                {
                    "Position": {"X": 0, "Y": 10},
                    "Mobility": preset["mobility"],
                    "Delay": preset["delay"],
                    "Acceleration": preset["acceleration"],
                    "Radius": preset["radius"]
                }
            ],
            "Normalization": {
                "Position": {
                    "Minimum": -10,
                    "Default": 0,
                    "Maximum": 10
                },
                "Angle": {
                    "Minimum": -10,
                    "Default": 0,
                    "Maximum": 10
                }
            }
        }

    def add_custom_physics(
        self,
        physics_json_path: Path,
        part_name: str,
        physics_params: Dict
    ) -> Path:
        """
        Add custom physics to existing physics file

        Args:
            physics_json_path: Path to existing physics3.json
            part_name: Name of the part to add physics to
            physics_params: Custom physics parameters

        Returns:
            Updated physics file path
        """
        # Load existing
        with open(physics_json_path, 'r') as f:
            physics_data = json.load(f)

        # Add new setting
        new_id = len(physics_data["PhysicsSettings"]) + 1
        new_setting = self._create_physics_setting(
            new_id,
            part_name,
            "custom",
            physics_params
        )

        physics_data["PhysicsSettings"].append(new_setting)
        physics_data["Meta"]["PhysicsSettingCount"] += 1
        physics_data["Meta"]["TotalInputCount"] += 1
        physics_data["Meta"]["TotalOutputCount"] += 1
        physics_data["Meta"]["VertexCount"] += 2

        # Save
        with open(physics_json_path, 'w') as f:
            json.dump(physics_data, f, indent=2)

        logger.info(f"Added custom physics for {part_name}")
        return physics_json_path

    def generate_physics_preview_data(self, assets: List[Asset]) -> Dict:
        """
        Generate preview data showing which assets will have physics

        Args:
            assets: List of assets

        Returns:
            Dictionary with physics preview info
        """
        preview = {
            "total_assets": len(assets),
            "physics_enabled": [],
            "no_physics": []
        }

        for asset in assets:
            physics_type = self._classify_asset_for_physics(asset)

            if physics_type:
                preview["physics_enabled"].append({
                    "asset": asset.layer_type,
                    "physics_type": physics_type,
                    "preset": self.PHYSICS_PRESETS.get(physics_type, {})
                })
            else:
                preview["no_physics"].append(asset.layer_type)

        return preview
