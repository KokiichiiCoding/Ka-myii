"""
Starter physics3.json for the Cubism scaffold, from the layers that exist.

One pendulum per group (front hair, back hair, accessories, skirt), shaped like Live2D's own
sample models: head and body sway in, an angle out on a parameter the scaffold declares in
cdi3.json, so the physics drive something once the artist rigs those parameters in Cubism.
Validated against the physics3.json schema in Live2D's CubismSpecs.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List

from src.models.vtuber_model import Asset

logger = logging.getLogger(__name__)

# (group name, output parameter id, parameter name, layer names, pendulum length, mobility, delay, acceleration, scale)
GROUPS = [
    ("Hair Front", "ParamHairFront", "Hair Front", {"front hair", "hair_front"}, 3, 0.95, 0.9, 1.5, 1.5),
    ("Hair Back", "ParamHairBack", "Hair Back", {"back hair", "hair_back"}, 15, 0.95, 0.8, 1.0, 1.2),
    ("Accessories", "ParamAccessorySway", "Accessory Sway",
     {"headwear", "earwear", "neckwear", "tail", "wings", "accessories"}, 8, 0.9, 0.85, 1.2, 1.0),
    ("Skirt", "ParamSkirtSway", "Skirt Sway", {"bottomwear"}, 10, 0.9, 0.8, 1.0, 1.0),
]
# Head and body sway feed every pendulum (ids from the scaffold's standard parameter set).
INPUTS = [("ParamAngleX", "X", 60), ("ParamAngleZ", "Angle", 60),
          ("ParamBodyAngleX", "X", 40), ("ParamBodyAngleZ", "Angle", 40)]


class AutoPhysicsGenerator:
    def generate_physics_json(self, assets: List[Asset], model_name: str, output_path: Path) -> List[Dict]:
        """Write physics3.json; returns the parameters it drives ({"Id", "Name"}) for cdi3.json."""
        present = {asset.layer_type.lower() for asset in assets}
        settings, dictionary, parameters = [], [], []
        for name, param, param_name, layers, length, mobility, delay, acceleration, scale in GROUPS:
            if not present & layers:
                continue
            setting_id = f"PhysicsSetting{len(settings) + 1}"
            dictionary.append({"Id": setting_id, "Name": name})
            parameters.append({"Id": param, "Name": param_name})
            settings.append({
                "Id": setting_id,
                "Input": [{"Source": {"Target": "Parameter", "Id": source}, "Weight": weight,
                           "Type": kind, "Reflect": False} for source, kind, weight in INPUTS],
                "Output": [{"Destination": {"Target": "Parameter", "Id": param}, "VertexIndex": 1,
                            "Scale": scale, "Weight": 100, "Type": "Angle", "Reflect": False}],
                "Vertices": [
                    {"Position": {"X": 0, "Y": 0}, "Mobility": 1, "Delay": 1, "Acceleration": 1, "Radius": 0},
                    {"Position": {"X": 0, "Y": length}, "Mobility": mobility, "Delay": delay,
                     "Acceleration": acceleration, "Radius": length},
                ],
                "Normalization": {"Position": {"Minimum": -10, "Default": 0, "Maximum": 10},
                                  "Angle": {"Minimum": -10, "Default": 0, "Maximum": 10}},
            })

        physics = {
            "Version": 3,
            "Meta": {
                "PhysicsSettingCount": len(settings),
                "TotalInputCount": sum(len(s["Input"]) for s in settings),
                "TotalOutputCount": sum(len(s["Output"]) for s in settings),
                "VertexCount": sum(len(s["Vertices"]) for s in settings),
                "Fps": 30,
                "EffectiveForces": {"Gravity": {"X": 0, "Y": -1}, "Wind": {"X": 0, "Y": 0}},
                "PhysicsDictionary": dictionary,
            },
            "PhysicsSettings": settings,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(physics, indent=2), encoding="utf-8")
        logger.info("Physics: %d pendulum(s) -> %s", len(settings), output_path.name)
        return parameters
