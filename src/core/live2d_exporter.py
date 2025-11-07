"""
Live2D Export Helper
Generates proper Live2D Cubism-compatible packages
"""
import logging
import json
import zipfile
from pathlib import Path
from typing import List, Dict, Optional
import shutil

from src.models.vtuber_model import VTuberModel, Asset
from src.core.auto_physics import AutoPhysicsGenerator

logger = logging.getLogger(__name__)


class Live2DExporter:
    """
    Export models in Live2D Cubism format
    """

    def __init__(self):
        self.physics_generator = AutoPhysicsGenerator()
        logger.info("Live2DExporter initialized")

    def export_for_live2d(
        self,
        model: VTuberModel,
        output_path: Path,
        include_expressions: bool = True,
        include_physics: bool = True,
        include_motions: bool = True
    ) -> Path:
        """
        Export complete Live2D package

        Args:
            model: VTuber model to export
            output_path: Path for export ZIP file
            include_expressions: Include expression files
            include_physics: Include physics configuration
            include_motions: Include motion files

        Returns:
            Path to exported ZIP file
        """
        logger.info(f"Exporting Live2D package for model: {model.id}")

        # Create temporary directory for package
        temp_dir = output_path.parent / f"temp_export_{model.id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            model_dir = temp_dir / model.name

            model_dir.mkdir(exist_ok=True)

            # Create directory structure
            textures_dir = model_dir / "textures"
            expressions_dir = model_dir / "expressions"
            motions_dir = model_dir / "motions"

            textures_dir.mkdir(exist_ok=True)

            # Copy textures
            self._copy_textures(model, textures_dir)

            # Generate model3.json
            model3_path = model_dir / f"{model.name}.model3.json"
            self._generate_model3_json(model, model3_path, textures_dir)

            # Generate physics
            if include_physics and model.assets:
                physics_path = model_dir / f"{model.name}.physics3.json"
                self.physics_generator.generate_physics_json(
                    model.assets,
                    model.name,
                    physics_path
                )

            # Copy expressions
            if include_expressions:
                expressions_dir.mkdir(exist_ok=True)
                self._copy_expressions(model, expressions_dir)

            # Generate motions
            if include_motions:
                motions_dir.mkdir(exist_ok=True)
                self._generate_default_motions(model, motions_dir)

            # Generate display info
            self._generate_display_info(model, model_dir)

            # Generate user data
            self._generate_user_data(model, model_dir)

            # Create ZIP package
            self._create_zip_package(model_dir, output_path)

            logger.info(f"Live2D package exported: {output_path}")

            return output_path

        finally:
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _copy_textures(self, model: VTuberModel, textures_dir: Path):
        """Copy texture files to export directory"""
        if not model.assets:
            return

        for asset in model.assets:
            if asset.file_path.exists():
                dest = textures_dir / asset.file_path.name
                shutil.copy2(asset.file_path, dest)

    def _generate_model3_json(
        self,
        model: VTuberModel,
        output_path: Path,
        textures_dir: Path
    ):
        """Generate complete model3.json file"""

        texture_files = [f.name for f in textures_dir.glob("*.png")]

        model3 = {
            "Version": 3,
            "FileReferences": {
                "Moc": f"{model.name}.moc3",
                "Textures": [f"textures/{tex}" for tex in texture_files],
                "Physics": f"{model.name}.physics3.json",
                "DisplayInfo": f"{model.name}.cdi3.json",
                "UserData": f"{model.name}.userdata3.json",
                "Motions": {
                    "Idle": [
                        {"File": "motions/idle.motion3.json"}
                    ],
                    "TapBody": [
                        {"File": "motions/tap_body.motion3.json"}
                    ]
                },
                "Expressions": [
                    {"Name": expr, "File": f"expressions/{expr}.exp3.json"}
                    for expr in self._get_expression_names(model)
                ]
            },
            "Groups": self._generate_groups(model),
            "HitAreas": [
                {"Id": "HitAreaHead", "Name": "Head"},
                {"Id": "HitAreaBody", "Name": "Body"}
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(model3, f, indent=2)

        logger.info(f"Generated model3.json: {output_path}")

    def _generate_groups(self, model: VTuberModel) -> List[Dict]:
        """Generate parameter groups"""
        groups = [
            {
                "Target": "Parameter",
                "Name": "EyeBlink",
                "Ids": ["ParamEyeLOpen", "ParamEyeROpen"]
            },
            {
                "Target": "Parameter",
                "Name": "LipSync",
                "Ids": ["ParamMouthOpenY"]
            }
        ]

        # Add groups for physics parts
        if model.assets:
            physics_parts = set()
            for asset in model.assets:
                if any(keyword in asset.layer_type.lower() for keyword in ["hair", "clothing"]):
                    physics_parts.add(asset.layer_type)

            for part in physics_parts:
                groups.append({
                    "Target": "Parameter",
                    "Name": part.title(),
                    "Ids": [f"Param{part.replace('_', '').title()}"]
                })

        return groups

    def _get_expression_names(self, model: VTuberModel) -> List[str]:
        """Get list of expression names from metadata"""
        expr_metadata = model.metadata.get("expressions", {})
        if isinstance(expr_metadata, dict):
            return expr_metadata.get("expressions", [])
        return []

    def _copy_expressions(self, model: VTuberModel, expressions_dir: Path):
        """Copy expression files and generate exp3.json files"""
        expr_names = self._get_expression_names(model)

        for expr_name in expr_names:
            exp3_path = expressions_dir / f"{expr_name}.exp3.json"

            # Generate expression JSON
            exp3_data = {
                "Type": "Live2D Expression",
                "FadeInTime": 0.5,
                "FadeOutTime": 0.5,
                "Parameters": self._generate_expression_parameters(expr_name)
            }

            with open(exp3_path, 'w') as f:
                json.dump(exp3_data, f, indent=2)

    def _generate_expression_parameters(self, expression_name: str) -> List[Dict]:
        """Generate parameter changes for an expression"""
        # Map expressions to parameter changes
        expression_params = {
            "happy": [
                {"Id": "ParamEyeLOpen", "Value": 1.0},
                {"Id": "ParamEyeROpen", "Value": 1.0},
                {"Id": "ParamMouthForm", "Value": 1.0},
                {"Id": "ParamMouthOpenY", "Value": 0.6}
            ],
            "sad": [
                {"Id": "ParamEyeLOpen", "Value": 0.5},
                {"Id": "ParamEyeROpen", "Value": 0.5},
                {"Id": "ParamMouthForm", "Value": -1.0}
            ],
            "angry": [
                {"Id": "ParamBrowLY", "Value": -1.0},
                {"Id": "ParamBrowRY", "Value": -1.0},
                {"Id": "ParamMouthForm", "Value": -0.8}
            ],
            "surprised": [
                {"Id": "ParamEyeLOpen", "Value": 1.5},
                {"Id": "ParamEyeROpen", "Value": 1.5},
                {"Id": "ParamMouthOpenY", "Value": 1.0}
            ]
        }

        return expression_params.get(expression_name, [])

    def _generate_default_motions(self, model: VTuberModel, motions_dir: Path):
        """Generate default motion files"""
        # Idle motion
        idle_motion = {
            "Version": 3,
            "Meta": {
                "Duration": 3.0,
                "Fps": 30.0,
                "Loop": True,
                "AreBeziersRestricted": True,
                "CurveCount": 2,
                "TotalSegmentCount": 4,
                "TotalPointCount": 12
            },
            "Curves": [
                {
                    "Target": "Parameter",
                    "Id": "ParamEyeLOpen",
                    "Segments": [0.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0, 0.0]
                },
                {
                    "Target": "Parameter",
                    "Id": "ParamEyeROpen",
                    "Segments": [0.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0, 0.0]
                }
            ]
        }

        with open(motions_dir / "idle.motion3.json", 'w') as f:
            json.dump(idle_motion, f, indent=2)

        # Tap body motion
        tap_motion = {
            "Version": 3,
            "Meta": {
                "Duration": 0.5,
                "Fps": 30.0,
                "Loop": False,
                "AreBeziersRestricted": True,
                "CurveCount": 1,
                "TotalSegmentCount": 2,
                "TotalPointCount": 6
            },
            "Curves": [
                {
                    "Target": "Parameter",
                    "Id": "ParamBodyAngleX",
                    "Segments": [0.0, 0.0, 0.25, 10.0, 0.5, 0.0]
                }
            ]
        }

        with open(motions_dir / "tap_body.motion3.json", 'w') as f:
            json.dump(tap_motion, f, indent=2)

    def _generate_display_info(self, model: VTuberModel, model_dir: Path):
        """Generate display info file"""
        cdi3 = {
            "Version": 3,
            "Parameters": [
                {
                    "Id": "ParamEyeLOpen",
                    "GroupId": "",
                    "Name": "Left Eye Open"
                },
                {
                    "Id": "ParamEyeROpen",
                    "GroupId": "",
                    "Name": "Right Eye Open"
                },
                {
                    "Id": "ParamMouthOpenY",
                    "GroupId": "",
                    "Name": "Mouth Open"
                }
            ],
            "ParameterGroups": [],
            "Parts": []
        }

        cdi3_path = model_dir / f"{model.name}.cdi3.json"
        with open(cdi3_path, 'w') as f:
            json.dump(cdi3, f, indent=2)

    def _generate_user_data(self, model: VTuberModel, model_dir: Path):
        """Generate user data file"""
        userdata = {
            "Version": 3,
            "Meta": {
                "UserDataCount": 0,
                "TotalUserDataSize": 0
            },
            "UserData": []
        }

        userdata_path = model_dir / f"{model.name}.userdata3.json"
        with open(userdata_path, 'w') as f:
            json.dump(userdata, f, indent=2)

    def _create_zip_package(self, model_dir: Path, output_path: Path):
        """Create ZIP package of the model"""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in model_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(model_dir.parent)
                    zipf.write(file_path, arcname)

        logger.info(f"Created ZIP package: {output_path}")
