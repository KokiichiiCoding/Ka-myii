"""
Model assembler — packages separated layers into a Live2D Cubism-ready project.

The output is everything an artist needs to open the character in **Live2D
Cubism 5** and finish rigging with its AI auto-rig feature:

* ``<name>.psd``        layered artwork (primary import target)
* ``<name>.ora``        open layered fallback
* ``<name>.model3.json`` model definition scaffold (groups, hit areas)
* ``<name>.physics3.json`` auto-generated hair/cloth physics
* ``<name>.cdi3.json``  display info: the standard Cubism parameter + part set
* ``textures/``         flattened texture placeholder
* ``RIGGING_GUIDE.md``  step-by-step finishing instructions

A genuine ``.moc3`` can only be produced by the proprietary Cubism Editor, so
Ka-myii prepares a complete, correctly-structured project right up to that step.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import config
from src.core import psd_exporter
from src.core.auto_physics import AutoPhysicsGenerator
from src.models.vtuber_model import Asset

logger = logging.getLogger(__name__)


def _part_id(layer_type: str) -> str:
    return "Part" + "".join(part.capitalize() for part in layer_type.split("_"))


class ModelAssembler:
    """Assembles separated assets into a Cubism-import-ready project folder."""

    def __init__(self, texture_size: int = None):
        self.texture_size = texture_size or config.MODEL_ASSEMBLY.get("texture_size", 4096)
        self.physics = AutoPhysicsGenerator()
        logger.info("ModelAssembler ready (texture size: %s)", self.texture_size)

    # ------------------------------------------------------------------
    def assemble(self, assets: List[Asset], output_dir: Path, model_name: str = "vtuber_model") -> Path:
        model_dir = Path(output_dir) / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Assembling Cubism project: %s", model_name)

        # 1. Layered art (PSD / ORA / flat) — the rig-ready deliverable.
        produced = psd_exporter.export_layered_documents(assets, model_dir, model_name)

        # 2. Texture placeholder from the flattened composite.
        textures_dir = model_dir / "textures"
        textures_dir.mkdir(exist_ok=True)
        texture_files: List[str] = []
        flat = psd_exporter.compose_flat(assets)
        if flat is not None:
            tex_path = textures_dir / "texture_00.png"
            flat.convert("RGBA").save(tex_path)
            texture_files.append("textures/texture_00.png")

        # 3. Physics from hair / cloth layers.
        physics_path = model_dir / f"{model_name}.physics3.json"
        try:
            self.physics.generate_physics_json(assets, model_name, physics_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Physics generation failed: %s", exc)

        # 4. Display info (parameters + parts) and 5. model definition.
        rig_layers = [a.layer_type for a in assets if a.layer_type != "character"]
        self._write_json(model_dir / f"{model_name}.cdi3.json", self._build_cdi3(rig_layers))
        self._write_json(
            model_dir / f"{model_name}.model3.json",
            self._build_model3(model_name, texture_files, rig_layers),
        )

        # 6. Minimal idle motion so the model3.json reference resolves.
        motions_dir = model_dir / "motions"
        motions_dir.mkdir(exist_ok=True)
        self._write_json(
            motions_dir / "idle.motion3.json",
            {
                "Version": 3,
                "Meta": {
                    "Duration": 4.0,
                    "Fps": 30.0,
                    "Loop": True,
                    "AreBeziersRestricted": True,
                    "CurveCount": 1,
                    "TotalSegmentCount": 2,
                    "TotalPointCount": 3,
                    "UserDataCount": 0,
                    "TotalUserDataSize": 0,
                },
                "Curves": [
                    {
                        "Target": "Parameter",
                        "Id": "ParamBreath",
                        "Segments": [0, 0, 1, 2.0, 1, 1, 4.0, 0],
                    }
                ],
            },
        )

        # 7. Helper manifest + rigging guide.
        self._write_json(
            model_dir / "kamyii_manifest.json",
            {
                "model_name": model_name,
                "layers": rig_layers,
                "artifacts": produced,
                "parameters": [p["Id"] for p in config.MODEL_ASSEMBLY["standard_parameters"]],
            },
        )
        (model_dir / "RIGGING_GUIDE.md").write_text(self._rigging_guide(model_name, rig_layers), encoding="utf-8")

        logger.info("Cubism project assembled at %s", model_dir)
        return model_dir

    # ------------------------------------------------------------------
    def _build_model3(self, model_name: str, textures: List[str], layers: List[str]) -> Dict:
        return {
            "Version": 3,
            "FileReferences": {
                "Moc": f"{model_name}.moc3",  # produced by Cubism Editor on export
                "Textures": textures or ["textures/texture_00.png"],
                "Physics": f"{model_name}.physics3.json",
                "DisplayInfo": f"{model_name}.cdi3.json",
                "Motions": {
                    "Idle": [{"File": "motions/idle.motion3.json"}],
                },
            },
            "Groups": [
                {"Target": "Parameter", "Name": "EyeBlink", "Ids": ["ParamEyeLOpen", "ParamEyeROpen"]},
                {"Target": "Parameter", "Name": "LipSync", "Ids": ["ParamMouthOpenY"]},
            ],
            "HitAreas": [
                {"Id": "HitAreaHead", "Name": "Head"},
                {"Id": "HitAreaBody", "Name": "Body"},
            ],
        }

    def _build_cdi3(self, layers: List[str]) -> Dict:
        parameters = [
            {"Id": p["Id"], "GroupId": "", "Name": p["Name"]}
            for p in config.MODEL_ASSEMBLY["standard_parameters"]
        ]
        parameter_groups = [
            {"Id": "Position", "GroupId": "", "Name": "Position"},
            {"Id": "Eyes", "GroupId": "", "Name": "Eyes"},
            {"Id": "Mouth", "GroupId": "", "Name": "Mouth"},
            {"Id": "Hair", "GroupId": "", "Name": "Hair"},
        ]
        parts = [{"Id": _part_id(layer), "Name": layer.replace("_", " ").title()} for layer in layers]
        return {
            "Version": 3,
            "Parameters": parameters,
            "ParameterGroups": parameter_groups,
            "Parts": parts,
        }

    @staticmethod
    def _write_json(path: Path, data: Dict):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def _rigging_guide(self, model_name: str, layers: List[str]) -> str:
        layer_lines = "\n".join(f"- `{layer}` → part `{_part_id(layer)}`" for layer in layers)
        return f"""# Rigging Guide — {model_name}

This folder is a **Live2D Cubism-ready project**. Ka-myii has decomposed your
character into separated, named layers and generated the supporting Cubism
files. Follow these steps to finish a fully rigged model.

## 1. Open the artwork
1. Install **Live2D Cubism 5** (the free version is fine for learning).
2. `File → Open` and select **`{model_name}.psd`**.
   - If your tool prefers OpenRaster, `{model_name}.ora` is also provided.
3. Cubism imports each layer as a separate **ArtMesh**.

## 2. Layers → Parts
The following layers were exported (back to front):

{layer_lines}

## 3. Auto-rig (Cubism 5 AI)
1. Select the model in the editor.
2. Open **Modeling → Auto-Rig (AI)** — Cubism analyses the layers and creates
   deformers for the eyes, eyebrows, mouth and head angle automatically.
3. Ka-myii has pre-defined the standard parameter set in
   **`{model_name}.cdi3.json`** (ParamAngleX/Y/Z, ParamEyeLOpen/ROpen,
   ParamMouthOpenY/Form, ParamBrowLY/RY, ParamBody*, ParamBreath …) so the
   auto-rig has the right targets to bind to.

## 4. Physics
`{model_name}.physics3.json` contains starter physics for hair and clothing.
Load it via **`Modeling → Physics/Style Sheet Settings → Import`** and tweak
the pendulum strengths to taste.

## 5. Export
`File → Export for Runtime → moc3` produces the binary `.moc3` plus updated
`.model3.json`. That package drops straight into VTube Studio, the Cubism Web
SDK, Unity, etc.

---
*Generated by Ka-myii — Automated VTuber Model Studio.*
"""

    # ------------------------------------------------------------------
    def create_preview_image(self, assets: List[Asset], output_path: Path, size: tuple = None):
        """Composite all layers into a single transparent preview image."""
        flat = psd_exporter.compose_flat(assets)
        if flat is None:
            from PIL import Image

            flat = Image.new("RGBA", size or (512, 768), (0, 0, 0, 0))
        if size:
            flat.thumbnail(size, __import__("PIL").Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        flat.save(output_path)
        logger.info("Preview image created: %s", output_path)


class DummyModelAssembler(ModelAssembler):
    """Demo mode uses the same real assembler — no GPU required for packaging."""

    pass
