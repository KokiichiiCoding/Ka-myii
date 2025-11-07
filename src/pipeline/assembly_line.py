"""
Assembly line pipeline for automated VTuber model generation
"""
import logging
import time
from pathlib import Path
from typing import Optional, Callable, Dict, List
import uuid

import config
from src.models.vtuber_model import (
    VTuberModel,
    GenerationRequest,
    GenerationStatus,
    Asset
)
from src.core.image_generator import ImageGenerator, DummyImageGenerator
from src.core.asset_separator import AssetSeparator, DummyAssetSeparator
from src.core.model_assembler import ModelAssembler, DummyModelAssembler
from src.core.rigging import AutoRigger
from src.core.sam_segmentation import SAMSegmentator
from src.core.expression_generator import ExpressionGenerator
from src.core.controlnet_expression import ControlNetExpressionGenerator
from src.core.accessory_generator import AccessoryGenerator, AccessoryType, Accessory
from src.core.auto_physics import AutoPhysicsGenerator
from src.utils.helpers import save_json

logger = logging.getLogger(__name__)


class AssemblyLine:
    """
    Orchestrates the complete VTuber model generation pipeline

    Pipeline stages:
    1. Image Generation - Create base character image
    2. Asset Separation - Split into layers
    3. Model Assembly - Combine into Live2D format
    4. Auto-Rigging (optional) - Add rigging and physics
    """

    def __init__(
        self,
        output_base_dir: Path,
        use_dummy_generators: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize the assembly line

        Args:
            output_base_dir: Base directory for outputs
            use_dummy_generators: Use dummy generators for testing
            progress_callback: Callback function for progress updates
        """
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        self.progress_callback = progress_callback

        # Initialize components
        if use_dummy_generators:
            logger.info("Using dummy generators for testing")
            self.image_generator = DummyImageGenerator()
            self.asset_separator = DummyAssetSeparator()
            self.model_assembler = DummyModelAssembler()
        else:
            self.image_generator = ImageGenerator()
            self.asset_separator = AssetSeparator()
            self.model_assembler = ModelAssembler()

        self.auto_rigger = AutoRigger()
        self.sam_segmentator = None if use_dummy_generators else SAMSegmentator()
        self.expression_generator = ExpressionGenerator()
        self.controlnet_expression_generator = ControlNetExpressionGenerator(use_controlnet=not use_dummy_generators)
        self.accessory_generator = AccessoryGenerator()
        self.auto_physics_generator = AutoPhysicsGenerator()
        self.pipeline_profiles = config.PIPELINE_PROFILES

        logger.info("AssemblyLine initialized")

    def generate_model(self, request: GenerationRequest, task_id: Optional[str] = None) -> VTuberModel:
        """
        Generate a complete VTuber model from a request

        Args:
            request: Generation request
            task_id: Optional task ID for progress tracking

        Returns:
            Generated VTuber model
        """
        start_time = time.time()

        # Create model instance
        model = VTuberModel(
            name=f"model_{uuid.uuid4().hex[:8]}",
            request=request
        )

        # Use task_id or model_id for progress tracking
        if not task_id:
            task_id = f"gen_{model.id[:8]}"

        # Store task_id in model metadata
        model.metadata["task_id"] = task_id

        profile = self._resolve_profile(request.pipeline_profile)
        segmentation_mode = self._resolve_segmentation_mode(request, profile)
        controlnet_enabled = self._should_use_controlnet(request, profile)
        expressions_enabled = self._should_generate_expressions(request, profile)
        accessories_enabled = self._should_generate_accessories(request, profile)
        auto_physics_enabled = self._should_enable_physics(request, profile)

        model.metadata.update({
            "pipeline_profile": request.pipeline_profile or "standard",
            "pipeline_profile_label": profile.get("label"),
            "profile_description": profile.get("description"),
            "segmentation_mode": segmentation_mode,
            "controlnet_enabled": controlnet_enabled,
            "controlnet_active": False,
            "expressions_enabled": expressions_enabled,
            "accessories_enabled": accessories_enabled,
            "auto_physics_enabled": auto_physics_enabled,
        })

        # Create output directory for this model
        model_output_dir = self.output_base_dir / model.id
        model_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting generation for model: {model.id} (task: {task_id})")

        try:
            # Stage 1: Image Generation
            self._update_progress("Generating base image...", 0.1)
            model.status = GenerationStatus.IMAGE_GENERATION
            base_image_path = self._generate_image(request, model_output_dir, task_id)
            model.base_image_path = base_image_path
            logger.info(f"Base image generated: {base_image_path}")

            # Stage 2: Asset Separation
            self._update_progress("Separating assets into layers...", 0.4)
            model.status = GenerationStatus.ASSET_SEPARATION
            assets, segmentation_report = self._separate_assets(
                request,
                base_image_path,
                model_output_dir,
                segmentation_mode
            )
            model.assets = assets
            logger.info(f"Assets separated: {len(assets)} layers")
            if segmentation_report:
                model.metadata["segmentation_report"] = segmentation_report

            accessories_generated = []
            if accessories_enabled:
                self._update_progress("Designing accessories...", 0.55)
                accessories_generated = self._generate_accessories(
                    request,
                    base_image_path,
                    model_output_dir
                )
                if accessories_generated:
                    assets.extend(accessories_generated)
                    model.assets = assets
                    model.metadata["accessories"] = [
                        {
                            "layer_type": asset.layer_type,
                            "file_path": str(asset.file_path),
                            "metadata": asset.metadata,
                        }
                        for asset in accessories_generated
                    ]

            # Stage 3: Model Assembly
            self._update_progress("Assembling Live2D model...", 0.7)
            model.status = GenerationStatus.MODEL_ASSEMBLY
            final_model_path = self._assemble_model(assets, model_output_dir, model.name)
            model.final_model_path = final_model_path
            logger.info(f"Model assembled: {final_model_path}")

            # Optional Stage: Expression Set
            if expressions_enabled:
                self._update_progress("Synthesising expression set...", 0.85)
                expression_metadata = self._generate_expressions(
                    request,
                    base_image_path,
                    model_output_dir,
                    controlnet_enabled
                )
                if expression_metadata:
                    model.metadata["expressions"] = expression_metadata
                    if expression_metadata.get("controlnet_variations"):
                        has_variations = any(
                            expression_metadata["controlnet_variations"].get(name)
                            for name in expression_metadata.get("controlnet_variations", {})
                        )
                        if has_variations:
                            model.metadata["controlnet_active"] = True

            # Stage 4: Auto-Rigging (optional)
            if request.include_rigging:
                self._update_progress("Applying auto-rigging...", 0.9)
                model.status = GenerationStatus.RIGGING
                rigging_data = self._apply_rigging(final_model_path, assets)
                model.metadata["rigging"] = rigging_data
                logger.info("Rigging applied")

            # Optional Stage: Auto physics enhancements
            physics_path = None
            if auto_physics_enabled:
                self._update_progress("Authoring physics presets...", 0.93)
                physics_path = self._enhance_physics(final_model_path, model.assets)
                if physics_path:
                    model.metadata["physics_path"] = str(physics_path)

            # Complete
            model.status = GenerationStatus.COMPLETED
            model.generation_time = time.time() - start_time
            self._update_progress("Model generation complete!", 1.0)

            logger.info(f"Model generation completed in {model.generation_time:.2f}s")

            # Generate preview
            self._generate_preview(model, model_output_dir)

            # Persist a pipeline summary for the UI / downloads
            summary_path = self._write_summary(model, model_output_dir)
            if summary_path:
                model.metadata["summary_path"] = str(summary_path)

            return model

        except Exception as e:
            model.status = GenerationStatus.FAILED
            model.error_message = str(e)
            model.generation_time = time.time() - start_time
            logger.error(f"Model generation failed: {e}", exc_info=True)
            raise

    def _generate_image(self, request: GenerationRequest, output_dir: Path, task_id: Optional[str] = None) -> Path:
        """Stage 1: Generate base image"""
        image_path = output_dir / "base_image.png"
        return self.image_generator.generate(request, image_path, task_id=task_id)

    def _separate_assets(
        self,
        request: GenerationRequest,
        image_path: Path,
        output_dir: Path,
        segmentation_mode: str,
    ) -> tuple[list[Asset], Dict]:
        """Stage 2: Separate image into assets"""
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        requested_layers = request.custom_layers or config.ASSET_SEPARATION.get("layers", [])

        logger.info(
            "Running asset separation with mode=%s (requested layers: %s)",
            segmentation_mode,
            requested_layers,
        )

        base_assets = self.asset_separator.separate(image_path, assets_dir, requested_layers)

        advanced_assets: List[Asset] = []
        segmentation_used = []

        if segmentation_mode in {"sam", "enhanced", "ultimate", "hybrid"} and self.sam_segmentator:
            try:
                sam_dir = assets_dir / "sam"
                advanced_assets = self.sam_segmentator.segment_image(image_path, sam_dir)
                segmentation_used.append("sam")
            except Exception as exc:
                logger.warning("SAM segmentation failed, falling back to base assets: %s", exc)

        merged_assets = self._merge_assets(base_assets, advanced_assets, segmentation_mode)

        report = {
            "base_layers": len(base_assets),
            "advanced_layers": len(advanced_assets),
            "requested_layers": requested_layers,
            "segmentation_mode": segmentation_mode,
            "methods": segmentation_used or ["rule_based"],
        }

        return merged_assets, report

    def _merge_assets(
        self,
        base_assets: List[Asset],
        advanced_assets: List[Asset],
        segmentation_mode: str,
    ) -> List[Asset]:
        merged: List[Asset] = []
        layer_counts: Dict[str, int] = {}

        for asset in base_assets:
            merged.append(asset)
            layer_counts[asset.layer_type] = layer_counts.get(asset.layer_type, 0) + 1

        for asset in advanced_assets:
            layer_key = asset.layer_type or "segment"
            count = layer_counts.get(layer_key, 0)
            if count:
                asset.layer_type = f"{layer_key}_{count + 1}"
            layer_counts[layer_key] = count + 1
            asset.metadata = {
                **asset.metadata,
                "segmentation_mode": segmentation_mode,
                "source": "sam",
            }
            merged.append(asset)

        return merged

    def _assemble_model(
        self,
        assets: list[Asset],
        output_dir: Path,
        model_name: str
    ) -> Path:
        """Stage 3: Assemble model"""
        models_dir = output_dir / "model"
        return self.model_assembler.assemble(assets, models_dir, model_name)

    def _apply_rigging(self, model_path: Path, assets: list[Asset]) -> Dict:
        """Stage 4: Apply auto-rigging"""
        return self.auto_rigger.rig_model(model_path, assets)

    def _generate_accessories(
        self,
        request: GenerationRequest,
        base_image_path: Path,
        output_dir: Path,
    ) -> List[Asset]:
        accessory_names = request.accessory_list or []
        profile = self._resolve_profile(request.pipeline_profile)

        if not accessory_names and profile.get("generate_accessories"):
            accessory_names = config.DEFAULT_ACCESSORIES

        if not accessory_names:
            return []

        try:
            from PIL import Image

            base_image = Image.open(base_image_path)
            accessories_dir = output_dir / "accessories"
            accessories_dir.mkdir(parents=True, exist_ok=True)

            generated_assets: List[Asset] = []

            for index, name in enumerate(accessory_names):
                spec = self._build_accessory_spec(name)
                if not spec:
                    continue

                output_path = accessories_dir / f"{spec.accessory_type}_{index}.png"
                asset = self.accessory_generator.generate_accessory(
                    base_image,
                    spec,
                    output_path
                )
                asset.metadata.setdefault("requested_name", name)
                generated_assets.append(asset)

            logger.info("Generated %d accessories", len(generated_assets))
            return generated_assets
        except Exception as exc:
            logger.warning("Accessory generation failed: %s", exc)
            return []

    def _build_accessory_spec(self, name: str) -> Optional[Accessory]:
        if not name:
            return None

        normalized = name.strip().lower()

        type_map = {
            "cat_ears": AccessoryType.CAT_EARS,
            "bunny_ears": AccessoryType.BUNNY_EARS,
            "halo": AccessoryType.HALO,
            "crown": AccessoryType.CROWN,
            "hat": AccessoryType.HAT,
            "headband": AccessoryType.HEADBAND,
            "hair_bow": AccessoryType.HAIR_BOW,
            "horns": AccessoryType.HORNS,
            "glasses": AccessoryType.GLASSES,
            "monocle": AccessoryType.MONOCLE,
            "necklace": AccessoryType.NECKLACE,
            "choker": AccessoryType.CHOKER,
            "bow_tie": AccessoryType.BOW_TIE,
            "scarf": AccessoryType.SCARF,
            "sparkles": AccessoryType.SPARKLES,
            "hearts": AccessoryType.HEARTS,
            "stars": AccessoryType.STARS,
            "flowers": AccessoryType.FLOWERS,
            "butterflies": AccessoryType.BUTTERFLIES,
        }

        accessory_type = type_map.get(normalized)
        if not accessory_type:
            logger.warning("Unknown accessory requested: %s", name)
            return None

        colour_map = {
            AccessoryType.CAT_EARS: (255, 200, 220, 255),
            AccessoryType.BUNNY_EARS: (255, 240, 240, 255),
            AccessoryType.HALO: (255, 255, 150, 200),
            AccessoryType.CROWN: (255, 215, 0, 255),
            AccessoryType.HAT: (120, 120, 120, 255),
            AccessoryType.HEADBAND: (200, 120, 200, 255),
            AccessoryType.HAIR_BOW: (255, 105, 180, 255),
            AccessoryType.HORNS: (180, 80, 80, 255),
            AccessoryType.GLASSES: (20, 20, 20, 220),
            AccessoryType.MONOCLE: (230, 200, 120, 220),
            AccessoryType.NECKLACE: (220, 220, 220, 255),
            AccessoryType.CHOKER: (80, 80, 80, 255),
            AccessoryType.BOW_TIE: (180, 0, 60, 255),
            AccessoryType.SCARF: (180, 30, 30, 255),
            AccessoryType.SPARKLES: (255, 255, 255, 255),
            AccessoryType.HEARTS: (255, 80, 120, 200),
            AccessoryType.STARS: (255, 255, 180, 220),
            AccessoryType.FLOWERS: (255, 192, 203, 220),
            AccessoryType.BUTTERFLIES: (180, 200, 255, 200),
        }

        color = colour_map.get(accessory_type, (255, 255, 255, 200))
        return Accessory(accessory_type, name=normalized, color=color)

    def _generate_expressions(
        self,
        request: GenerationRequest,
        base_image_path: Path,
        output_dir: Path,
        controlnet_enabled: bool,
    ) -> Optional[Dict]:
        expression_names = request.expression_list or []
        profile = self._resolve_profile(request.pipeline_profile)

        if request.generate_expressions is False:
            return None

        if not expression_names and profile.get("generate_expressions", True):
            expression_names = config.DEFAULT_EXPRESSIONS

        if not expression_names:
            return None

        try:
            from PIL import Image

            expression_dir = output_dir / "expressions"
            expression_set = self.expression_generator.generate_expression_set(
                base_image_path,
                expression_dir,
                expression_names
            )

            base_image = Image.open(base_image_path)
            previews: Dict[str, str] = {}
            total_assets = 0

            for expr_name in expression_set.list_expressions():
                assets = expression_set.get_expression(expr_name) or []
                total_assets += len(assets)
                preview_path = expression_dir / expr_name / f"preview_{expr_name}.png"
                try:
                    self.expression_generator.create_expression_preview(
                        assets,
                        base_image,
                        preview_path
                    )
                except Exception as exc:
                    logger.warning("Failed to build preview for %s: %s", expr_name, exc)
                previews[expr_name] = str(preview_path)

            controlnet_variations: Dict[str, List[str]] = {}
            if controlnet_enabled and self.controlnet_expression_generator.use_controlnet:
                try:
                    if self.controlnet_expression_generator.pipeline is None:
                        self.controlnet_expression_generator.load_controlnet_model()

                    variation_count = request.expression_variations or 0
                    if variation_count:
                        controlnet_dir = expression_dir / "controlnet"
                        for expr_name in expression_set.list_expressions():
                            target_dir = controlnet_dir / expr_name
                            variations = self.controlnet_expression_generator.generate_expression_variations(
                                base_image_path,
                                expr_name,
                                target_dir,
                                num_variations=variation_count
                            )
                            controlnet_variations[expr_name] = [str(path) for path in variations]
                except Exception as exc:
                    logger.warning("ControlNet expression generation failed: %s", exc)

            logger.info(
                "Generated %d expressions (%d assets)",
                len(expression_set.list_expressions()),
                total_assets,
            )

            return {
                "count": len(expression_set.list_expressions()),
                "names": expression_set.list_expressions(),
                "total_assets": total_assets,
                "previews": previews,
                "controlnet_variations": controlnet_variations,
            }
        except Exception as exc:
            logger.warning("Expression generation failed: %s", exc)
            return None

    def _enhance_physics(self, model_dir: Path, assets: List[Asset]) -> Optional[Path]:
        if not self.auto_physics_generator or not assets:
            return None

        model_name = model_dir.name
        output_path = model_dir / f"{model_name}.physics3.json"

        try:
            return self.auto_physics_generator.generate_physics_json(
                assets,
                model_name,
                output_path
            )
        except Exception as exc:
            logger.warning("Auto physics generation failed: %s", exc)
            return None

    def _write_summary(self, model: VTuberModel, output_dir: Path) -> Optional[Path]:
        summary = {
            "model_id": model.id,
            "name": model.name,
            "created_at": model.created_at.isoformat(),
            "status": model.status.value,
            "generation_time": model.generation_time,
            "pipeline": {
                "profile": model.metadata.get("pipeline_profile"),
                "label": model.metadata.get("pipeline_profile_label"),
                "segmentation": model.metadata.get("segmentation_mode"),
                "controlnet": model.metadata.get("controlnet_enabled"),
                "expressions": model.metadata.get("expressions_enabled"),
                "accessories": model.metadata.get("accessories_enabled"),
                "auto_physics": model.metadata.get("auto_physics_enabled"),
            },
            "assets": [
                {
                    "layer_type": asset.layer_type,
                    "file_path": str(asset.file_path),
                    "metadata": asset.metadata,
                }
                for asset in model.assets
            ],
            "metadata": model.metadata,
        }

        try:
            summary_path = output_dir / "summary.json"
            save_json(summary, summary_path)
            return summary_path
        except Exception as exc:
            logger.warning("Failed to write summary: %s", exc)
            return None

    def _resolve_profile(self, profile_name: Optional[str]) -> Dict:
        if not profile_name:
            profile_name = "standard"
        return self.pipeline_profiles.get(profile_name, self.pipeline_profiles.get("standard", {}))

    def _resolve_segmentation_mode(self, request: GenerationRequest, profile: Dict) -> str:
        mode = (request.segmentation_mode or "auto").lower()
        if mode in {"auto", "default"}:
            return profile.get("segmentation", "basic")
        return mode

    def _should_use_controlnet(self, request: GenerationRequest, profile: Dict) -> bool:
        if request.use_controlnet is True:
            return True
        if request.use_controlnet is False:
            return False
        return bool(profile.get("controlnet"))

    def _should_generate_expressions(self, request: GenerationRequest, profile: Dict) -> bool:
        if request.generate_expressions is False:
            return False
        if request.generate_expressions is True:
            return True
        return bool(profile.get("generate_expressions", True))

    def _should_generate_accessories(self, request: GenerationRequest, profile: Dict) -> bool:
        if request.generate_accessories is True or (request.accessory_list and len(request.accessory_list) > 0):
            return True
        if request.generate_accessories is False:
            return False
        return bool(profile.get("generate_accessories"))

    def _should_enable_physics(self, request: GenerationRequest, profile: Dict) -> bool:
        if request.enable_auto_physics is True:
            return True
        if request.enable_auto_physics is False:
            return False
        return bool(profile.get("auto_physics", True))

    def _generate_preview(self, model: VTuberModel, output_dir: Path):
        """Generate preview image for the model"""
        preview_path = output_dir / "preview.png"
        try:
            self.model_assembler.create_preview_image(
                model.assets,
                preview_path
            )
            model.metadata["preview_path"] = str(preview_path)
        except Exception as e:
            logger.warning(f"Failed to generate preview: {e}")

    def _update_progress(self, message: str, progress: float):
        """Update progress via callback"""
        logger.info(f"Progress: {message} ({progress * 100:.0f}%)")
        if self.progress_callback:
            try:
                self.progress_callback(message, progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up assembly line")
        if hasattr(self.image_generator, 'unload_model'):
            self.image_generator.unload_model()


class BatchProcessor:
    """
    Process multiple model generation requests in batch
    """

    def __init__(self, assembly_line: AssemblyLine):
        self.assembly_line = assembly_line

    def process_batch(self, requests: list[GenerationRequest]) -> list[VTuberModel]:
        """
        Process multiple generation requests

        Args:
            requests: List of generation requests

        Returns:
            List of generated models
        """
        models = []

        for i, request in enumerate(requests):
            logger.info(f"Processing batch item {i + 1}/{len(requests)}")

            try:
                model = self.assembly_line.generate_model(request)
                models.append(model)
            except Exception as e:
                logger.error(f"Batch item {i + 1} failed: {e}")
                # Continue with next item

        logger.info(f"Batch processing complete: {len(models)}/{len(requests)} successful")
        return models
