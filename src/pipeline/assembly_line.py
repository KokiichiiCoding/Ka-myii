"""
Assembly line pipeline for automated VTuber model generation
"""
import logging
import time
from pathlib import Path
from typing import Optional, Callable, Dict
import uuid

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
        self.use_dummy_generators = use_dummy_generators
        self._generator_signature = None

        logger.info("AssemblyLine initialized")

    def _ensure_generator(self, request: GenerationRequest):
        """Rebuild the image generator if the request selects a different
        checkpoint / sampler / LoRA stack (real mode only)."""
        if self.use_dummy_generators:
            return
        signature = (
            request.model_id,
            request.sampler,
            request.clip_skip,
            tuple((lora.get("path") or lora.get("id"), lora.get("weight", 1.0)) for lora in (request.loras or [])),
        )
        if signature == self._generator_signature and self.image_generator.pipeline is not None:
            return

        from src.utils.model_scanner import resolve_checkpoint

        resolved = resolve_checkpoint(request.model_id) if request.model_id else {}
        logger.info("Configuring generator: %s", resolved.get("model_name") or "<default>")
        self.image_generator = ImageGenerator(
            model_name=resolved.get("model_name"),
            pipeline=resolved.get("pipeline"),
            custom_model_path=resolved.get("custom_model_path"),
            sampler=request.sampler,
            clip_skip=request.clip_skip,
            loras=request.loras,
        )
        self._generator_signature = signature

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

        # Create output directory for this model
        model_output_dir = self.output_base_dir / model.id
        model_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting generation for model: {model.id} (task: {task_id})")

        try:
            # Stage 1: Image Generation
            self._update_progress("Generating base image...", 0.1)
            model.status = GenerationStatus.IMAGE_GENERATION
            self._ensure_generator(request)
            base_image_path = self._generate_image(request, model_output_dir, task_id)
            model.base_image_path = base_image_path
            logger.info(f"Base image generated: {base_image_path}")

            # Stage 2: Asset Separation
            self._update_progress("Separating assets into layers...", 0.4)
            model.status = GenerationStatus.ASSET_SEPARATION
            assets = self._separate_assets(base_image_path, model_output_dir, request.custom_layers)
            model.assets = assets
            logger.info(f"Assets separated: {len(assets)} layers")

            # Stage 3: Model Assembly
            self._update_progress("Assembling Live2D model...", 0.7)
            model.status = GenerationStatus.MODEL_ASSEMBLY
            final_model_path = self._assemble_model(assets, model_output_dir, model.name)
            model.final_model_path = final_model_path
            logger.info(f"Model assembled: {final_model_path}")

            # Surface the rig-ready artifacts (PSD/ORA/flat) on the model.
            manifest_path = final_model_path / "kamyii_manifest.json"
            if manifest_path.exists():
                try:
                    import json as _json

                    manifest = _json.loads(manifest_path.read_text())
                    model.metadata["artifacts"] = manifest.get("artifacts", {})
                    model.metadata["layers"] = manifest.get("layers", [])
                    model.metadata["parameters"] = manifest.get("parameters", [])
                except Exception as exc:  # pragma: no cover
                    logger.warning("Could not read model manifest: %s", exc)

            # Stage 4: Auto-Rigging (optional)
            if request.include_rigging:
                self._update_progress("Applying auto-rigging...", 0.9)
                model.status = GenerationStatus.RIGGING
                rigging_data = self._apply_rigging(final_model_path, assets)
                model.metadata["rigging"] = rigging_data
                logger.info("Rigging applied")

            # Complete
            model.status = GenerationStatus.COMPLETED
            model.generation_time = time.time() - start_time
            self._update_progress("Model generation complete!", 1.0)

            logger.info(f"Model generation completed in {model.generation_time:.2f}s")

            # Generate preview
            self._generate_preview(model, model_output_dir)

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
        image_path: Path,
        output_dir: Path,
        custom_layers: Optional[list] = None
    ) -> list[Asset]:
        """Stage 2: Separate image into assets"""
        assets_dir = output_dir / "assets"
        return self.asset_separator.separate(image_path, assets_dir, custom_layers)

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
