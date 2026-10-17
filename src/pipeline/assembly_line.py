"""
Assembly line pipeline for automated VTuber model generation
"""
import logging
import shutil
import time
from dataclasses import replace
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
from src.utils.helpers import save_json

logger = logging.getLogger(__name__)


class AssemblyLine:
    """
    Orchestrates the complete VTuber model generation pipeline

    Pipeline stages:
    1. Image Generation - Create base character image
    2. Asset Separation - Split into layers
    3. Model Assembly - PSD/ORA + Live2D Cubism scaffold
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

        # Resolve randomness once so the saved brief can reproduce this character.
        if request.seed is None and not request.reference_image:
            from secrets import randbits
            request = replace(request, seed=randbits(31))

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
            self._enter_stage(task_id, "image", "Generating base image...", 0.1)
            model.status = GenerationStatus.IMAGE_GENERATION
            if request.reference_image:  # the user's own artwork, validated by the upload route
                base_image_path = model_output_dir / "base_image.png"
                shutil.move(request.reference_image, base_image_path)
                model.request = request = replace(request, reference_image="upload")
                model.metadata["source"] = "upload"
            else:
                self._ensure_generator(request)
                base_image_path = self._generate_image(request, model_output_dir, task_id)
            model.base_image_path = base_image_path
            logger.info(f"Base image generated: {base_image_path}")

            # Stage 2: Asset Separation
            self._enter_stage(task_id, "separate", "Separating assets into layers...", 0.4)
            model.status = GenerationStatus.ASSET_SEPARATION
            if not self.use_dummy_generators:
                # See-through runs in its own process and needs the GPU memory the generator holds.
                self.image_generator.unload_model()
            assets = self._separate_assets(base_image_path, model_output_dir, request.custom_layers, task_id)
            model.assets = assets
            logger.info(f"Assets separated: {len(assets)} layers")

            # Stage 3: Model Assembly
            self._enter_stage(task_id, "assemble", "Assembling Live2D model...", 0.7)
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

            # Complete
            model.status = GenerationStatus.COMPLETED
            model.generation_time = time.time() - start_time
            self._update_progress("Model generation complete!", 1.0)

            logger.info(f"Model generation completed in {model.generation_time:.2f}s")

            # Generate preview
            self._generate_preview(model, model_output_dir)

            return model

        except InterruptedError:
            # Cancelled by the user: nothing in the half-built folder is wanted.
            shutil.rmtree(model_output_dir, ignore_errors=True)
            logger.info("Generation %s cancelled; removed %s", task_id, model_output_dir)
            raise
        except Exception as e:
            model.status = GenerationStatus.FAILED
            model.error_message = str(e)
            model.generation_time = time.time() - start_time
            logger.error(f"Model generation failed: {e}", exc_info=True)
            if (model_output_dir / "base_image.png").exists():
                # Keep the artwork (e.g. See-through ran out of memory): the Gallery lists it as
                # failed, and it can be re-uploaded with "Use my artwork" or deleted.
                save_json(model.to_dict(), model_output_dir / "metadata.json")
            else:
                shutil.rmtree(model_output_dir, ignore_errors=True)
            raise

    def _generate_image(self, request: GenerationRequest, output_dir: Path, task_id: Optional[str] = None) -> Path:
        """Stage 1: Generate base image"""
        image_path = output_dir / "base_image.png"
        return self.image_generator.generate(request, image_path, task_id=task_id)

    def _separate_assets(
        self,
        image_path: Path,
        output_dir: Path,
        custom_layers: Optional[list] = None,
        task_id: Optional[str] = None,
    ) -> list[Asset]:
        """Stage 2: Separate image into assets"""
        assets_dir = output_dir / "assets"
        progress = None
        if task_id:
            from src.utils.progress_tracker import get_progress_manager

            tracker = get_progress_manager().create_tracker(task_id, 100)

            def progress(frac, label):
                if tracker.is_cancelled():  # See-through's subprocess is killed on the way out
                    raise InterruptedError("Generation cancelled by user")
                tracker.update(int(frac * 100), f"Decomposing layers — {label}")
        return self.asset_separator.separate(image_path, assets_dir, custom_layers, progress=progress)

    def _assemble_model(
        self,
        assets: list[Asset],
        output_dir: Path,
        model_name: str
    ) -> Path:
        """Stage 3: Assemble model"""
        models_dir = output_dir / "model"
        return self.model_assembler.assemble(assets, models_dir, model_name)

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

    def _enter_stage(self, task_id: str, stage: str, message: str, progress: float):
        from src.utils.progress_tracker import get_progress_manager

        manager = get_progress_manager()
        if manager.is_cancelled(task_id):
            raise InterruptedError("Generation cancelled by user")
        manager.set_stage(task_id, stage)
        self._update_progress(message, progress)

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
