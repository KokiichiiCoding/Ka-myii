"""
Image generation module.

Generates high-quality anime / VTuber character art using modern SDXL-class
diffusion models (Illustrious-XL, NoobAI-XL, Animagine, Pony, Holodayo, …).

Design goals:
* **Never crash the app on import.** ``torch`` and ``diffusers`` are heavy,
  optional dependencies — they are imported lazily so the web UI and the whole
  non-GPU pipeline (decomposition, PSD, Live2D packaging) work without them.
* **Modern diffusers API.** Uses ``callback_on_step_end`` (the current standard)
  with a graceful fallback to the legacy ``callback`` argument.
* **WebUI-style controls.** Sampler selection, LoRA stacking, clip-skip and
  custom VAE are all supported.
"""
from __future__ import annotations

import importlib
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import config
from src.models.vtuber_model import GenerationRequest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional heavy dependencies (loaded lazily, never at app import time)
# ---------------------------------------------------------------------------
def _try_import_torch():
    try:
        import torch  # noqa: WPS433 (runtime import is intentional)

        return torch
    except Exception:  # pragma: no cover - torch is optional
        return None


def _diffusers_available() -> bool:
    try:
        importlib.import_module("diffusers")
        return True
    except Exception:  # pragma: no cover - diffusers is optional
        return False


# Progress tracking is light-weight and safe to import eagerly.
try:
    from src.utils.progress_tracker import get_progress_manager

    PROGRESS_AVAILABLE = True
except Exception:  # pragma: no cover
    PROGRESS_AVAILABLE = False


class ImageGenerator:
    """Generates character art with a Stable Diffusion / SDXL pipeline."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        pipeline: Optional[str] = None,
        custom_model_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        original_config_file: Optional[str] = None,
        sampler: Optional[str] = None,
        loras: Optional[List[Dict]] = None,
        clip_skip: Optional[int] = None,
    ):
        settings = getattr(config, "IMAGE_GENERATION", {})

        self.model_name = model_name or settings.get("model_name")
        self.pipeline_type = (pipeline or settings.get("pipeline", "auto") or "auto").lower()
        self.sampler = sampler or settings.get("default_sampler", "DPM++ 2M Karras")
        self.clip_skip = clip_skip if clip_skip is not None else settings.get("clip_skip", 2)
        self.loras = loras or []

        custom_path = custom_model_path or settings.get("custom_model_path")
        self.custom_model_path = Path(custom_path).expanduser() if custom_path else None
        vae_override = vae_path or settings.get("vae_path")
        self.vae_path = Path(vae_override).expanduser() if vae_override else None
        oc = original_config_file or settings.get("original_config_file")
        self.original_config_file = Path(oc).expanduser() if oc else None

        torch = _try_import_torch()
        self._torch = torch
        if device:
            self.device = device
        elif torch is not None and torch.cuda.is_available():
            self.device = "cuda"
        elif torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.pipeline = None

        logger.info(
            "ImageGenerator ready (model=%s, pipeline=%s, sampler=%s, device=%s, source=%s)",
            self.model_name,
            self.pipeline_type,
            self.sampler,
            self.device,
            self.custom_model_path if self.custom_model_path else "<huggingface>",
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self):
        """Load (and download, if necessary) the diffusion pipeline."""
        if not _diffusers_available():
            raise ImportError(
                "The 'diffusers' library is not installed. Install the AI extras with:\n"
                "    pip install torch diffusers transformers accelerate safetensors\n"
                "or run Ka-myii in demo mode (python app.py --dummy)."
            )

        torch = self._torch or _try_import_torch()
        if torch is None:
            raise ImportError("PyTorch is required for real generation. Install 'torch'.")
        self._torch = torch

        pipeline_cls = self._determine_pipeline_class()
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        if self.custom_model_path and not self.custom_model_path.exists():
            raise FileNotFoundError(f"Custom model not found at {self.custom_model_path}")
        if self.vae_path and not self.vae_path.exists():
            raise FileNotFoundError(f"Custom VAE not found at {self.vae_path}")

        logger.info(
            "Loading %s from %s",
            pipeline_cls.__name__,
            self.custom_model_path if self.custom_model_path else self.model_name,
        )

        load_kwargs: Dict[str, object] = {"torch_dtype": dtype}
        # Disable the safety checker for SD1.5 pipelines (anime false-positives).
        if pipeline_cls.__name__ == "StableDiffusionPipeline":
            load_kwargs["safety_checker"] = None
            load_kwargs["requires_safety_checker"] = False

        if self.custom_model_path:
            if self.original_config_file:
                load_kwargs["original_config_file"] = str(self.original_config_file)
            self.pipeline = pipeline_cls.from_single_file(str(self.custom_model_path), **load_kwargs)
        else:
            self.pipeline = pipeline_cls.from_pretrained(self.model_name, **load_kwargs)

        self._apply_sampler()
        self._apply_vae(dtype)

        # Memory / placement strategy.
        if config.IMAGE_GENERATION.get("enable_cpu_offload") and self.device == "cuda":
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline = self.pipeline.to(self.device)

        if self.device == "cuda":
            try:
                self.pipeline.enable_attention_slicing()
            except Exception:  # pragma: no cover
                pass
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception as exc:  # pragma: no cover
                logger.debug("xformers unavailable: %s", exc)

        self._apply_loras()
        logger.info("Model loaded successfully")

    def _apply_sampler(self):
        """Swap in the scheduler that matches the selected sampler."""
        sampler_def = config.SAMPLERS.get(self.sampler)
        if not sampler_def or not hasattr(self.pipeline, "scheduler"):
            return
        try:
            schedulers = importlib.import_module("diffusers")
            scheduler_cls = getattr(schedulers, sampler_def["class"], None)
            if scheduler_cls is None:
                return
            self.pipeline.scheduler = scheduler_cls.from_config(
                self.pipeline.scheduler.config, **sampler_def.get("config", {})
            )
            logger.info("Sampler set to %s (%s)", self.sampler, sampler_def["class"])
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not set sampler '%s': %s", self.sampler, exc)

    def _apply_vae(self, dtype):
        if not self.vae_path:
            return
        try:
            from diffusers import AutoencoderKL

            logger.info("Loading custom VAE from %s", self.vae_path)
            if self.vae_path.is_file():
                vae = AutoencoderKL.from_single_file(str(self.vae_path), torch_dtype=dtype)
            else:
                vae = AutoencoderKL.from_pretrained(str(self.vae_path), torch_dtype=dtype)
            self.pipeline.vae = vae.to(self.device)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load custom VAE: %s", exc)

    def _apply_loras(self):
        """Stack any requested LoRA adapters with their weights."""
        if not self.loras:
            return
        adapter_names, adapter_weights = [], []
        for i, lora in enumerate(self.loras):
            path = lora.get("path") or lora.get("id")
            if not path or not Path(path).exists():
                logger.warning("Skipping missing LoRA: %s", path)
                continue
            name = f"lora_{i}"
            try:
                self.pipeline.load_lora_weights(
                    str(Path(path).parent), weight_name=Path(path).name, adapter_name=name
                )
                adapter_names.append(name)
                adapter_weights.append(float(lora.get("weight", 1.0)))
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to load LoRA %s: %s", path, exc)
        if adapter_names:
            try:
                self.pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
                logger.info("Activated %d LoRA adapter(s)", len(adapter_names))
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to activate LoRAs: %s", exc)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        request: GenerationRequest,
        output_path: Path,
        task_id: Optional[str] = None,
    ) -> Path:
        """Generate an image for ``request`` and save it to ``output_path``."""
        if self.pipeline is None:
            self.load_model()

        torch = self._torch
        logger.info("Generating: %s", request.prompt[:60])

        positive, negative = self._build_prompts(request)

        tracker = None
        if PROGRESS_AVAILABLE:
            try:
                manager = get_progress_manager()
                if manager:
                    task_id = task_id or f"gen_{uuid.uuid4().hex[:8]}"
                    tracker = manager.create_tracker(task_id, request.steps)
            except Exception as exc:  # pragma: no cover
                logger.debug("Progress tracking unavailable: %s", exc)

        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(request.seed))

        call_kwargs = dict(
            prompt=positive,
            negative_prompt=negative,
            width=request.width,
            height=request.height,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )
        if self.clip_skip and self.clip_skip > 1:
            # diffusers counts hidden layers to skip from the end.
            call_kwargs["clip_skip"] = self.clip_skip

        self._attach_progress_callback(call_kwargs, tracker, request.steps)

        try:
            result = self.pipeline(**call_kwargs)
            image = result.images[0]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            if tracker:
                tracker.complete("Image generated successfully")
            logger.info("Saved generated image -> %s", output_path)
            return output_path
        except InterruptedError:
            if tracker:
                tracker.complete("Generation cancelled", status="cancelled")
            raise
        except Exception as exc:
            if tracker:
                tracker.complete(f"Generation failed: {exc}", status="failed")
            logger.error("Generation failed: %s", exc)
            raise

    def _attach_progress_callback(self, call_kwargs: dict, tracker, total_steps: int):
        """Wire up live progress using the modern callback, with a fallback."""
        if tracker is None:
            return

        def _on_step_end(pipe, step, timestep, callback_kwargs):
            tracker.update(step + 1, f"Diffusion step {step + 1}/{total_steps}")
            if tracker.is_cancelled():
                raise InterruptedError("Generation cancelled by user")
            return callback_kwargs

        try:
            import inspect

            sig = inspect.signature(self.pipeline.__call__)
            if "callback_on_step_end" in sig.parameters:
                call_kwargs["callback_on_step_end"] = _on_step_end
            elif "callback" in sig.parameters:  # legacy diffusers
                def _legacy(step, timestep, latents):
                    tracker.update(step + 1, f"Diffusion step {step + 1}/{total_steps}")
                    if tracker.is_cancelled():
                        raise InterruptedError("Generation cancelled by user")

                call_kwargs["callback"] = _legacy
                call_kwargs["callback_steps"] = 1
        except Exception as exc:  # pragma: no cover
            logger.debug("Could not attach progress callback: %s", exc)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_prompts(self, request: GenerationRequest):
        """Return (positive, negative) prompts enriched with style + quality tags."""
        style = (request.style or "anime").lower()
        preset = config.STYLE_PRESETS.get(style, config.STYLE_PRESETS["anime"])

        positive_parts = [request.prompt.strip(), preset["positive"]]
        positive = ", ".join(p for p in positive_parts if p)

        negative_parts = [request.negative_prompt.strip(), preset.get("negative", ""), config.DEFAULT_NEGATIVE_PROMPT]
        negative = ", ".join(p for p in negative_parts if p)
        return positive, negative

    def _determine_pipeline_class(self):
        diffusers = importlib.import_module("diffusers")
        choice = self.pipeline_type
        if choice == "auto":
            source = self.custom_model_path.name if self.custom_model_path else (self.model_name or "")
            choice = "sdxl" if "xl" in source.lower() else "sd15"

        if choice == "sdxl":
            cls = getattr(diffusers, "StableDiffusionXLPipeline", None)
            if cls is None:
                raise ImportError("Install diffusers>=0.19 for SDXL support.")
            return cls
        cls = getattr(diffusers, "StableDiffusionPipeline", None)
        if cls is None:
            raise ImportError("diffusers StableDiffusionPipeline unavailable.")
        return cls

    def unload_model(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch = self._torch or _try_import_torch()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded")


# ---------------------------------------------------------------------------
# Demo generator (no GPU / no weights required)
# ---------------------------------------------------------------------------
class DummyImageGenerator(ImageGenerator):
    """Renders a clean, *layer-friendly* placeholder character with PIL.

    Unlike a flat placeholder, this draws a stylised character whose regions
    (hair, body, face, eyes, mouth) use distinct colours and positions so the
    downstream decomposition / PSD / Live2D stages produce a meaningful,
    demonstrable result without any model weights.
    """

    def __init__(self, *args, **kwargs):  # noqa: D401 - simple override
        # Avoid touching torch/diffusers at all.
        self.model_name = "demo-procedural"
        self.pipeline_type = "demo"
        self.sampler = "n/a"
        self.device = "cpu"
        self.pipeline = "demo"
        self.loras = []
        logger.info("Using DummyImageGenerator (procedural demo character)")

    def load_model(self):  # noqa: D401
        logger.info("DummyImageGenerator: no model to load")

    def generate(self, request: GenerationRequest, output_path: Path, task_id: Optional[str] = None) -> Path:
        import time

        from src.core.demo_character import render

        tracker = None
        if PROGRESS_AVAILABLE:
            try:
                manager = get_progress_manager()
                if manager:
                    task_id = task_id or f"gen_{uuid.uuid4().hex[:8]}"
                    tracker = manager.create_tracker(task_id, request.steps or 10)
            except Exception:  # pragma: no cover
                tracker = None

        # Simulate denoising steps so the UI progress bar behaves like the real thing.
        total = max(1, request.steps or 10)
        for step in range(total):
            if tracker:
                tracker.update(step + 1, f"Rendering demo character {step + 1}/{total}")
                if tracker.is_cancelled():
                    tracker.complete("Generation cancelled", status="cancelled")
                    raise InterruptedError("Generation cancelled by user")
            time.sleep(0.02)  # keep the progress bar visible for the demo

        image, region_map = render(request.prompt, request.width, request.height, seed=request.seed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(output_path)
        # Region-map sidecar enables pixel-perfect demo decomposition downstream.
        region_map.save(output_path.with_suffix(".regions.png"))
        if tracker:
            tracker.complete("Demo character rendered")
        logger.info("DummyImageGenerator saved -> %s", output_path)
        return output_path
