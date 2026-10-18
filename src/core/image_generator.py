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


def chunk_token_ids(ids: List[int], comma: int, size: int = 75, lookback: int = 20) -> List[List[int]]:
    """Split prompt tokens into CLIP-sized chunks, preferring to cut just after a comma
    within the last ``lookback`` tokens so a tag is not split across chunks (A1111's rule)."""
    chunks = []
    while len(ids) > size:
        commas = [i for i in range(size - lookback, size) if ids[i] == comma]
        cut = commas[-1] + 1 if commas else size
        chunks.append(ids[:cut])
        ids = ids[cut:]
    return chunks + [ids]


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

        # Memory / placement strategy. SDXL in fp16 is ~6.8 GB, so cards under 12 GB
        # (e.g. 8 GB laptops) need model offload to leave room for activations.
        low_vram = self.device == "cuda" and torch.cuda.get_device_properties(0).total_memory < 12 * 1024**3
        if self.device == "cuda" and (config.IMAGE_GENERATION.get("enable_cpu_offload") or low_vram):
            logger.info("Enabling model CPU offload (low VRAM)")
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline = self.pipeline.to(self.device)

        if self.device == "cuda":
            # No attention slicing: torch 2 SDPA is already memory-efficient and slicing only slows it down.
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
        tracker = None
        if PROGRESS_AVAILABLE:
            try:
                manager = get_progress_manager()
                if manager:
                    task_id = task_id or f"gen_{uuid.uuid4().hex[:8]}"
                    tracker = manager.create_tracker(task_id, request.steps)
            except Exception as exc:  # pragma: no cover
                logger.debug("Progress tracking unavailable: %s", exc)

        if self.pipeline is None:
            if tracker:
                tracker.update(0, self._loading_message())
            self.load_model()

        torch = self._torch
        logger.info("Generating: %s", request.prompt[:60])

        positive, negative = self._build_prompts(request)

        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(request.seed))

        call_kwargs = dict(
            width=request.width,
            height=request.height,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )
        if getattr(self.pipeline, "tokenizer_2", None) is not None:
            call_kwargs.update(self._sdxl_prompt_embeds(positive, negative))
        else:
            call_kwargs.update(prompt=positive, negative_prompt=negative)
            if self.clip_skip and self.clip_skip > 1:
                # UI uses the A1111 convention (2 = penultimate); diffusers SD1.5 counts layers skipped past it.
                # shortcut: SD1.5 prompts are still cut at 77 tokens; chunk them like SDXL if SD1.5 matters.
                call_kwargs["clip_skip"] = self.clip_skip - 1

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

    def _loading_message(self) -> str:
        if not self.custom_model_path:
            try:
                from huggingface_hub import try_to_load_from_cache

                if not isinstance(try_to_load_from_cache(self.model_name, "unet/config.json"), str):
                    return f"Downloading {self.model_name} from Hugging Face (first use only, ~7 GB)…"
            except Exception:  # pragma: no cover - only changes the message
                pass
        return f"Loading checkpoint {self.model_name} (about a minute)…"

    # ------------------------------------------------------------------
    # Inpainting (region edits, outpainting, expression art)
    # ------------------------------------------------------------------
    def inpaint(self, image, mask, prompt: str, negative: str = "", strength: float = 0.9,
                steps: Optional[int] = None, guidance: Optional[float] = None, seed: Optional[int] = None,
                task_id: Optional[str] = None, label: str = "Repainting the selection", detail: int = 1024,
                feather: int = 6):
        """Repaint the white part of ``mask`` (PIL L) in ``image`` and return the full image.

        Works on a crop around the mask at SDXL's native ~``detail`` px (a face gets far more
        pixels than in the whole picture), with the loaded checkpoint shared through
        ``from_pipe``. Everything outside the dilated mask is returned unchanged; the seam is
        a ``feather``-px blend.
        """
        from PIL import Image, ImageFilter

        if self.pipeline is None:
            self.load_model()
        import numpy as np

        torch = self._torch
        image = image.convert("RGB")
        hard = mask.convert("L").point(lambda v: 255 if v > 127 else 0)
        box = hard.getbbox()
        if box is None:
            return image
        # Context around the selection, at least ~40 % of the picture so the model sees the style.
        W, H = image.size
        x0, y0, x1, y1 = box
        side = max(x1 - x0, y1 - y0)
        pad = max(48, int(side * 0.35))
        cw, ch = max(x1 - x0 + 2 * pad, int(0.4 * W)), max(y1 - y0 + 2 * pad, int(0.4 * H))
        cw, ch = min(cw, W), min(ch, H)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        left = min(max(0, cx - cw // 2), W - cw)
        top = min(max(0, cy - ch // 2), H - ch)
        crop_box = (left, top, left + cw, top + ch)
        scale = detail / max(cw, ch)
        tw, th = max(64, int(cw * scale) // 8 * 8), max(64, int(ch * scale) // 8 * 8)

        tracker = None
        if PROGRESS_AVAILABLE and task_id:
            tracker = get_progress_manager().create_tracker(task_id, max(1, int((steps or 28) * strength)))
            tracker.update(0, label + "…")
        from diffusers import AutoPipelineForInpainting

        pipe = AutoPipelineForInpainting.from_pipe(self.pipeline)
        positive, negative = self._build_prompts(GenerationRequest(prompt=prompt, negative_prompt=negative,
                                                                   style="anime", framing="none"))
        call = dict(
            image=image.crop(crop_box).resize((tw, th), Image.Resampling.LANCZOS),
            mask_image=hard.crop(crop_box).resize((tw, th), Image.Resampling.NEAREST),
            width=tw, height=th, strength=float(strength),
            num_inference_steps=int(steps or config.IMAGE_GENERATION.get("default_steps", 28)),
            guidance_scale=float(guidance if guidance is not None else config.IMAGE_GENERATION.get("default_guidance_scale", 5)),
            generator=torch.Generator(device=self.device).manual_seed(int(seed)) if seed is not None else None,
        )
        call.update(self._sdxl_prompt_embeds(positive, negative) if getattr(pipe, "tokenizer_2", None) is not None
                    else dict(prompt=positive, negative_prompt=negative))
        if tracker:
            def _step(_pipe, step, _t, kwargs):
                tracker.update(step + 1, f"{label} — step {step + 1}")
                if tracker.is_cancelled():
                    raise InterruptedError("Generation cancelled by user")
                return kwargs
            call["callback_on_step_end"] = _step
        out = pipe(**call).images[0].resize((cw, ch), Image.Resampling.LANCZOS)

        soft = hard.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(feather)) if feather else hard
        region = Image.composite(out, image.crop(crop_box), soft.crop(crop_box))
        result = image.copy()
        result.paste(region, crop_box[:2])
        if tracker:
            tracker.complete(label + " — done")
        logger.info("Inpainted %s px region (crop %s at %dx%d, strength %.2f)",
                    int((np.asarray(hard) > 0).sum()), crop_box, tw, th, strength)
        return result

    def _sdxl_prompt_embeds(self, positive: str, negative: str) -> dict:
        """SDXL conditioning without CLIP's 77-token cut-off, the way A1111/ComfyUI do it.

        Prompts are split into 75-token chunks; each one is wrapped in BOS/EOS, encoded by both
        text encoders and the results are concatenated. Encoding ourselves also gives the UI's
        clip skip its A1111 meaning (2 = penultimate layer, SDXL's native one) for BOTH prompts;
        diffusers' own ``clip_skip`` counts past the penultimate and ignores the negative prompt.
        """
        torch, pipe = self._torch, self.pipeline
        device = pipe._execution_device
        layer = -max(1, int(self.clip_skip or 2))
        tok = pipe.tokenizer  # both SDXL tokenizers share CLIP's BPE vocabulary; only padding differs
        comma = tok.convert_tokens_to_ids(",</w>")
        chunks = [
            chunk_token_ids(tok(text, add_special_tokens=False, truncation=False, verbose=False).input_ids, comma)
            for text in (positive, negative)
        ]
        count = max(len(c) for c in chunks)
        encoded = []
        with torch.no_grad():
            for ids in chunks:
                ids = ids + [[]] * (count - len(ids))  # equal lengths for classifier-free guidance
                hidden, pooled = [], None
                for tokenizer, encoder in ((pipe.tokenizer, pipe.text_encoder), (pipe.tokenizer_2, pipe.text_encoder_2)):
                    parts = []
                    for chunk in ids:
                        row = [tokenizer.bos_token_id, *chunk, tokenizer.eos_token_id]
                        row += [tokenizer.pad_token_id] * (tokenizer.model_max_length - len(row))
                        out = encoder(torch.tensor([row], device=device), output_hidden_states=True)
                        parts.append(out.hidden_states[layer])
                        if pooled is None and out[0].ndim == 2:  # text_encoder_2's projected pooled output
                            pooled = out[0]
                    hidden.append(torch.cat(parts, dim=1))
                encoded.append((torch.cat(hidden, dim=-1), pooled))
        if count > 1:
            logger.info("Long prompt: encoded as %d chunks of 75 tokens", count)
        (pos, pos_pooled), (neg, neg_pooled) = encoded
        return dict(prompt_embeds=pos, pooled_prompt_embeds=pos_pooled,
                    negative_prompt_embeds=neg, negative_pooled_prompt_embeds=neg_pooled)

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

        style_prompt = preset["positive"]
        framing = getattr(request, "framing", "full_body")
        if framing == "upper_body":
            style_prompt = style_prompt.replace("full body", "upper body").replace("reference sheet", "portrait")
            style_prompt += ", upper body, waist up, detailed face, both eyes visible, centered composition"
        model_tags = {} if getattr(self, "custom_model_path", None) else next(
            (m for m in config.MODEL_PRESETS if m["id"] == getattr(self, "model_name", None)), {})
        positive_parts = [request.prompt.strip(), style_prompt, model_tags.get("positive", "")]
        positive = ", ".join(p for p in positive_parts if p)

        negative_parts = [request.negative_prompt.strip(), preset.get("negative", ""), config.DEFAULT_NEGATIVE_PROMPT,
                          model_tags.get("negative", "")]
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
            import gc

            del self.pipeline
            self.pipeline = None
            gc.collect()  # offload hooks hold reference cycles; collect before freeing CUDA cache
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

    def inpaint(self, image, mask, prompt: str, negative: str = "", strength: float = 0.9, steps=None,
                guidance=None, seed=None, task_id=None, label="Repainting the selection", detail=1024, feather=6):
        """Demo: fill the selection with a blurred, colour-shifted version so the flows are testable."""
        from PIL import Image, ImageFilter, ImageOps

        image = image.convert("RGB")
        hard = mask.convert("L").point(lambda v: 255 if v > 127 else 0)
        painted = ImageOps.posterize(image.filter(ImageFilter.GaussianBlur(12)), 3)
        soft = hard.filter(ImageFilter.GaussianBlur(feather)) if feather else hard
        return Image.composite(painted, image, soft)

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
