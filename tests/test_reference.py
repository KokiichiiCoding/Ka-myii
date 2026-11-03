"""Reference fidelity contracts without torch, CUDA, or downloaded weights."""
from contextlib import nullcontext
import sys
from types import SimpleNamespace
import weakref

import numpy as np
from PIL import Image, ImageDraw
import pytest


class Tensor:
    """Small ndarray-backed stand-in for the tensor operations at our API boundary."""
    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def chunk(self, count):
        return [Tensor(a) for a in np.array_split(self.values, count)]


def test_reference_square_keeps_hem_and_boots_without_changing_source():
    from src.core.reference import reference_square
    source = Image.new("RGBA", (80, 224), (0, 0, 0, 0))
    draw = ImageDraw.Draw(source)
    draw.rectangle((0, 0, 79, 39), fill=(240, 180, 20, 255))
    draw.rectangle((0, 40, 79, 180), fill=(20, 60, 140, 255))
    draw.rectangle((0, 181, 79, 223), fill=(90, 45, 20, 255))
    draw.rectangle((20, 80, 40, 100), fill=(255, 0, 0, 128))
    before = source.tobytes()
    result = reference_square(source)
    assert result.size == (224, 224) and result.mode == "RGB"
    assert result.getpixel((112, 0)) == (240, 180, 20)
    assert result.getpixel((112, 223)) == (90, 45, 20)
    assert result.getpixel((0, 112)) == (255, 255, 255)
    assert result.getpixel((100, 90)) == (255, 127, 127)
    assert source.mode == "RGBA" and source.tobytes() == before


@pytest.mark.parametrize("failure", [None, "before_load", "after_load", "inference"])
def test_reference_encoder_contract_and_release(monkeypatch, failure):
    from src.core import reference
    events, encoder_refs = [], []

    class Encoder:
        @classmethod
        def from_pretrained(cls, model, **options):
            assert model == reference.MODEL and options["revision"] == reference.REVISION
            assert options["use_safetensors"] and options["low_cpu_mem_usage"]
            instance = cls()
            encoder_refs.append(weakref.ref(instance))
            events.append("load")
            return instance

        def to(self, device):
            assert device == "test-device"
            return self

        def __call__(self, pixels, output_hidden_states):
            assert output_hidden_states
            events.append("infer")
            if failure == "inference":
                raise RuntimeError("inference failed")
            value = 7 if np.any(pixels.values) else -3
            return SimpleNamespace(hidden_states=[None, Tensor(np.full((1, 3, 4), value)), None])

    class Processor:
        def __call__(self, *, images, return_tensors):
            assert images.size == (224, 224) and return_tensors == "pt"
            return SimpleNamespace(pixel_values=Tensor(np.ones((1, 3, 224, 224))))

    fake_torch = SimpleNamespace(
        float16="fp16", inference_mode=nullcontext,
        zeros_like=lambda t: Tensor(np.zeros_like(t.values)),
        cat=lambda tensors: Tensor(np.concatenate([t.values for t in tensors])),
        cuda=SimpleNamespace(empty_cache=lambda: events.append("release-cache")),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        CLIPImageProcessor=Processor, CLIPVisionModelWithProjection=Encoder))
    def cancelled():
        return failure == "before_load" or failure == "after_load" and bool(encoder_refs)
    if failure:
        expected = RuntimeError if failure == "inference" else InterruptedError
        with pytest.raises(expected):
            reference.encode(Image.new("RGB", (80, 120)), "test-device", cancelled)
    else:
        result = reference.encode(Image.new("RGB", (80, 120)), "test-device", cancelled)
        assert result.values.shape == (2, 1, 3, 4)
        assert np.all(result.values[0] == -3), "Unconditional features precede reference features"
        assert np.all(result.values[1] == 7)
    assert events[-1] == "release-cache"
    assert all(ref() is None for ref in encoder_refs), "The vision encoder must not linger beside ControlNet"
    if failure == "before_load":
        assert not encoder_refs


def _guided_generator(monkeypatch, *, failure=None):
    from src.core import reference
    from src.core.image_generator import ImageGenerator
    events, calls = [], []
    original_fraction = .7

    class Cuda:
        fraction = original_fraction
        is_available = staticmethod(lambda: True)
        get_per_process_memory_fraction = staticmethod(lambda: Cuda.fraction)
        empty_cache = staticmethod(lambda: None)
        reset_peak_memory_stats = staticmethod(lambda: None)
        max_memory_allocated = staticmethod(lambda: 0)
        max_memory_reserved = staticmethod(lambda: 0)

        @staticmethod
        def set_per_process_memory_fraction(value):
            Cuda.fraction = value
            events.append(("memory-limit", value))

    class Component:
        def __init__(self, name):
            self.name = name

        def enable_layerwise_casting(self, **options):
            events.append("cast-" + self.name)

    class MainPipe:
        tokenizer_2 = object()
        unet = Component("unet")

        def remove_all_hooks(self):
            events.append("remove-base-hooks")

        def register_modules(self, **kwargs):
            assert kwargs == {"text_encoder": None, "text_encoder_2": None}
            events.append("release-text-encoders")

        def load_ip_adapter(self, model, **options):
            assert model == reference.MODEL and options["revision"] == reference.REVISION
            assert options["image_encoder_folder"] is None
            events.append("load-adapter")
            if failure == "load-adapter":
                raise RuntimeError("adapter failed")

        def set_ip_adapter_scale(self, scale):
            events.append(("scale", scale))

    class ControlNet:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            events.append("load-controlnet")
            return Component("controlnet")

    class InpaintPipe:
        vae = SimpleNamespace(enable_tiling=lambda: events.append("tile-vae"))

        def enable_model_cpu_offload(self):
            events.append("offload")

        def remove_all_hooks(self):
            events.append("remove-inpaint-hooks")

        def __call__(self, **call):
            calls.append(call)
            if failure == "denoise":
                raise RuntimeError("denoise failed")
            return SimpleNamespace(images=[call["image"]])

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        ControlNetModel=ControlNet,
        AutoPipelineForInpainting=SimpleNamespace(from_pipe=lambda *args, **kwargs: InpaintPipe())))
    def encode(image, device, cancelled):
        assert not cancelled()
        events.append("encode-reference")
        return Tensor(np.array([-3, 7]).reshape(2, 1, 1, 1))
    monkeypatch.setattr(reference, "encode", encode)
    generator = object.__new__(ImageGenerator)
    generator.pipeline = MainPipe()
    generator.device = "cpu"
    generator._torch = SimpleNamespace(cuda=Cuda, float16="fp16", float8_e4m3fn="fp8")
    def prompt_embeds(*args):
        events.append("encode-text")
        return {"prompt_embeds": "encoded-text", "negative_prompt_embeds": "negative-text"}
    generator._sdxl_prompt_embeds = prompt_embeds
    def unload():
        events.append("unload")
        generator.pipeline = None
    generator.unload_model = unload
    return generator, events, calls, Cuda


@pytest.mark.parametrize("guidance, reference_strength", [(6.5, .65), (1.0, .8), (6.5, 0)])
def test_guided_redraw_encodes_before_controlnet_and_passes_correct_cfg(monkeypatch, guidance, reference_strength):
    generator, events, calls, cuda = _guided_generator(monkeypatch)
    image = Image.new("RGB", (96, 128), "blue")
    before = image.tobytes()
    output = generator.inpaint(image, Image.new("L", image.size, 255), "blue cloak",
        pose=image, appearance=image, guidance=guidance, reference_strength=reference_strength, detail=128)
    assert output.size == image.size and image.tobytes() == before
    assert events.index("encode-text") < events.index("release-text-encoders") < events.index("load-controlnet")
    if reference_strength:
        assert events.index("release-text-encoders") < events.index("encode-reference") < events.index("load-adapter")
        assert events.index("load-adapter") < events.index("cast-unet") < events.index("load-controlnet")
        assert ("scale", reference_strength) in events
        embeddings = calls[0]["ip_adapter_image_embeds"][0].values
        assert embeddings.ravel().tolist() == ([-3, 7] if guidance > 1 else [7])
    else:
        assert "encode-reference" not in events and "load-adapter" not in events
        assert "ip_adapter_image_embeds" not in calls[0]
    assert calls[0]["prompt_embeds"] == "encoded-text"
    assert "remove-inpaint-hooks" in events and generator.pipeline is None
    assert cuda.fraction == .7, "Restore the caller's lower CUDA allocation cap"


@pytest.mark.parametrize("failure", ["load-adapter", "denoise"])
def test_guided_reference_failure_unloads_and_restores_memory_limit(monkeypatch, failure):
    generator, events, calls, cuda = _guided_generator(monkeypatch, failure=failure)
    image = Image.new("RGB", (96, 128))
    with pytest.raises(RuntimeError, match="failed"):
        generator.inpaint(image, Image.new("L", image.size, 255), "blue cloak",
            pose=image, appearance=image, detail=128)
    assert generator.pipeline is None and "unload" in events
    assert cuda.fraction == .7
    if failure == "denoise":
        assert "remove-inpaint-hooks" in events


@pytest.mark.parametrize("mode", ["front", "full_body", None])
def test_repaint_uses_visual_reference_only_for_front_and_honours_saved_controls(tmp_path, mode):
    from src.models.vtuber_model import GenerationRequest
    from src.pipeline.assembly_line import AssemblyLine
    staged = tmp_path / "staged"
    staged.mkdir()
    output = tmp_path / "new"
    output.mkdir()
    image = Image.new("RGB", (96, 128), (20, 60, 140))
    image.save(staged / "source.png")
    Image.new("L", image.size, 255).save(staged / "mask.png")
    image.save(staged / "pose.png")
    calls = []
    def inpaint(source, mask, prompt, **options):
        calls.append(options)
        assert prompt == "gold embroidery"
        return source
    line = object.__new__(AssemblyLine)
    line.image_generator = SimpleNamespace(inpaint=inpaint)
    request = GenerationRequest(prompt="blue cloak", edit_prompt="gold embroidery", operation="extend",
        reference_image=str(staged / "source.png"), mask_image=str(staged / "mask.png"),
        pose_image=str(staged / "pose.png"), steps=37, guidance_scale=4.25,
        seed=1234, extension={"mode": mode, "reference_strength": .8} if mode else None)
    result = line._repaint(request, output, None)
    assert Image.open(result).tobytes() == image.tobytes()
    assert not list(staged.iterdir()), "Staged reference/mask/pose are removed after saving the new copy"
    assert calls[0]["steps"] == 37 and calls[0]["guidance"] == 4.25 and calls[0]["seed"] == 1234
    appearance = calls[0]["appearance"]
    assert (appearance is not None) == (mode == "front")
    if appearance is not None:
        assert appearance.tobytes() == image.tobytes()
        assert calls[0]["reference_strength"] == .8


def test_guided_preparation_failure_still_unloads(monkeypatch):
    generator, events, calls, cuda = _guided_generator(monkeypatch)
    image = Image.new("RGB", (96, 128))
    with pytest.raises(ValueError):
        generator.inpaint(image, Image.new("L", image.size, 255), "blue cloak",
            pose=image, appearance=image, detail=128, guidance="invalid")
    assert not calls and generator.pipeline is None
    assert "remove-inpaint-hooks" in events and cuda.fraction == .7
