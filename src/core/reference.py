"""Local visual conditioning for front redraws (lazy GPU imports; no ComfyUI)."""
from PIL import Image, ImageOps

MODEL = "h94/IP-Adapter"
REVISION = "018e402774aeeddd60609b4ecdb7e298259dc729"
WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"


def reference_square(image):
    """Fit the entire costume in CLIP's square; its default centre crop loses hems/boots."""
    rgba = image.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, "white")
    bg.alpha_composite(rgba)
    return ImageOps.pad(bg.convert("RGB"), (224, 224), method=Image.Resampling.LANCZOS, color="white")


def encode(image, device, cancelled=lambda: False):
    """Encode once, then free ViT-H before ControlNet loads (16 GB laptop RAM)."""
    import gc
    import torch
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    encoder = None
    try:
        if cancelled():
            raise InterruptedError("Generation cancelled by user")
        encoder = CLIPVisionModelWithProjection.from_pretrained(
            MODEL, subfolder="models/image_encoder", revision=REVISION,
            torch_dtype=torch.float16, use_safetensors=True, low_cpu_mem_usage=True).to(device)
        if cancelled():
            raise InterruptedError("Generation cancelled by user")
        pixels = CLIPImageProcessor()(images=reference_square(image), return_tensors="pt").pixel_values
        pixels = pixels.to(device=device, dtype=torch.float16)
        with torch.inference_mode():
            positive = encoder(pixels, output_hidden_states=True).hidden_states[-2]
            negative = encoder(torch.zeros_like(pixels), output_hidden_states=True).hidden_states[-2]
            # Diffusers expects [unconditional, conditional] x references x patches x features.
            result = torch.cat([negative, positive])[:, None].cpu()
        del pixels, positive, negative
        return result
    finally:
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
