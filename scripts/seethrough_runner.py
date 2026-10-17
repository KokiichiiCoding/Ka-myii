"""
See-through runner for low-VRAM GPUs. Executed by src/core/seethrough.py with the
engine's own python (engines/see-through/.venv) and the see-through repo as cwd.

Why not the repo's stock 8 GB scripts: on Windows (bitsandbytes 0.49 + diffusers 0.37)
diffusers *group offload* of NF4 weights crashes with "illegal memory access", and
NF4 without offload overflows 8 GB at 1280. This runner keeps the NF4 weights
(5 GB download instead of 13 GB bf16), caches the tag embeddings while everything is
on the GPU, then swaps UNet blocks in/out with plain module.to() using the repo's own
block-swap pipeline. Activations still dominate: ~8 GB at 1280, ~5 GB at 1024.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), "inference", "scripts"))

import torch  # noqa: E402
from inference_psd_quantized import (  # noqa: E402  (also pins BLAS thread env vars)
    build_layerdiff_pipeline,
    build_marigold_pipeline,
    run_layerdiff,
    run_marigold,
)
from inference_psd_blockswap import KDiffusionStableDiffusionXLPipelineBlockSwap  # noqa: E402
from utils.inference_utils import further_extr  # noqa: E402
from utils.torch_utils import seed_everything  # noqa: E402

LAYERDIFF_NF4 = "24yearsold/seethroughv0.0.2_layerdiff3d_nf4"
MARIGOLD_NF4 = "24yearsold/seethroughv0.0.1_marigold_nf4"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--srcp", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--resolution_depth", type=int, default=768)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tblr_split", action="store_true")
    a = p.parse_args()

    saved = os.path.join(a.save_dir, os.path.splitext(os.path.basename(a.srcp))[0])
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    print("Building LayerDiff3D pipeline...", flush=True)
    seed_everything(a.seed)
    pipe = build_layerdiff_pipeline(argparse.Namespace(
        quant_mode="nf4", repo_id_layerdiff=LAYERDIFF_NF4, cpu_offload=False, group_offload=False))
    # Re-class the working NF4 pipeline as the repo's block-swap pipeline (skipping its
    # __init__, whose bf16 cast does not apply to quantized weights), then enable swapping.
    pipe.__class__ = KDiffusionStableDiffusionXLPipelineBlockSwap
    pipe.blockswap_enabled, pipe.blockswap_device, pipe._blockswap_hooks = False, torch.device("cuda"), []
    # The block-swap __call__ feeds bf16; the NF4 checkpoint keeps its non-quantized layers
    # (conv_in, norms, biases) in fp16. Packed 4-bit weights are uint8 and are left alone.
    for t in list(pipe.unet.parameters()) + list(pipe.unet.buffers()):
        if t.is_floating_point() and t.dtype != torch.bfloat16:
            t.data = t.data.to(torch.bfloat16)
    # Cached tag embeddings come out of the text encoders as fp32; the block-swap path doesn't cast them.
    pipe._cached_prompt_embeds = {
        tag: [e.to(torch.bfloat16) for e in embeds] for tag, embeds in pipe._cached_prompt_embeds.items()
    }
    pipe.enable_blockswap("cuda")
    torch.cuda.empty_cache()

    print("Running LayerDiff3D (body + head)...", flush=True)
    run_layerdiff(pipe, a.srcp, a.save_dir, a.seed, a.num_inference_steps, a.resolution)
    del pipe
    torch.cuda.empty_cache()

    print("Building Marigold depth pipeline...", flush=True)
    marigold = build_marigold_pipeline(argparse.Namespace(
        quant_mode="nf4", repo_id_depth=MARIGOLD_NF4, cpu_offload=False, group_offload=False))
    print("Running Marigold depth...", flush=True)
    run_marigold(marigold, a.srcp, a.save_dir, a.seed, resolution_depth=a.resolution_depth)
    del marigold
    torch.cuda.empty_cache()

    print("Running PSD assembly...", flush=True)
    further_extr(saved, rotate=False, save_to_psd=True, tblr_split=a.tblr_split)
    print(f"Done in {time.time() - t0:.0f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
