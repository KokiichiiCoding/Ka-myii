"""
Train an SDXL LoRA on a Ka-myii dataset from the command line, with progress you can follow: a
live bar per epoch (loss, learning rate, VRAM, time left) and a summary line after each epoch,
also appended to <work>/<name>/output/training_log.csv. Same method as the notebook
(Downloads/image_checkpoint_trainer_sdxl.ipynb): aspect buckets, cached latents and Ka-myii's
chunked prompt encoding, LoRA on the UNet attention layers, min-SNR loss, noise offset.

    .venv\\Scripts\\python.exe scripts\\train_lora.py
    .venv\\Scripts\\python.exe scripts\\train_lora.py --dataset datasets\\vtmodel.zip --epochs 12
    .venv\\Scripts\\python.exe scripts\\train_lora.py --resume     # continue after an interruption
    .venv\\Scripts\\python.exe scripts\\train_lora.py --samples    # also a before/after picture

Don't run it while Ka-myii is generating: both need the whole 8 GB GPU.
Only train on images you have the rights to.
"""
import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
BATCH_SIZE = 1
GRAD_ACCUMULATION = 2
CAPTION_VARIANTS = 4      # tag orders per image (trigger stays first)
CAPTION_DROPOUT = 0.05    # share of steps with an empty caption (keeps the model promptable)
MIN_SNR_GAMMA = 5.0
NOISE_OFFSET = 0.0357     # learns pure whites/blacks better (white backgrounds)


def make_buckets(area, step=64):
    out = set()
    for w in range(step * 8, 2048 + 1, step):
        h = int(area / w) // step * step
        if h >= step * 8 and 0.5 <= w / h <= 2:
            out.add((w, h))
    return sorted(out)


def prepare_images(dataset: Path, data_dir: Path, trigger: str, area: int) -> list:
    """Unpack (flattened) and fit each image to its nearest aspect bucket; returns the items."""
    from PIL import Image, ImageOps

    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True)
    source = dataset
    if source.is_file() and source.suffix.lower() == ".zip":
        source = data_dir / "unpacked"
        source.mkdir()
        with zipfile.ZipFile(dataset) as z:
            for info in z.infolist():
                name = Path(info.filename).name  # flattened: no paths from the archive are used
                if not info.is_dir() and name and Path(name).suffix.lower() in EXTS | {".txt"}:
                    (source / name).write_bytes(z.read(info))
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    buckets = make_buckets(area)
    items = []
    for path in sorted(p for p in source.rglob("*") if p.suffix.lower() in EXTS):
        txt = path.with_suffix(".txt")
        tags = [t.strip() for t in (txt.read_text(encoding="utf-8") if txt.exists() else "").split(",") if t.strip()]
        if trigger and (not tags or tags[0] != trigger):
            tags = [trigger] + [t for t in tags if t != trigger]
        try:
            with Image.open(path) as raw:
                img = ImageOps.exif_transpose(raw).convert("RGB")
        except Exception as exc:
            print("  skipping unreadable", path.name, exc)
            continue
        if min(img.size) < 512:
            print("  skipping small image", path.name, img.size)
            continue
        w, h = min(buckets, key=lambda b: abs(math.log((b[0] / b[1]) / (img.width / img.height))))
        scale = max(w / img.width, h / img.height)
        img = img.resize((math.ceil(img.width * scale), math.ceil(img.height * scale)), Image.Resampling.LANCZOS)
        left, top = (img.width - w) // 2, (img.height - h) // 2
        name = f"{len(items):05d}.png"
        img.crop((left, top, left + w, top + h)).save(data_dir / name)
        items.append({"file": name, "tags": tags, "size": [w, h], "source": path.name})
    if len(items) < 10:
        raise ValueError(f"Only {len(items)} usable images; a LoRA needs at least 10 (50-150 is better).")
    (data_dir / "items.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    return items


def variant(tags, rng):
    rest = tags[1:]
    rng.shuffle(rest)
    return ", ".join(tags[:1] + rest)


def epoch_batches(items, views, batch_size=BATCH_SIZE):
    """Shuffled batches of (item, view) that share one bucket size."""
    by_bucket = {}
    for i, it in enumerate(items):
        for view in range(views[i]):
            by_bucket.setdefault(tuple(it["size"]), []).append((i, view))
    batches = []
    for samples in by_bucket.values():
        random.shuffle(samples)
        batches += [samples[k:k + batch_size] for k in range(0, len(samples), batch_size)]
    random.shuffle(batches)
    return batches


def duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def cache(items, args, data_dir, cache_dir, device):
    """Latents (plus mirrored copies) and caption embeddings, so training holds only the UNet."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from tqdm import tqdm
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from src.core.image_generator import chunk_token_ids

    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True)
    rng = random.Random(args.seed)
    vae = AutoencoderKL.from_pretrained(args.base, subfolder="vae", torch_dtype=torch.float32).to(device)
    vae.requires_grad_(False)
    for i, it in enumerate(tqdm(items, desc="  images", unit="img", dynamic_ncols=True)):
        img = Image.open(data_dir / it["file"]).convert("RGB")
        latents = []
        for view in ([img, ImageOps.mirror(img)] if args.flip else [img]):
            x = torch.from_numpy(np.asarray(view)).permute(2, 0, 1).float().div(127.5).sub(1)[None].to(device)
            with torch.no_grad():
                latents.append((vae.encode(x).latent_dist.sample() * vae.config.scaling_factor)[0].half().cpu())
        torch.save(latents, cache_dir / f"{i:05d}.lat.pt")
    del vae
    torch.cuda.empty_cache()

    tok1 = CLIPTokenizer.from_pretrained(args.base, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(args.base, subfolder="tokenizer_2")
    te1 = CLIPTextModel.from_pretrained(args.base, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    te2 = CLIPTextModelWithProjection.from_pretrained(args.base, subfolder="text_encoder_2",
                                                      torch_dtype=torch.float16).to(device)
    comma = tok1.convert_tokens_to_ids(",</w>")

    @torch.no_grad()
    def encode(text, max_chunks=3):
        # Ka-myii's prompt encoding: 75-token chunks cut at commas, penultimate layer (clip skip 2).
        ids = tok1(text, add_special_tokens=False, truncation=False, verbose=False).input_ids
        chunks = chunk_token_ids(ids, comma)[:max_chunks]
        hidden, pooled = [], None
        for tokenizer, encoder in ((tok1, te1), (tok2, te2)):
            parts = []
            for chunk in chunks:
                row = [tokenizer.bos_token_id, *chunk, tokenizer.eos_token_id]
                row += [tokenizer.pad_token_id] * (tokenizer.model_max_length - len(row))
                out = encoder(torch.tensor([row], device=device), output_hidden_states=True)
                parts.append(out.hidden_states[-2])
                if pooled is None and out[0].ndim == 2:  # text_encoder_2's projected pooled output
                    pooled = out[0]
            hidden.append(torch.cat(parts, dim=1))
        return torch.cat(hidden, dim=-1)[0].half().cpu(), pooled[0].half().cpu()

    for i, it in enumerate(tqdm(items, desc="  captions", unit="img", dynamic_ncols=True)):
        texts = [", ".join(it["tags"])] + [variant(it["tags"], rng) for _ in range(CAPTION_VARIANTS - 1)]
        torch.save([encode(t) for t in texts], cache_dir / f"{i:05d}.txt.pt")
    torch.save(encode(""), cache_dir / "empty.pt")
    del te1, te2
    torch.cuda.empty_cache()


def train(items, args, cache_dir, out_dir, device):
    """Returns the final LoRA path, or None when stopped with Ctrl+C."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from diffusers import DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import cast_training_params, compute_snr
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    unet = UNet2DConditionModel.from_pretrained(args.base, subfolder="unet", torch_dtype=dtype).to(device)
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    cast_training_params(unet, dtype=torch.float32)  # trainable LoRA weights in fp32
    params = [p for p in unet.parameters() if p.requires_grad]
    noise_scheduler = DDPMScheduler.from_pretrained(args.base, subfolder="scheduler")

    latents_cache = [torch.load(cache_dir / f"{i:05d}.lat.pt") for i in range(len(items))]
    caption_cache = [torch.load(cache_dir / f"{i:05d}.txt.pt") for i in range(len(items))]
    empty = torch.load(cache_dir / "empty.pt")
    empty_chunk = empty[0][:77]  # pads shorter captions in a batch, like Ka-myii pads prompts
    views = [len(lat) for lat in latents_cache]

    def make_batch(batch):
        lat = torch.stack([latents_cache[i][view] for i, view in batch])
        conds = [empty if random.random() < CAPTION_DROPOUT else random.choice(caption_cache[i]) for i, _ in batch]
        longest = max(c[0].shape[0] for c in conds)
        embeds = torch.stack([torch.cat([c[0]] + [empty_chunk] * ((longest - c[0].shape[0]) // 77)) for c in conds])
        pooled = torch.stack([c[1] for c in conds])
        w, h = items[batch[0][0]]["size"]
        time_ids = torch.tensor([[h, w, 0, 0, h, w]] * len(batch), dtype=torch.float32)
        return lat, embeds, pooled, time_ids

    steps_per_epoch = len(epoch_batches(items, views))
    updates_per_epoch = math.ceil(steps_per_epoch / GRAD_ACCUMULATION)
    total_updates = updates_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    lr_scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=max(1, total_updates // 20),
                                 num_training_steps=total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    def save_lora(weight_name):
        state = {k: v.to(torch.float16) for k, v in convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)).items()}
        StableDiffusionXLPipeline.save_lora_weights(out_dir, unet_lora_layers=state, weight_name=weight_name)

    start_epoch = 0
    resume_file = out_dir / "resume.pt"
    if args.resume:
        if not resume_file.exists():
            raise FileNotFoundError(f"--resume: no resume point at {resume_file}")
        state = torch.load(resume_file, map_location="cpu")
        set_peft_model_state_dict(unet, state["lora"])
        cast_training_params(unet, dtype=torch.float32)
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"]
        print(f"  resuming after epoch {start_epoch}")

    log_path = out_dir / "training_log.csv"
    new_log = not log_path.exists() or not args.resume
    log = open(log_path, "w" if new_log else "a", newline="", encoding="utf-8")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["epoch", "steps", "mean_loss", "seconds", "learning_rate", "peak_vram_gb", "saved"])

    print(f"  {sum(p.numel() for p in params) / 1e6:.1f} M trainable LoRA parameters, {dtype}")
    print(f"  {len(items)} images x {views[0]} views = {steps_per_epoch} steps per epoch, "
          f"{args.epochs - start_epoch} epochs to go ({steps_per_epoch * (args.epochs - start_epoch)} steps)")
    unet.train()
    epoch_times, losses, saved_epoch, epoch = [], [], start_epoch, start_epoch
    try:
        for epoch in range(start_epoch, args.epochs):
            began = time.time()
            batches = epoch_batches(items, views)
            epoch_losses = []
            torch.cuda.reset_peak_memory_stats()
            bar = tqdm(batches, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="step", dynamic_ncols=True)
            for k, batch in enumerate(bar):
                lat, embeds, pooled, time_ids = (t.to(device) for t in make_batch(batch))
                lat = lat.to(dtype)
                noise = torch.randn_like(lat)
                if NOISE_OFFSET:
                    noise += NOISE_OFFSET * torch.randn(lat.shape[0], lat.shape[1], 1, 1, device=device, dtype=lat.dtype)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (lat.shape[0],), device=device).long()
                noisy = noise_scheduler.add_noise(lat, noise, timesteps)
                with torch.autocast("cuda", dtype=dtype):
                    pred = unet(noisy, timesteps, encoder_hidden_states=embeds.to(dtype),
                                added_cond_kwargs={"text_embeds": pooled.to(dtype), "time_ids": time_ids.to(dtype)}).sample
                loss = F.mse_loss(pred.float(), noise.float(), reduction="none").mean(dim=(1, 2, 3))
                if MIN_SNR_GAMMA:
                    snr = compute_snr(noise_scheduler, timesteps)
                    loss = loss * torch.clamp(snr, max=MIN_SNR_GAMMA) / snr  # epsilon prediction
                loss = loss.mean()
                scaler.scale(loss / GRAD_ACCUMULATION).backward()
                if (k + 1) % GRAD_ACCUMULATION == 0 or k + 1 == len(batches):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(loss.item())
                epoch_losses.append(losses[-1])
                bar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.1e}",
                                vram=f"{torch.cuda.max_memory_allocated() / 1024**3:.1f}G")
            bar.close()

            saved = ""
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                saved = f"{args.name}-ep{epoch + 1:02d}.safetensors"
                save_lora(saved)
                torch.save({"lora": get_peft_model_state_dict(unet), "optimizer": optimizer.state_dict(),
                            "lr": lr_scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch + 1},
                           resume_file)
                saved_epoch = epoch + 1
            seconds = time.time() - began
            epoch_times.append(seconds)
            left = (args.epochs - epoch - 1) * sum(epoch_times) / len(epoch_times)
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Epoch {epoch + 1}/{args.epochs} done in {duration(seconds)} "
                  f"({seconds / len(batches):.2f} s/step) | mean loss {np.mean(epoch_losses):.4f} | "
                  f"lr {lr_scheduler.get_last_lr()[0]:.1e} | peak VRAM {peak:.1f} GB | "
                  f"{duration(left) + ' left' if left else 'finished'}" + (f" | saved {saved}" if saved else ""),
                  flush=True)
            writer.writerow([epoch + 1, len(batches), f"{np.mean(epoch_losses):.5f}", round(seconds, 1),
                             f"{lr_scheduler.get_last_lr()[0]:.3e}", round(peak, 2), saved])
            log.flush()
    except KeyboardInterrupt:
        print(f"\nStopped during epoch {epoch + 1}. " + (
            f"Last saved: epoch {saved_epoch}; continue with --resume." if saved_epoch else "Nothing was saved yet."))
        return None
    finally:
        log.close()
    final = out_dir / f"{args.name}.safetensors"
    save_lora(final.name)
    return final


def samples(args, lora: Path, vram_gb: float):
    """Same prompts and seeds without (top row) and with the LoRA (bottom), at Ka-myii's settings."""
    import torch
    from PIL import Image
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

    prompts = [f"{args.trigger}, 1girl, silver hair, long hair, blue eyes, light smile, hoodie, upper body",
               f"{args.trigger}, 1girl, pink hair, twintails, red eyes, idol clothes, cowboy shot",
               f"{args.trigger}, 1boy, black hair, short hair, grey eyes, black jacket, upper body"]
    quality = "masterpiece, best quality, very aesthetic, absurdres, high score, great score"
    negative = ("lowres, worst quality, low quality, bad anatomy, bad hands, blurry, jpeg artifacts, watermark, "
                "signature, text, cropped, multiple views, low score, bad score, average score")
    pipe = StableDiffusionXLPipeline.from_pretrained(args.base, torch_dtype=torch.float16)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    if vram_gb < 12:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="look")
    grid = Image.new("RGB", (416 * len(prompts), 608 * 2), "white")
    for r, weight in enumerate((0.0, 0.8)):
        pipe.set_adapters(["look"], adapter_weights=[weight])
        for c, prompt in enumerate(prompts):
            print(f"  sample {r * len(prompts) + c + 1}/{2 * len(prompts)} (LoRA weight {weight})", flush=True)
            g = torch.Generator("cuda").manual_seed(args.seed + c)
            image = pipe(f"{prompt}, {quality}", negative_prompt=negative, width=832, height=1216,
                         num_inference_steps=28, guidance_scale=5.0, generator=g).images[0]
            grid.paste(image.resize((416, 608)), (c * 416, r * 608))
    path = lora.parent / "samples_before_after.jpg"
    grid.save(path)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "vtmodel.zip",
                   help="zip or folder of images with same-name .txt captions")
    p.add_argument("--name", default="kamyii-vtmodel", help="LoRA file name")
    p.add_argument("--base", default="cagliostrolab/animagine-xl-4.0", help="SDXL model in diffusers format")
    p.add_argument("--trigger", default="kmyvt", help="first tag of every caption")
    p.add_argument("--work", type=Path, default=ROOT / "datasets" / "lora_training", help="working folder")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate (5e-5 for a lighter touch)")
    p.add_argument("--rank", type=int, default=16, help="8 = lighter, 32 = more capacity (bigger file)")
    p.add_argument("--area", type=int, default=None,
                   help="training pixels per image (default 1024*1024, or 896*896 under 12 GB of VRAM)")
    p.add_argument("--save-every", type=int, default=2, help="save a LoRA and a resume point every N epochs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-flip", dest="flip", action="store_false", help="no mirrored copies (asymmetric designs)")
    p.add_argument("--resume", action="store_true", help="continue after the last saved epoch")
    p.add_argument("--samples", action="store_true", help="draw a before/after grid when done (~10 min on 8 GB)")
    p.add_argument("--no-install", dest="install", action="store_false",
                   help="don't copy the LoRA into Ka-myii's models/loras")
    args = p.parse_args()

    import torch

    if not torch.cuda.is_available():
        sys.exit("Training needs an NVIDIA GPU with CUDA PyTorch; run this with Ka-myii's .venv python.")
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    area = args.area or (1024 * 1024 if vram_gb >= 12 else 896 * 896)
    project = args.work / args.name
    out_dir = project / "output"
    print(f"GPU {props.name}, {vram_gb:.1f} GB | dataset {args.dataset} | output {out_dir}")
    started = time.time()

    print("[1/4] Preparing images")
    items = prepare_images(args.dataset, project / "data", args.trigger, area)
    sizes = {}
    for it in items:
        sizes[tuple(it["size"])] = sizes.get(tuple(it["size"]), 0) + 1
    print(f"  {len(items)} images; buckets " + ", ".join(f"{w}x{h}: {n}" for (w, h), n in sorted(sizes.items())))
    print("[2/4] Encoding images and captions (once, so training holds only the UNet)")
    cache(items, args, project / "data", project / "cache", "cuda")
    print(f"[3/4] Training ({duration(time.time() - started)} so far)")
    final = train(items, args, project / "cache", out_dir, "cuda")
    if final is None:
        sys.exit(1)
    info = (f"Base model: {args.base}\nTrigger: {args.trigger}\nImages: {len(items)}\nEpochs: {args.epochs}\n"
            f"Rank/alpha: {args.rank}/{args.rank}\nLearning rate: {args.lr}\nBucket area: {area}\n")
    (out_dir / "TRAINING_INFO.txt").write_text(info, encoding="utf-8")
    print(f"[4/4] Done in {duration(time.time() - started)}: {final}")
    if args.install:
        target = ROOT / "models" / "loras" / final.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final, target)
        print(f"  installed in Ka-myii: {target} (pick it under LoRA, start prompts with '{args.trigger}')")
    if args.samples:
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        print("Drawing before/after samples")
        print("  saved", samples(args, final, vram_gb))


if __name__ == "__main__":
    main()
