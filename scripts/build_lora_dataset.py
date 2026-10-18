"""
Build a LoRA training set of rig-ready VTuber character art: one character facing the viewer,
upper body, eyes and mouth clear, arms down, on a plain white background. That is what
See-through decomposes and the expression painter animates best, so a LoRA trained on it makes
the checkpoint produce that reliably.

    python scripts/build_lora_dataset.py --out datasets/vtmodel --count 240
    python scripts/build_lora_dataset.py --out datasets/vtmodel --extra D:/my_commissions

Stages, each resumable (re-run the same command to continue):
1. generate: varied designs (the lists below) with the default checkpoint, saved to raw/;
2. curate: the pipeline's own checks (one figure, not cut off at the top or sides, plain
   background, WD-tagger tags for pose, eyes, hands, text, rating), keepers to images/;
3. caption: "<trigger>, <tags>" .txt beside each image, a contact sheet and a zip for the
   trainer notebook (image + same-name .txt).

--extra adds your own pictures (framed and whitened like imports). Only use art you have the
rights to train on.
"""
import argparse
import json
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TRIGGER = "kmyvt"
NEGATIVE = (
    "lowres, worst quality, low quality, blurry, jpeg artifacts, watermark, signature, text, username, logo, "
    "cropped, out of frame, head out of frame, multiple girls, multiple boys, multiple views, reference sheet, "
    "profile, from side, from behind, looking away, looking to the side, dutch angle, foreshortening, "
    "closed eyes, one eye closed, wink, hair over eyes, hair over one eye, eyepatch, glasses, mask, "
    "covering mouth, hand on face, hand up, hands near face, holding, weapon, crossed arms, bad anatomy, "
    "bad hands, extra fingers, missing fingers, extra arms, deformed, busy background, scenery, "
    "detailed background, gradient background, shadow on face, nsfw"
)
HAIR_COLOUR = ["silver hair", "white hair", "black hair", "blonde hair", "pink hair", "light blue hair", "blue hair",
               "purple hair", "red hair", "brown hair", "green hair", "orange hair", "aqua hair", "grey hair",
               "multicolored hair, streaked hair", "gradient hair"]  # "two-tone hair" split whole pictures
HAIR_STYLE_GIRL = ["long hair", "very long hair", "medium hair", "short hair", "twintails", "ponytail",
                   "side ponytail", "twin braids", "side braid", "bob cut", "hime cut", "low twintails",
                   "wavy hair", "hair bun", "double bun", "drill hair"]
HAIR_STYLE_BOY = ["short hair", "messy hair", "medium hair", "spiked hair", "low ponytail", "undercut", "wolf cut"]
BANGS = ["blunt bangs", "swept bangs", "parted bangs", "ahoge", "sidelocks"]
EYES = ["blue eyes", "red eyes", "green eyes", "purple eyes", "yellow eyes", "pink eyes", "aqua eyes",
        "brown eyes", "orange eyes", "grey eyes", "heterochromia"]
EXTRAS_ANY = ["cat ears", "fox ears", "wolf ears", "horns", "demon horns", "halo", "headphones",
              "animal ear headphones", "beret", "witch hat", "earrings", "choker", "hair ornament", "hairclip"]
EXTRAS_GIRL = ["hair ribbon", "hair flower", "star hair ornament", "x hair ornament", "tiara", "maid headdress",
               "rabbit ears", "hair bow", "neck ribbon"]
OUTFIT_ANY = ["oversized hoodie, open jacket", "techwear, cropped jacket, high collar", "japanese clothes, haori",
              "military uniform, epaulettes", "turtleneck sweater, ribbed sweater", "cardigan, collared shirt",
              "track jacket", "dress shirt, vest, necktie", "business suit, white shirt, necktie",
              "witch, cape, capelet", "chinese clothes", "streetwear, jacket, t-shirt", "trench coat",
              "bodysuit, cyberpunk, jacket", "priest, cassock", "varsity jacket"]
OUTFIT_GIRL = ["idol, idol clothes, frilled dress", "school uniform, serafuku, sailor collar, neckerchief",
               "maid, maid apron, puffy sleeves", "japanese clothes, kimono, obi", "miko, hakama, detached sleeves",
               "gothic lolita, frills, black dress", "nun, habit", "magical girl, frilled skirt, bow",
               "china dress", "sleeveless shirt, detached sleeves", "off-shoulder dress",
               "sailor dress, sailor hat", "elegant dress, frills", "blazer, school uniform, plaid skirt"]
OUTFIT_BOY = ["school uniform, gakuran"]
EXPRESSION = ["light smile, closed mouth", "smile, closed mouth", "closed mouth", ":3", "gentle smile",
              "smug, closed mouth", "expressionless, closed mouth", "happy, closed mouth"]
RENDER = ["flat color, clean lineart", "cel shading, clean lineart", "soft shading, clean lineart",
          "anime coloring, clean lineart", "pastel colors, clean lineart", "vibrant colors, clean lineart"]
FIXED = "solo, looking at viewer, facing viewer, straight-on, arms at sides, virtual youtuber, simple background, white background"

# Curation: tags (WD tagger v3 names) that break rigging or captions, with the score that rejects.
REJECT_TAGS = {
    "closed eyes": 0.35, "one eye closed": 0.35, "hair over eyes": 0.35, "hair over one eye": 0.35,
    "eyepatch": 0.35, "blindfold": 0.35, "glasses": 0.5, "mask": 0.5, "covering own mouth": 0.35,
    "hand on own face": 0.35, "hand on own cheek": 0.35, "hand up": 0.5, "hands up": 0.5, "arm up": 0.5,
    "v": 0.5, "finger to mouth": 0.35, "holding": 0.5, "from side": 0.5, "from behind": 0.35, "profile": 0.35,
    "looking to the side": 0.5, "looking away": 0.5, "multiple views": 0.35, "multiple girls": 0.35,
    "multiple boys": 0.35, "2girls": 0.35, "text": 0.5, "english text": 0.5, "watermark": 0.35,
    "signature": 0.5, "blurry": 0.5, "monochrome": 0.35, "greyscale": 0.35, "sketch": 0.5,
    "border": 0.35, "framed": 0.35, "outside border": 0.35, "picture frame": 0.35, "split screen": 0.35,
}
REQUIRE_TAGS = {"solo": 0.5, "looking at viewer": 0.5, "simple background": 0.35}
CAPTION_DROP = {"virtual youtuber"}  # in every prompt; the trigger carries it


def compose(rng: random.Random):
    """(prompt, framing) for one varied design."""
    boy = rng.random() < 0.2
    parts = ["1boy" if boy else "1girl",
             rng.choice(HAIR_COLOUR), rng.choice(HAIR_STYLE_BOY if boy else HAIR_STYLE_GIRL), rng.choice(BANGS),
             rng.choice(EYES), rng.choice(EXPRESSION)]
    extras = EXTRAS_ANY + ([] if boy else EXTRAS_GIRL)
    parts += rng.sample(extras, rng.choice([0, 1, 1, 2]))
    parts.append(rng.choice(OUTFIT_ANY + (OUTFIT_BOY if boy else OUTFIT_GIRL)))
    parts.append(rng.choice(RENDER))
    cowboy = rng.random() < 0.5  # "upper body" alone often fills the frame to the top edge
    parts.append("cowboy shot, standing" if cowboy else "upper body")
    parts.append(FIXED)
    return ", ".join(parts), ("none" if cowboy else "upper_body")


def generate(out: Path, count: int, seed: int) -> None:
    from src.core.image_generator import ImageGenerator
    from src.models.vtuber_model import GenerationRequest

    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    jobs = [(i, *compose(rng), rng.randrange(2 ** 31)) for i in range(count)]  # same list on every resume
    todo = [j for j in jobs if not (raw / f"{j[0]:04d}.png").exists()]
    if not todo:
        return
    generator = ImageGenerator(sampler="Euler a")
    started = time.time()
    for n, (i, prompt, framing, image_seed) in enumerate(todo, 1):
        request = GenerationRequest(prompt=prompt, negative_prompt=NEGATIVE, style="anime", framing=framing,
                                    width=832, height=1216, steps=28, guidance_scale=5.0, seed=image_seed)
        generator.generate(request, raw / f"{i:04d}.png")
        (raw / f"{i:04d}.json").write_text(json.dumps({"prompt": prompt, "framing": framing, "seed": image_seed}))
        eta = (time.time() - started) / n * (len(todo) - n)
        print(f"generated {i:04d} ({n}/{len(todo)}, ~{eta / 60:.0f} min left)", flush=True)
    generator.unload_model()


class Tagger:
    """The WD tagger with one ONNX session for the whole run (tagger.tag_image opens one per call)."""

    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from src.core import tagger

        repo, revision = tagger.TAGGER
        self.labels = tagger._labels(hf_hub_download(repo, "selected_tags.csv", revision=revision))
        self.session = ort.InferenceSession(hf_hub_download(repo, "model.onnx", revision=revision),
                                            providers=["CPUExecutionProvider"])
        self.size = self.session.get_inputs()[0].shape[1]

    def __call__(self, image):
        from src.core import tagger

        name = self.session.get_inputs()[0].name
        return tagger.select(self.labels, self.session.run(None, {name: tagger.prepare(image, self.size)})[0][0])


def verdict(image: Image.Image, alpha: np.ndarray, tags: dict) -> list:
    """Why ``image`` is not a rig-ready character (empty list = keep). ``alpha`` is its
    character mask (uint8), ``tags`` the tagger's output."""
    from src.core.asset_separator import _components
    from src.core.outpaint import touched_borders

    reasons = []
    fg = alpha > 128
    if fg.mean() < 0.05:
        return ["no character found"]
    small = np.asarray(Image.fromarray(fg.astype(np.uint8) * 255).resize((fg.shape[1] // 8, fg.shape[0] // 8))) > 128
    sizes = np.bincount(_components(small).ravel())[1:]
    if len(sizes) > 1 and np.sort(sizes)[-2] > 0.03 * sizes.max():
        reasons.append("more than one figure")
    if touched_borders(fg)["top"]:
        reasons.append("cut off at the top")
    # Shoulders and sleeves running off the sides low down are normal for upper-body art.
    head = touched_borders(fg[: int(0.55 * fg.shape[0])])
    reasons += [f"cut off at the {side}" for side in ("left", "right") if head[side]]
    ys = np.nonzero(fg.any(axis=1))[0]
    if ys.max() - ys.min() < 0.7 * fg.shape[0]:
        reasons.append("figure too small")
    rgb = np.asarray(image.convert("RGB"))
    outside = alpha < 16
    if outside.sum() > 1000 and (rgb[outside].min(axis=1) > 225).mean() < 0.9:
        reasons.append("background not plain white")
    # Ink away from the figure: frames, borders, sparkles, text (thin, so the share above misses it).
    near = np.asarray(Image.fromarray(fg.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(25))) > 0
    if ((rgb.min(axis=2) <= 225) & ~near).sum() > 0.01 * max(1, (~near).sum()):
        reasons.append("decorations in the background")
    scores = tags["scores"]
    reasons += [f"tag: {t}" for t, limit in REJECT_TAGS.items() if scores.get(t, 0) >= limit]
    reasons += [f"missing tag: {t}" for t, limit in REQUIRE_TAGS.items() if scores.get(t, 0) < limit]
    if tags["rating"] in ("questionable", "explicit"):
        reasons.append(f"rating: {tags['rating']}")
    return reasons


def caption(tags: dict, trigger: str) -> str:
    """Trigger first, then the general tags (most confident first); character names are dropped
    so the LoRA learns a look, not someone's character."""
    words = [t for t in tags["general"] if t not in CAPTION_DROP]
    return ", ".join([trigger] + [w.replace("(", "\\(").replace(")", "\\)") for w in words])


def _near_duplicate(image: Image.Image, seen: list) -> bool:
    small = np.asarray(image.convert("L").resize((16, 16), Image.Resampling.BOX), np.float32)
    bits = small > small.mean()
    if any((bits != other).mean() < 0.05 for other in seen):
        return True
    seen.append(bits)
    return False


def curate(out: Path, extra: Path, trigger: str) -> dict:
    from src.core import importer

    images = out / "images"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir(parents=True)
    tag = Tagger()
    sources = [(p, "generated") for p in sorted((out / "raw").glob("*.png"))]
    if extra:
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        sources += [(p, "extra") for p in sorted(extra.rglob("*")) if p.suffix.lower() in exts]
    report, seen = {}, []
    for path, kind in sources:
        with Image.open(path) as img:
            image = img.convert("RGB")
        if kind == "extra":  # framed on white like an import
            image, _ = importer.prepare(image)
        alpha = importer.character_alpha(image)
        tags = tag(image)
        reasons = verdict(image, alpha, tags)
        if not reasons and _near_duplicate(image, seen):
            reasons = ["near-duplicate"]
        name = f"{'gen' if kind == 'generated' else 'own'}_{path.stem}"
        report[name] = {"source": str(path), "kept": not reasons, "reasons": reasons}
        if not reasons:
            image.save(images / f"{name}.png")
            (images / f"{name}.txt").write_text(caption(tags, trigger), encoding="utf-8")
        print(f"{name}: {'keep' if not reasons else '; '.join(reasons)}", flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def package(out: Path, trigger: str, report: dict) -> Path:
    kept = sorted((out / "images").glob("*.png"))
    thumbs = [Image.open(p).convert("RGB").resize((104, 152)) for p in kept]
    cols = 12
    sheet = Image.new("RGB", (cols * 104, max(1, -(-len(thumbs) // cols)) * 152), "white")
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * 104, (i // cols) * 152))
    sheet.save(out / "contact_sheet.jpg", quality=88)
    reasons = {}
    for item in report.values():
        for r in item["reasons"]:
            reasons[r] = reasons.get(r, 0) + 1
    info = {"trigger": trigger, "images": len(kept), "candidates": len(report),
            "rejections": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
            "base_model": "cagliostrolab/animagine-xl-4.0 (SDXL)", "resolution": "832x1216",
            "use": f"Prompt with '{trigger}' plus normal Danbooru tags; LoRA weight 0.6-1.0."}
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in kept:
            z.write(p, p.name)
            z.write(p.with_suffix(".txt"), p.with_suffix(".txt").name)
        z.write(out / "dataset_info.json", "dataset_info.json")
    print(json.dumps(info, indent=2))
    print("Dataset:", out / "images", "| zip:", archive, "| contact sheet:", out / "contact_sheet.jpg")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("datasets/vtmodel"))
    parser.add_argument("--count", type=int, default=240, help="candidates to generate (about half are kept)")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--trigger", default=TRIGGER)
    parser.add_argument("--extra", type=Path, help="a folder of your own art to add (framed and checked too)")
    parser.add_argument("--skip-generate", action="store_true", help="curate what raw/ already holds")
    args = parser.parse_args()
    if not args.skip_generate and args.count:
        generate(args.out, args.count, args.seed)
    report = curate(args.out, args.extra, args.trigger)
    package(args.out, args.trigger, report)


if __name__ == "__main__":
    main()
