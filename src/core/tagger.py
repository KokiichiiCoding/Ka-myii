"""
Danbooru-style tags for any image (SmilingWolf's WD ViT tagger v3, Apache-2.0, via onnxruntime).

Used to describe imported artwork (a Perchance / ChatGPT design has no Ka-myii prompt), to
seed region-edit prompts, and to check facial state (is the drawn mouth open?). Preprocessing
follows the reference demo (SmilingWolf/wd-tagger): white-padded square, 448 px, BGR float.
The 378 MB model downloads from Hugging Face on first use (inside a job, never a request).
"""
from __future__ import annotations

import csv
import logging
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

TAGGER = ("SmilingWolf/wd-vit-tagger-v3", "7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f")
GENERAL_THRESHOLD = 0.35
CHARACTER_THRESHOLD = 0.85
# Kaomoji tags keep their underscores (reference demo).
KAOMOJI = {"0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9",
           ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"}
# Describe the character, not the picture it came in: these would fight Ka-myii's own framing.
NOT_THE_CHARACTER = {"simple background", "white background", "grey background", "gradient background",
                     "transparent background", "solo", "looking at viewer", "upper body", "full body",
                     "cowboy shot", "portrait", "signature", "watermark", "artist name", "web address",
                     "border", "letterboxed", "pillarboxed", "english text", "text"}


def prepare(image: Image.Image, size: int = 448) -> np.ndarray:
    """(1, size, size, 3) float32 BGR in 0..255, the image padded to a white square."""
    rgba = image.convert("RGBA")
    flat = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    flat.alpha_composite(rgba)
    side = max(flat.size)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(flat.convert("RGB"), ((side - flat.width) // 2, (side - flat.height) // 2))
    if side != size:
        square = square.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(square, dtype=np.float32)[:, :, ::-1][None].copy()


def _labels(csv_path) -> List[tuple]:
    with open(csv_path, encoding="utf-8") as fh:
        return [(row["name"] if row["name"] in KAOMOJI else row["name"].replace("_", " "), int(row["category"]))
                for row in csv.DictReader(fh)]


def select(labels: List[tuple], probs: np.ndarray, general: float = GENERAL_THRESHOLD,
           character: float = CHARACTER_THRESHOLD) -> Dict:
    """Tags above threshold, most confident first: {"general": [...], "character": [...],
    "rating": str, "scores": {tag: p}}. Category 9 = rating, 4 = character, 0 = general."""
    order = np.argsort(-probs)
    out = {"general": [], "character": [], "rating": None, "scores": {}}
    best_rating = -1.0
    for i in order:
        name, cat = labels[i]
        p = float(probs[i])
        if cat == 9:
            if p > best_rating:
                best_rating, out["rating"] = p, name
        elif cat == 4 and p > character:
            out["character"].append(name)
        elif cat == 0 and p > general:
            out["general"].append(name)
        if cat in (0, 4):
            out["scores"][name] = p
    return out


def tag_image(image: Image.Image, download: bool = True) -> Optional[Dict]:
    """Tags for ``image``, or None when onnxruntime or the model is unavailable."""
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        repo, revision = TAGGER
        model = hf_hub_download(repo, "model.onnx", revision=revision, local_files_only=not download)
        labels = _labels(hf_hub_download(repo, "selected_tags.csv", revision=revision, local_files_only=not download))
        # Created per call: it is used rarely, and 400 MB must not stay resident beside See-through.
        session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    except Exception as exc:
        logger.warning("Image tagger unavailable (%s: %s)", type(exc).__name__, exc)
        return None
    size = session.get_inputs()[0].shape[1]
    probs = session.run(None, {session.get_inputs()[0].name: prepare(image, size)})[0][0]
    return select(labels, probs)


def describe(tags: Dict, limit: int = 40) -> str:
    """A prompt describing the character: named characters first, then visual tags."""
    keep = [t for t in tags["general"] if t not in NOT_THE_CHARACTER and not t.endswith(" background")]
    names = [t.replace("(", "\\(").replace(")", "\\)") for t in tags["character"] + keep[:limit]]
    return ", ".join(names)
