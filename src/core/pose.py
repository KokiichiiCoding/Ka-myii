"""
Whole-body keypoints of the character with DWPose (yzd-v/DWPose, Apache-2.0) on ONNX Runtime.

Top-down only: the character's mask already gives the person box, so DWPose's YOLOX detector
is not needed. Returns COCO-WholeBody's 133 points (17 body, 6 feet, 68 face, 2 x 21 hands;
"left" is the character's left) in image pixels with confidences. Trained on photos, it is good
on clean full-body anime art and unreliable on hidden limbs: callers check the scores.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MODEL = ("yzd-v/DWPose", "dw-ll_ucoco_384.onnx", "1a7144101628d69ee7a3768d1ee3a094070dc388")
INPUT_W, INPUT_H = 288, 384
BODY = dict(nose=0, l_eye=1, r_eye=2, l_ear=3, r_ear=4, l_shoulder=5, r_shoulder=6, l_elbow=7, r_elbow=8,
            l_wrist=9, r_wrist=10, l_hip=11, r_hip=12, l_knee=13, r_knee=14, l_ankle=15, r_ankle=16,
            l_big_toe=17, l_small_toe=18, l_heel=19, r_big_toe=20, r_small_toe=21, r_heel=22)
FACE0, L_HAND0, R_HAND0 = 23, 91, 112   # face: 68-point layout (8 = chin, 36-47 eyes, 48-67 mouth)


def model_path(download: bool = False) -> Optional[str]:
    try:
        from huggingface_hub import hf_hub_download

        repo, name, revision = MODEL
        return hf_hub_download(repo, name, revision=revision, local_files_only=not download)
    except Exception as exc:
        logger.info("DWPose unavailable (%s)", type(exc).__name__)
        return None


def keypoints(image: Image.Image, mask: np.ndarray) -> Optional[np.ndarray]:
    """[133, 3] array of (x, y, score) in image pixels, or None without the model."""
    path = model_path()
    if path is None:
        return None
    import onnxruntime as ort

    ys, xs = np.nonzero(mask)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    w, h = (xs.max() - xs.min()) * 1.25, (ys.max() - ys.min()) * 1.25   # mmpose's bbox padding
    if w > h * INPUT_W / INPUT_H:
        h = w * INPUT_H / INPUT_W
    else:
        w = h * INPUT_W / INPUT_H
    box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    crop = np.asarray(image.convert("RGB").crop(box).resize((INPUT_W, INPUT_H), Image.Resampling.BILINEAR), np.float32)
    x = (crop - (123.675, 116.28, 103.53)) / (58.395, 57.12, 57.375)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    sx, sy = session.run(None, {"input": x.transpose(2, 0, 1)[None].astype(np.float32)})
    sx, sy = sx[0], sy[0]
    split = sx.shape[1] / INPUT_W   # SimCC bins per pixel (2.0)
    px, py = sx.argmax(1) / split, sy.argmax(1) / split
    score = np.minimum(sx.max(1), sy.max(1))
    return np.stack([box[0] + px * w / INPUT_W, box[1] + py * h / INPUT_H, score], 1)
