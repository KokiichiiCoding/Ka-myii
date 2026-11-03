"""
Imported designs: framing the main figure, white backgrounds, and the upload job.
GPU-free: the colour-key cut-out stands in for the segmentation model.
"""
import numpy as np
from PIL import Image, ImageDraw


def _sheet():
    """A square 'reference sheet' on a busy background: a big figure and a small second view."""
    img = Image.new("RGB", (600, 600), (40, 120, 60))
    d = ImageDraw.Draw(img)
    for x in range(0, 600, 30):  # scenery stripes touching every border
        d.line((x, 0, x + 80, 600), fill=(60, 140, 80), width=6)
    alpha = np.zeros((600, 600), np.uint8)
    for box, colour in (((80, 120, 260, 560), (230, 190, 170)), ((420, 380, 500, 560), (230, 190, 170))):
        d.ellipse(box, fill=colour)
        a = Image.new("L", (600, 600), 0); ImageDraw.Draw(a).ellipse(box, fill=255)
        alpha = np.maximum(alpha, np.asarray(a))
    return img, alpha


def test_frame_keeps_the_main_figure_centred_and_large():
    from src.core import importer

    img, alpha = _sheet()
    framed, a = importer.frame(img, alpha)
    assert framed.size == importer.TARGET
    ys, xs = np.nonzero(a > 128)
    w, h = importer.TARGET
    assert abs((xs.min() + xs.max()) / 2 - w / 2) < 8           # centred
    assert (ys.max() - ys.min()) > 0.8 * h                      # scaled up to fill the frame
    assert (xs.max() - xs.min()) < 0.6 * w                      # only the big view (not both)


def test_white_background_keeps_soft_edges():
    from src.core import importer

    img, alpha = _sheet()
    white = np.asarray(importer.white_background(img, alpha))
    assert tuple(white[5, 5]) == (255, 255, 255) and tuple(white[340, 170]) == (230, 190, 170)
    soft = alpha.copy(); soft[340, 79] = 128
    mixed = np.asarray(importer.white_background(img, soft))[340, 79]
    assert 100 < mixed[1] < 255  # half background green, half white


def test_upload_job_prepares_a_framed_sketch(tmp_path, monkeypatch):
    from src.core import importer
    from src.models.vtuber_model import GenerationRequest
    from src.pipeline.assembly_line import AssemblyLine

    img, alpha = _sheet()
    monkeypatch.setattr(importer, "character_alpha", lambda image: alpha if image.size == (600, 600) else np.asarray(image.convert("L")) * 0)
    img.save(tmp_path / "upload.png")
    line = AssemblyLine(output_base_dir=tmp_path / "out", use_dummy_generators=True)
    model = line.generate_model(GenerationRequest(
        prompt="Uploaded artwork: sheet", source="upload", sketch=True, reference_image=str(tmp_path / "upload.png"),
        import_options={"frame": True, "plain_background": True, "enhance": 0.0}))
    base = Image.open(tmp_path / "out" / model.id / "base_image.png")
    assert model.status.value == "sketch" and base.size == importer.TARGET
    assert base.getpixel((3, 3)) == (255, 255, 255)        # the scenery is gone
    assert model.request.import_options is None and model.request.width == 832
