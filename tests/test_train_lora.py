"""GPU-free checks for scripts/train_lora.py: buckets, image preparation and batching."""
import importlib.util
import random
import zipfile
from pathlib import Path

from PIL import Image

spec = importlib.util.spec_from_file_location("train_lora", Path(__file__).resolve().parents[1] / "scripts" / "train_lora.py")
train_lora = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_lora)


def test_buckets_are_64_px_multiples_near_the_area():
    area = 896 * 896
    for w, h in train_lora.make_buckets(area):
        assert w % 64 == 0 and h % 64 == 0
        assert 0.5 <= w / h <= 2
        assert 0.85 * area <= w * h <= area


def test_prepare_fits_buckets_and_puts_the_trigger_first(tmp_path):
    archive = tmp_path / "set.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for i in range(12):
            size = (600, 880) if i % 2 else (880, 600)
            path = tmp_path / f"{i}.png"
            Image.new("RGB", size, (i * 20, 90, 160)).save(path)
            z.write(path, f"nested/{i}.png")
            z.writestr(f"nested/{i}.txt", "1girl, kmyvt, smile" if i else "")
        Image.new("RGB", (300, 300)).save(tmp_path / "tiny.png")
        z.write(tmp_path / "tiny.png", "tiny.png")  # under 512 px: skipped

    items = train_lora.prepare_images(archive, tmp_path / "data", "kmyvt", 896 * 896)

    assert len(items) == 12
    assert items[0]["tags"] == ["kmyvt"] and items[1]["tags"] == ["kmyvt", "1girl", "smile"]
    for it in items:
        w, h = it["size"]
        portrait = it["source"] in {f"{i}.png" for i in range(1, 12, 2)}
        assert (h > w) == portrait
        with Image.open(tmp_path / "data" / it["file"]) as im:
            assert im.size == (w, h)


def test_epoch_batches_cover_every_view_once_within_one_bucket():
    random.seed(0)
    items = [{"size": [832, 1216] if i % 3 else [1216, 832]} for i in range(9)]
    views = [2] * 9
    batches = train_lora.epoch_batches(items, views, batch_size=2)
    seen = [s for batch in batches for s in batch]
    assert sorted(seen) == [(i, v) for i in range(9) for v in range(2)]
    for batch in batches:
        assert len({tuple(items[i]["size"]) for i, _ in batch}) == 1
