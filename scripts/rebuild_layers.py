"""
Rebuild existing results' layers and exports from their saved See-through PSD — no GPU.

    python scripts/rebuild_layers.py <model_id> [<model_id> ...]

Use after improving Ka-myii's layer post-processing (draw order, filtering) to upgrade older
characters in seconds. The artwork and See-through's own PSD are untouched; the layer PNGs,
PSD/ORA, Cubism files, preview image and metadata are regenerated.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.core import seethrough  # noqa: E402
from src.core.asset_separator import AssetSeparator  # noqa: E402
from src.core.model_assembler import ModelAssembler  # noqa: E402
from src.models.vtuber_model import VTuberModel  # noqa: E402
from src.utils.helpers import load_json, save_json  # noqa: E402


def rebuild(model_id: str) -> int:
    model_dir = config.OUTPUT_DIR / model_id
    psd = model_dir / "assets" / "seethrough" / "base_image.psd"
    if not psd.exists():
        raise SystemExit(f"{model_id}: no saved See-through PSD (demo or older result)")
    model = VTuberModel.from_dict(load_json(model_dir / "metadata.json"))

    seethrough.available = lambda: True
    seethrough.decompose = lambda image_path, work_dir, progress=None: psd
    staging = model_dir / "assets_rebuild"
    shutil.rmtree(staging, ignore_errors=True)
    # Separate into a staging folder first so a failure leaves the current layers intact.
    assets = AssetSeparator().separate(model_dir / "base_image.png", staging)
    for old in (model_dir / "assets").glob("*.png"):
        old.unlink()
    for asset in assets:
        asset.file_path = Path(shutil.move(str(asset.file_path), str(model_dir / "assets" / asset.file_path.name)))
    shutil.rmtree(staging)

    model.assets = assets
    shutil.rmtree(model_dir / "model", ignore_errors=True)
    assembler = ModelAssembler()
    model.final_model_path = assembler.assemble(assets, model_dir / "model", model.name)
    manifest = load_json(model.final_model_path / "kamyii_manifest.json")
    for key in ("artifacts", "layers", "parameters"):
        model.metadata[key] = manifest.get(key, model.metadata.get(key))
    assembler.create_preview_image(assets, model_dir / "preview.png")
    save_json(model.to_dict(), model_dir / "metadata.json")
    return len(assets) - 1  # minus the helper cut-out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for model_id in sys.argv[1:]:
        print(f"{model_id}: rebuilt with {rebuild(model_id)} layers")
