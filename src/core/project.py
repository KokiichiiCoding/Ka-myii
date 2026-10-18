"""
Reopen a character exported from the Gallery (the "Export" zip of its folder) on this or another
machine: the zip is unpacked as a new result and the absolute paths its JSON files recorded
(metadata.json, the package's kamyii_manifest.json) are moved to the new folder.
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
import zipfile
from pathlib import Path, PurePosixPath

MAX_FILES = 5000
MAX_BYTES = 4 * 1024 ** 3  # unpacked; a full character with its package is ~100-300 MB
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def import_project(archive, output_dir: Path) -> str:
    """Unpack ``archive`` (a path or file object holding an exported character) into
    ``output_dir``; returns the new result's id. Raises ValueError for anything else."""
    try:
        package = zipfile.ZipFile(archive)
    except zipfile.BadZipFile:
        raise ValueError("That file is not a zip archive")
    with package:
        files = [i for i in package.infolist() if not i.is_dir()]
        # The metadata at the top, or inside one folder if the export was re-zipped with its folder.
        roots = sorted((PurePosixPath(i.filename).parent for i in files if PurePosixPath(i.filename).name == "metadata.json"),
                       key=lambda p: len(p.parts))
        if not roots or len(roots[0].parts) > 1:
            raise ValueError("Not a Ka-myii character export (no metadata.json)")
        root = roots[0]
        files = [i for i in files if root in PurePosixPath(i.filename).parents]
        if len(files) > MAX_FILES or sum(i.file_size for i in files) > MAX_BYTES:
            raise ValueError("The archive is too large for a character export")
        try:
            data = json.loads(package.read(str(root / "metadata.json") if root.parts else "metadata.json"))
        except (ValueError, UnicodeDecodeError):
            raise ValueError("The export's metadata.json is not valid JSON")
        if not isinstance(data, dict) or not isinstance(data.get("id"), str):
            raise ValueError("The export's metadata.json has no character id")

        old_id = data["id"]
        new_id = old_id if _ID.match(old_id) and not (output_dir / old_id).exists() else str(uuid.uuid4())
        staging = output_dir / "_uploads" / f"import-{uuid.uuid4().hex[:8]}"  # out of the Gallery's sight
        written = 0
        try:
            for info in files:
                rel = PurePosixPath(info.filename).relative_to(root)
                if info.filename.startswith(("/", "\\")) or ":" in info.filename or "\\" in info.filename or ".." in rel.parts:
                    raise ValueError(f"Unsafe path in the archive: {info.filename}")
                dest = staging.joinpath(*rel.parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with package.open(info) as src, open(dest, "wb") as out:
                    while chunk := src.read(1 << 20):  # counted, whatever the headers claim
                        written += len(chunk)
                        if written > MAX_BYTES:
                            raise ValueError("The archive is too large for a character export")
                        out.write(chunk)
            target = output_dir / new_id
            for path in staging.rglob("*.json"):
                _rebase_file(path, old_id, target)
            meta = json.loads((staging / "metadata.json").read_text(encoding="utf-8"))
            meta["id"] = new_id
            if new_id != old_id and meta.get("display_name"):
                meta["display_name"] = f"{meta['display_name']} (imported)"
            (staging / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            staging.rename(target)
        except zipfile.BadZipFile as e:
            shutil.rmtree(staging, ignore_errors=True)
            raise ValueError(f"The archive is damaged: {e}")
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return new_id


def rebase(value, old_id: str, target: Path):
    """``value`` with every path that ran through a folder named ``old_id`` moved into ``target``."""
    if isinstance(value, dict):
        return {k: rebase(v, old_id, target) for k, v in value.items()}
    if isinstance(value, list):
        return [rebase(v, old_id, target) for v in value]
    if isinstance(value, str) and ("/" in value or "\\" in value):
        parts = re.split(r"[\\/]", value)
        if old_id in parts and len(parts) > 1:
            return str(target.joinpath(*parts[parts.index(old_id) + 1:]))
    return value


def _rebase_file(path: Path, old_id: str, target: Path) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return  # not ours to fix
    moved = rebase(data, old_id, target)
    if moved != data:
        path.write_text(json.dumps(moved, indent=2), encoding="utf-8")
