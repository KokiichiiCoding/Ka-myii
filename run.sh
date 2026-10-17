#!/bin/bash
# Ka-myii — VTuber Model Studio (Linux / macOS launcher)
set -e
cd "$(dirname "$0")"
export FLASK_ENV=production

if [ ! -x .venv/bin/python ]; then
    PY=""
    for candidate in python3.12 python3.11 python3.13 python3.10 python3; do
        if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)'; then
            PY="$candidate"; break
        fi
    done
    if [ -z "$PY" ]; then
        echo "Ka-myii needs Python 3.10 to 3.13 (3.12 recommended)." >&2
        exit 1
    fi
    echo "Creating .venv with $PY ..."
    "$PY" -m venv .venv
fi

# Install on first run and whenever requirements.txt changes.
if ! cmp -s requirements.txt .venv/kamyii-requirements.txt; then
    echo "Installing dependencies (first run or an update)..."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    cp requirements.txt .venv/kamyii-requirements.txt
fi

if command -v nvidia-smi >/dev/null && [ ! -x engines/see-through/.venv/bin/python ] && [ ! -e engines/see-through.skip ]; then
    read -r -p "Install the See-through layer engine (8 GB+ NVIDIA GPU, ~9 GB download)? [Y/n] " answer
    if [ "$answer" = "n" ] || [ "$answer" = "N" ]; then
        mkdir -p engines && echo declined > engines/see-through.skip
    else
        .venv/bin/python scripts/install_seethrough.py || echo "See-through install failed; the basic splitter will be used."
    fi
fi

echo "  1) Studio - real generation (NVIDIA GPU)"
echo "  2) Demo   - no GPU or downloads; a drawn character runs the full pipeline"
default=2; command -v nvidia-smi >/dev/null && default=1
read -r -p "Choose 1 or 2 [$default]: " mode
mode=${mode:-$default}
if [ "$mode" = "2" ]; then
    exec .venv/bin/python app.py --dummy --open
else
    exec .venv/bin/python app.py --open
fi
