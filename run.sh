#!/bin/bash
# Ka-myii — Automated VTuber Model Studio (quick start)
set -e

echo "=================================="
echo " Ka-myii — VTuber Model Studio"
echo "=================================="
echo ""

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --quiet --upgrade pip

echo "Select mode:"
echo "  1) Demo mode      — no GPU, lightweight deps, explores the full pipeline"
echo "  2) Full / GPU     — installs PyTorch + diffusers for real generation"
read -r -p "Enter choice (1 or 2): " mode
echo ""

if [ "$mode" = "1" ]; then
    if ! python -c "import flask, PIL, numpy, psd_tools, pytoshop" 2>/dev/null; then
        echo "Installing demo dependencies..."
        pip install Flask flask-cors flask-socketio Pillow numpy psd-tools pytoshop six python-dotenv
    fi
    echo "Starting Ka-myii in DEMO mode at http://localhost:${PORT:-5000} ..."
    python app.py --dummy
else
    if ! python -c "import torch" 2>/dev/null; then
        echo "Installing full dependencies (this downloads PyTorch, may take a while)..."
        pip install -r requirements.txt
    fi
    echo "Starting Ka-myii at http://localhost:${PORT:-5000} ..."
    python app.py
fi
