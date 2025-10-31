#!/bin/bash
# Ka-myii Quick Start Script

echo "================================"
echo "Ka-myii VTuber Model Generator"
echo "================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv venv
    echo "Virtual environment created!"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Dependencies not found. Installing..."
    pip install -r requirements.txt
    echo "Dependencies installed!"
    echo ""
fi

# Ask user which mode to run
echo "Select mode:"
echo "1) Dummy mode (no GPU required, for testing)"
echo "2) Production mode (requires GPU)"
read -p "Enter choice (1 or 2): " mode

echo ""
echo "Starting Ka-myii..."
echo ""

if [ "$mode" = "1" ]; then
    python app.py --dummy
else
    python app.py
fi
