#!/bin/sh
set -e

# Check if .env file exists
if [ ! -f .env ]; then
    echo ".env file not found. Creating from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo ".env file created from .env.example."
    else
        echo "Error: .env.example file not found. Cannot create .env file. Exiting..."
        exit 1
    fi
fi

# Load environment variables from .env
echo "Loading environment variables from .env..."
export $(grep -v '^#' .env | xargs)

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Virtual environment created."
fi

. .venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo "Dependencies installed."

pip install -e . --quiet

python3 main.py

deactivate