#!/bin/sh
set -e

# Check if .env file exists
if [ ! -f .env ]; then
    echo ".env file not found. Creating from env.example..."
    if [ -f env.example ]; then
        cp env.example .env
        echo ".env file created from env.example."
    else
        echo "Error: env.example file not found. Cannot create .env file. Exiting..."
        exit 1
    fi
fi

rm -rf .venv
uv venv -p 3.12
uv pip install -e . --quiet
uv run main.py
