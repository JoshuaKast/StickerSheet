#!/bin/bash
# Build Sticker Sheet.app for macOS using py2app.
# This script is macOS-only — do not run on other platforms.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Sticker Sheet macOS Build ==="

# Clean previous builds
rm -rf build dist

# Create build venv
python3 -m venv .venv-build
source .venv-build/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt py2app

# Generate icon PNGs if needed
if [ ! -f icons/StickerSheet.png ]; then
    echo "Generating icon PNGs..."
    python icons/generate_icons.py
    mv StickerSheet.png icons/
    mv StickerDoc.png icons/
fi

# Check for .icns files (require macOS iconutil)
if [ ! -f icons/StickerSheet.icns ]; then
    echo "WARNING: icons/StickerSheet.icns not found."
    echo "Run the iconutil commands from icons/README.md to create .icns files."
    echo "Continuing build without custom icons..."
fi

# Build the app bundle
python setup.py py2app

# Cleanup build venv
deactivate
rm -rf .venv-build

echo ""
echo "=== Build complete ==="
echo "App bundle: dist/Sticker Sheet.app"
echo ""
echo "To test: open 'dist/Sticker Sheet.app'"
echo "To install: cp -R 'dist/Sticker Sheet.app' /Applications/"
