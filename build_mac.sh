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

# Convert PNGs to .icns if needed (requires macOS sips + iconutil)
for icon_name in StickerSheet StickerDoc; do
    if [ ! -f "icons/${icon_name}.icns" ] && [ -f "icons/${icon_name}.png" ]; then
        echo "Converting icons/${icon_name}.png -> icons/${icon_name}.icns ..."
        iconset_dir="icons/${icon_name}.iconset"
        mkdir -p "$iconset_dir"
        for size in 16 32 64 128 256 512; do
            sips -z $size $size "icons/${icon_name}.png" --out "${iconset_dir}/icon_${size}x${size}.png" >/dev/null
            sips -z $((size*2)) $((size*2)) "icons/${icon_name}.png" --out "${iconset_dir}/icon_${size}x${size}@2x.png" >/dev/null
        done
        iconutil -c icns "$iconset_dir" -o "icons/${icon_name}.icns"
        rm -rf "$iconset_dir"
        echo "  Created icons/${icon_name}.icns"
    fi
done

if [ ! -f icons/StickerSheet.icns ]; then
    echo "WARNING: icons/StickerSheet.icns not found and could not be generated."
    echo "Ensure icons/StickerSheet.png exists, or see icons/README.md."
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
