# Icon Generation

## Step 1: Generate placeholder PNGs

```bash
cd icons
python generate_icons.py
```

This creates `StickerSheet.png` and `StickerDoc.png` (1024x1024 RGBA).

## Step 2: Convert to .icns (macOS only)

macOS requires `.icns` files in an iconset bundle. Run these commands on macOS:

```bash
# App icon
mkdir StickerSheet.iconset
for size in 16 32 64 128 256 512; do
    sips -z $size $size StickerSheet.png --out StickerSheet.iconset/icon_${size}x${size}.png
    sips -z $((size*2)) $((size*2)) StickerSheet.png --out StickerSheet.iconset/icon_${size}x${size}@2x.png
done
iconutil -c icns StickerSheet.iconset
rm -rf StickerSheet.iconset

# Document icon
mkdir StickerDoc.iconset
for size in 16 32 64 128 256 512; do
    sips -z $size $size StickerDoc.png --out StickerDoc.iconset/icon_${size}x${size}.png
    sips -z $((size*2)) $((size*2)) StickerDoc.png --out StickerDoc.iconset/icon_${size}x${size}@2x.png
done
iconutil -c icns StickerDoc.iconset
rm -rf StickerDoc.iconset
```

The resulting `StickerSheet.icns` and `StickerDoc.icns` are used by `setup.py` for py2app bundling.
