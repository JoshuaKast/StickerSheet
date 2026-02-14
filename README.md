# Sticker Sheet Maker

A desktop app for tiling images onto a US Letter page for printing. Copy images to your clipboard, paste them into the app, and print onto sticker paper, photo paper, or any other media.

## How It Works

1. Copy an image to your clipboard and paste it into the app (Cmd/Ctrl+V), or drag and drop image files
2. The app automatically tiles all images onto a printable page with straight cutting paths between them
3. Print and cut along the gaps

Images are sized using log-scale compression so they come out roughly consistent in size regardless of source resolution. Heights are quantized into row bins to create straight horizontal cut lines across the full page.

## Requirements

- Python 3.10+
- PySide6
- Pillow

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python sticker_app.py
```

## Features

- **Clipboard paste & drag-and-drop** for adding images
- **Automatic row-based tiling** with straight cut-line gaps
- **Per-image scaling** (Shift+scroll on an image) and **page zoom** (Cmd/Ctrl+scroll)
- **Print** via native print dialog (File > Print), including print-to-PDF
- **Save/Load** projects as `.sticker` files
- **Undo/Redo** for paste and delete operations
- **Selection** — click to select, Delete to remove, right-click context menu

## Running Tests

```bash
pip install -r requirements-dev.txt
QT_QPA_PLATFORM=offscreen pytest -v
```

## License

See repository for license details.
