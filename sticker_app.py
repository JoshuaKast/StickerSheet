#!/usr/bin/env python3
"""Sticker Sheet Maker - Tile pasted images onto a US Letter page for sticker printing.

This module serves as the application entry point and re-exports all public names
from the MVC modules for backward compatibility (existing tests, scripts, and
pickle deserialization all import from 'sticker_app').

Architecture:
    models.py      - Data model classes and constants (Model)
    tiler.py       - Layout engine (Tiler)
    views.py       - PageWidget, SettingsDialog, MaskEditorDialog (View)
    controller.py  - MainWindow, undo commands, StickerApp (Controller)
"""

import sys

# Re-export all public names so that `from sticker_app import X` continues to work.
# This also ensures pickle deserialization of old .sticker files (which reference
# 'sticker_app.StickerProject', etc.) can find the classes in this module.
from models import (  # noqa: F401
    DPI, PAGE_WIDTH, PAGE_HEIGHT, MARGIN, CUT_GAP,
    PRINTABLE_WIDTH, PRINTABLE_HEIGHT, STICKER_FILTER,
    PAPER_SIZES, CUT_LINE_STYLES,
    PageSettings, StickerImage, PlacedImage, LayoutRow, LayoutResult, StickerProject,
)
from tiler import Tiler  # noqa: F401
from views import PageWidget, SettingsDialog, MaskEditorDialog  # noqa: F401
from controller import (  # noqa: F401
    MainWindow, StickerApp,
    PasteImageCommand, ScaleImageCommand, DeleteImageCommand, MaskImageCommand,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sticker Sheet Maker")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("file", nargs="?", default=None, help="Open a .sticker file")
    args = parser.parse_args()

    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    app = StickerApp(sys.argv)
    app.setApplicationName("Sticker Sheet Maker")
    window = MainWindow()
    window.show()

    # Connect macOS file-open events (double-click .sticker in Finder)
    app.file_open_requested.connect(window.open_file)

    # Handle command-line file argument
    if args.file:
        window.open_file(args.file)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
