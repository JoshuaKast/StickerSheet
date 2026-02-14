#!/usr/bin/env python3
"""Generate HOWTO.md with screenshots by driving the app headlessly.

Uses QT_QPA_PLATFORM=offscreen so no display server is needed.
Produces howto/ directory with step screenshots and a HOWTO.md file.

Usage:
    QT_QPA_PLATFORM=offscreen python generate_howto.py
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PIL import Image, ImageDraw
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication

from sticker_app import MainWindow, StickerApp


def generate_shape_images():
    """Create a variety of colorful shape images using Pillow."""
    shapes = []

    # Red circle
    img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([10, 10, 190, 190], fill='red')
    shapes.append(('red_circle', img))

    # Blue rectangle
    img = Image.new('RGBA', (300, 150), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 290, 140], fill='blue')
    shapes.append(('blue_rect', img))

    # Green diamond
    img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.polygon([(100, 10), (190, 100), (100, 190), (10, 100)], fill='green')
    shapes.append(('green_diamond', img))

    # Orange triangle
    img = Image.new('RGBA', (250, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.polygon([(125, 10), (240, 190), (10, 190)], fill='orange')
    shapes.append(('orange_triangle', img))

    # Purple star (approximated as a polygon)
    img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    import math
    cx, cy, r_outer, r_inner = 100, 100, 90, 40
    points = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        r = r_outer if i % 2 == 0 else r_inner
        points.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
    draw.polygon(points, fill='purple')
    shapes.append(('purple_star', img))

    # Yellow hexagon
    img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    cx, cy, r = 100, 100, 90
    hex_points = []
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3
        hex_points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(hex_points, fill='gold')
    shapes.append(('yellow_hexagon', img))

    return shapes


def capture_widget(widget, path):
    """Grab a widget's rendered content as a PNG file."""
    pixmap = widget.grab(widget.rect())
    pixmap.save(path, 'PNG')
    print(f'  Captured: {path}')


def main():
    app = StickerApp([])
    win = MainWindow()
    # Override closeEvent to avoid unsaved-changes dialog
    win.closeEvent = lambda event: event.accept()
    win.resize(QSize(800, 600))
    win.show()

    # Force initial layout
    app.processEvents()

    output_dir = 'howto'
    os.makedirs(output_dir, exist_ok=True)

    print('Generating howto screenshots...')

    # Step 1: Empty app
    capture_widget(win, os.path.join(output_dir, '01_empty.png'))

    # Generate shape images
    shapes = generate_shape_images()

    # Step 2: Paste first image
    name, pil_img = shapes[0]
    sticker = win.page_widget._pil_to_sticker(pil_img)
    win.project.images.append(sticker)
    win._retile()
    app.processEvents()
    capture_widget(win, os.path.join(output_dir, '02_first_paste.png'))

    # Step 3: Paste a few more
    for name, pil_img in shapes[1:4]:
        sticker = win.page_widget._pil_to_sticker(pil_img)
        win.project.images.append(sticker)
    win._retile()
    app.processEvents()
    capture_widget(win, os.path.join(output_dir, '03_several_pasted.png'))

    # Step 4: Paste all shapes — full tiling view
    for name, pil_img in shapes[4:]:
        sticker = win.page_widget._pil_to_sticker(pil_img)
        win.project.images.append(sticker)
    win._retile()
    app.processEvents()
    capture_widget(win, os.path.join(output_dir, '04_all_tiled.png'))

    # Step 5: Select an image
    win.page_widget.selected_index = 0
    win.page_widget.selection_changed.emit()
    win.page_widget.update()
    app.processEvents()
    capture_widget(win, os.path.join(output_dir, '05_selected.png'))

    # Step 6: Save the project
    save_path = os.path.join(output_dir, 'demo.sticker')
    win._write_file(save_path)
    app.processEvents()
    capture_widget(win, os.path.join(output_dir, '06_saved.png'))

    # Write HOWTO.md
    howto_path = 'HOWTO.md'
    with open(howto_path, 'w') as f:
        f.write('# Sticker Sheet Maker - How To Use\n\n')

        f.write('## 1. Launch the App\n\n')
        f.write('When you open the app, you see an empty US Letter page ready for stickers.\n\n')
        f.write('![Empty app](howto/01_empty.png)\n\n')

        f.write('## 2. Paste Your First Image\n\n')
        f.write('Copy an image from your browser or any source, then press **Ctrl+V** (or **Cmd+V** on macOS) ')
        f.write('to paste it. The image appears on the page.\n\n')
        f.write('![First paste](howto/02_first_paste.png)\n\n')

        f.write('## 3. Paste More Images\n\n')
        f.write('Keep pasting images. The app automatically tiles them into rows with uniform height ')
        f.write('for easy cutting. You can also drag and drop image files onto the window.\n\n')
        f.write('![Several pasted](howto/03_several_pasted.png)\n\n')

        f.write('## 4. View the Tiled Layout\n\n')
        f.write('All images are arranged in rows with straight cutting lines (gaps) between them. ')
        f.write('Horizontal gaps span the full page width between rows. Vertical gaps separate images within a row.\n\n')
        f.write('![All tiled](howto/04_all_tiled.png)\n\n')

        f.write('## 5. Select and Manage Images\n\n')
        f.write('Click an image to select it (shown with a blue highlight). ')
        f.write('Press **Delete** to remove the selected image, or right-click for a context menu with Copy and Delete options. ')
        f.write('Use **Ctrl+Z** / **Ctrl+Y** to undo and redo.\n\n')
        f.write('![Selected image](howto/05_selected.png)\n\n')

        f.write('## 6. Save Your Project\n\n')
        f.write('Use **File > Save** (Ctrl+S) to save your project as a `.sticker` file. ')
        f.write('You can reopen it later with **File > Open**.\n\n')
        f.write('![Saved project](howto/06_saved.png)\n\n')

        f.write('## 7. Print\n\n')
        f.write('Use **File > Print** (Ctrl+P) to open the system print dialog. ')
        f.write('Print onto sticker paper, then cut along the gaps between images. ')
        f.write('The print output is high-quality at 300 DPI.\n\n')

        f.write('You can also use the system print dialog\'s "Print to PDF" option to save a PDF.\n\n')

        f.write('---\n\n')
        f.write('*This guide was auto-generated by `generate_howto.py` using headless rendering.*\n')

    print(f'\nGenerated {howto_path} with screenshots in {output_dir}/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
