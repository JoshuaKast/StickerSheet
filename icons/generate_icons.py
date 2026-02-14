#!/usr/bin/env python3
"""Generate placeholder icon PNGs for Sticker Sheet Maker.

Creates 1024x1024 PNGs that can be converted to .icns on macOS using iconutil.
See README.md in this directory for conversion instructions.
"""
from PIL import Image, ImageDraw, ImageFont


def draw_app_icon(size=1024):
    """App icon: white rounded-rect page with colorful sticker squares."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = size // 8
    # Page background
    draw.rounded_rectangle(
        [margin, margin // 2, size - margin, size - margin // 2],
        radius=size // 20,
        fill=(255, 255, 255, 255),
        outline=(180, 180, 180, 255),
        width=size // 80,
    )

    # Grid of colorful "sticker" rectangles
    colors = [
        (231, 76, 60), (46, 204, 113), (52, 152, 219),
        (241, 196, 15), (155, 89, 182), (230, 126, 34),
    ]
    cols, rows = 3, 2
    pad = size // 6
    inner_w = size - 2 * pad
    inner_h = (size - 2 * pad) // 2
    cell_w = (inner_w - (cols - 1) * size // 40) // cols
    cell_h = (inner_h - (rows - 1) * size // 40) // rows

    for r in range(rows):
        for c in range(cols):
            x0 = pad + c * (cell_w + size // 40)
            y0 = pad + r * (cell_h + size // 40)
            color = colors[(r * cols + c) % len(colors)]
            draw.rounded_rectangle(
                [x0, y0, x0 + cell_w, y0 + cell_h],
                radius=size // 60,
                fill=color,
            )

    # "SS" text at bottom
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size // 6)
    except (OSError, IOError):
        font = ImageFont.load_default()
    text = "SS"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (size - tw) // 2
    ty = size - margin // 2 - th - size // 12
    draw.text((tx, ty), text, fill=(80, 80, 80, 255), font=font)

    return img


def draw_doc_icon(size=1024):
    """Document icon: page with a small sticker grid and .sticker label."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = size // 6
    # Dog-ear page shape
    page_left = margin
    page_top = margin // 2
    page_right = size - margin
    page_bottom = size - margin // 2
    ear = size // 5

    points = [
        (page_left, page_top),
        (page_right - ear, page_top),
        (page_right, page_top + ear),
        (page_right, page_bottom),
        (page_left, page_bottom),
    ]
    draw.polygon(points, fill=(255, 255, 255, 255), outline=(180, 180, 180, 255), width=size // 80)

    # Dog-ear fold
    draw.polygon(
        [(page_right - ear, page_top), (page_right, page_top + ear), (page_right - ear, page_top + ear)],
        fill=(220, 220, 220, 255),
        outline=(180, 180, 180, 255),
        width=size // 160,
    )

    # Small sticker grid
    colors = [(231, 76, 60), (52, 152, 219), (46, 204, 113), (241, 196, 15)]
    grid_margin = size // 4
    gw = (size - 2 * grid_margin) // 2 - size // 40
    gh = gw
    for i, color in enumerate(colors):
        r, c = divmod(i, 2)
        x0 = grid_margin + c * (gw + size // 20)
        y0 = grid_margin + r * (gh + size // 20)
        draw.rounded_rectangle([x0, y0, x0 + gw, y0 + gh], radius=size // 60, fill=color)

    # ".sticker" label
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size // 10)
    except (OSError, IOError):
        font = ImageFont.load_default()
    label = ".sticker"
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    tx = (size - tw) // 2
    ty = page_bottom - size // 6
    draw.text((tx, ty), label, fill=(100, 100, 100, 255), font=font)

    return img


if __name__ == '__main__':
    app_icon = draw_app_icon()
    app_icon.save('StickerSheet.png')
    print('Created StickerSheet.png (1024x1024)')

    doc_icon = draw_doc_icon()
    doc_icon.save('StickerDoc.png')
    print('Created StickerDoc.png (1024x1024)')
