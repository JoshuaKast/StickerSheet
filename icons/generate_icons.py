#!/usr/bin/env python3
"""Generate icon PNGs for Sticker Sheet Maker.

Creates 1024x1024 PNGs that can be converted to .icns on macOS using iconutil.
See README.md in this directory for conversion instructions.

Design: light cream page with scattered colorful rounded rectangles (stickers).
"""
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# Light cream page color
PAGE_COLOR = (255, 253, 240, 255)
PAGE_OUTLINE = (210, 200, 175, 255)
PAGE_SHADOW = (180, 170, 150, 80)

# Sticker colors — vibrant, friendly palette
STICKER_COLORS = [
    (231, 76, 60),    # red
    (52, 152, 219),   # blue
    (46, 204, 113),   # green
    (241, 196, 15),   # yellow
    (155, 89, 182),   # purple
    (230, 126, 34),   # orange
    (26, 188, 156),   # teal
    (236, 100, 159),  # pink
]


def _draw_page(draw, size, margin_x, margin_top, margin_bottom, radius):
    """Draw a light cream page with subtle shadow."""
    page_rect = [margin_x, margin_top, size - margin_x, size - margin_bottom]

    # Subtle drop shadow (offset down-right)
    shadow_offset = size // 80
    shadow_rect = [
        page_rect[0] + shadow_offset,
        page_rect[1] + shadow_offset,
        page_rect[2] + shadow_offset,
        page_rect[3] + shadow_offset,
    ]
    draw.rounded_rectangle(shadow_rect, radius=radius, fill=PAGE_SHADOW)

    # Page body
    draw.rounded_rectangle(
        page_rect,
        radius=radius,
        fill=PAGE_COLOR,
        outline=PAGE_OUTLINE,
        width=max(size // 120, 2),
    )
    return page_rect


def draw_app_icon(size=1024):
    """App icon: cream page with colorful rounded-rect stickers in a grid layout."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = size // 8
    page_rect = _draw_page(draw, size, margin, margin // 2, margin // 2, size // 18)
    px0, py0, px1, py1 = page_rect

    # Sticker grid: 3 columns, 3 rows of rounded rectangles
    # with varying widths to look like actual sticker sheets
    pad_x = int((px1 - px0) * 0.07)
    pad_y = int((py1 - py0) * 0.06)
    gap = size // 36
    inner_x0 = px0 + pad_x
    inner_y0 = py0 + pad_y
    inner_x1 = px1 - pad_x
    inner_y1 = py1 - pad_y
    inner_w = inner_x1 - inner_x0
    inner_h = inner_y1 - inner_y0

    # Define sticker layout: list of (row, col_start_frac, col_end_frac, color_idx)
    # This gives an organic sticker-sheet feel with different widths
    stickers = [
        # Row 0 (top): two stickers
        (0, 0.00, 0.55, 0),
        (0, 0.58, 1.00, 1),
        # Row 1 (middle): three stickers
        (1, 0.00, 0.30, 2),
        (1, 0.33, 0.67, 3),
        (1, 0.70, 1.00, 4),
        # Row 2 (bottom): two stickers
        (2, 0.00, 0.42, 5),
        (2, 0.45, 1.00, 6),
    ]

    n_rows = 3
    row_h = (inner_h - (n_rows - 1) * gap) / n_rows
    corner_r = size // 30

    for row, c0_frac, c1_frac, ci in stickers:
        x0 = inner_x0 + int(c0_frac * inner_w)
        x1 = inner_x0 + int(c1_frac * inner_w)
        y0 = inner_y0 + int(row * (row_h + gap))
        y1 = y0 + int(row_h)
        color = STICKER_COLORS[ci % len(STICKER_COLORS)]

        # Draw sticker with slight border for depth
        draw.rounded_rectangle(
            [x0, y0, x1, y1],
            radius=corner_r,
            fill=color,
        )
        # Subtle lighter highlight on top edge
        highlight = tuple(min(c + 40, 255) for c in color[:3]) + (100,)
        draw.rounded_rectangle(
            [x0 + 2, y0 + 2, x1 - 2, y0 + int(row_h * 0.15)],
            radius=corner_r,
            fill=highlight,
        )

    return img


def draw_doc_icon(size=1024):
    """Document icon: cream dog-ear page with small sticker grid and .sticker label."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = size // 6
    page_left = margin
    page_top = margin // 2
    page_right = size - margin
    page_bottom = size - margin // 2
    ear = size // 5

    # Shadow
    shadow_offset = size // 80
    shadow_points = [
        (page_left + shadow_offset, page_top + shadow_offset),
        (page_right - ear + shadow_offset, page_top + shadow_offset),
        (page_right + shadow_offset, page_top + ear + shadow_offset),
        (page_right + shadow_offset, page_bottom + shadow_offset),
        (page_left + shadow_offset, page_bottom + shadow_offset),
    ]
    draw.polygon(shadow_points, fill=PAGE_SHADOW)

    # Dog-ear page
    points = [
        (page_left, page_top),
        (page_right - ear, page_top),
        (page_right, page_top + ear),
        (page_right, page_bottom),
        (page_left, page_bottom),
    ]
    draw.polygon(points, fill=PAGE_COLOR, outline=PAGE_OUTLINE, width=max(size // 120, 2))

    # Dog-ear fold
    fold_color = (245, 240, 225, 255)
    draw.polygon(
        [(page_right - ear, page_top), (page_right, page_top + ear), (page_right - ear, page_top + ear)],
        fill=fold_color,
        outline=PAGE_OUTLINE,
        width=max(size // 160, 1),
    )

    # Small sticker grid (2x2)
    colors = [STICKER_COLORS[0], STICKER_COLORS[1], STICKER_COLORS[2], STICKER_COLORS[3]]
    grid_margin = size // 4
    gw = (size - 2 * grid_margin) // 2 - size // 40
    gh = gw
    corner_r = size // 50
    for i, color in enumerate(colors):
        r, c = divmod(i, 2)
        x0 = grid_margin + c * (gw + size // 20)
        y0 = grid_margin + r * (gh + size // 20)
        draw.rounded_rectangle([x0, y0, x0 + gw, y0 + gh], radius=corner_r, fill=color)

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
    draw.text((tx, ty), label, fill=(120, 110, 95, 255), font=font)

    return img


if __name__ == '__main__':
    app_icon = draw_app_icon()
    app_icon.save('StickerSheet.png')
    print('Created StickerSheet.png (1024x1024)')

    doc_icon = draw_doc_icon()
    doc_icon.save('StickerDoc.png')
    print('Created StickerDoc.png (1024x1024)')
