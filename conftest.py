"""Shared pytest fixtures for Sticker Sheet tests."""
import os

os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # must be set before QApplication import

import io
import pytest
from PIL import Image, ImageDraw


@pytest.fixture(scope='session')
def qapp():
    """Create a single QApplication for all tests."""
    from sticker_app import StickerApp
    app = StickerApp([])
    yield app


@pytest.fixture
def sample_images():
    """Generate simple Pillow shape images as PNG bytes (varied sizes and colors)."""
    images = []
    specs = [
        ('red', (200, 200)),
        ('blue', (300, 150)),
        ('green', (150, 300)),
        ('orange', (400, 400)),
        ('purple', (100, 250)),
    ]
    for color, size in specs:
        img = Image.new('RGB', size, color)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        images.append(buf.getvalue())
    return images


@pytest.fixture
def circle_image():
    """A single circle-on-white image for simple tests."""
    img = Image.new('RGB', (200, 200), 'white')
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 20, 180, 180], fill='red', outline='black')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


@pytest.fixture
def make_png():
    """Factory fixture: make_png(width, height, color) -> PNG bytes."""
    def _make(width, height, color='red'):
        img = Image.new('RGB', (width, height), color)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    return _make
