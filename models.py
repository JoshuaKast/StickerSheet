"""Data model classes and constants for Sticker Sheet Maker.

All layout math happens in 300 DPI pixel space.
"""

from dataclasses import dataclass, field


# === Constants (all layout math in 300 DPI pixel space) ===
DPI = 300
PAGE_WIDTH = int(8.5 * DPI)    # 2550
PAGE_HEIGHT = int(11.0 * DPI)  # 3300
MARGIN = int(0.25 * DPI)       # 75
CUT_GAP = 4                    # ~4px at 300 DPI between images

PRINTABLE_WIDTH = PAGE_WIDTH - 2 * MARGIN    # 2400
PRINTABLE_HEIGHT = PAGE_HEIGHT - 2 * MARGIN  # 3150

STICKER_FILTER = "Sticker Files (*.sticker)"

# Standard paper sizes as (name, width_inches, height_inches)
PAPER_SIZES = [
    ("US Letter (8.5 \u00d7 11 in)", 8.5, 11.0),
    ("US Legal (8.5 \u00d7 14 in)", 8.5, 14.0),
    ("Tabloid (11 \u00d7 17 in)", 11.0, 17.0),
    ("A4 (210 \u00d7 297 mm)", 8.267, 11.692),
    ("A3 (297 \u00d7 420 mm)", 11.692, 16.535),
]

CUT_LINE_STYLES = ["None", "Dashed", "Dotted", "Solid"]


# === Data Model ===

@dataclass
class PageSettings:
    """Configurable page layout settings."""
    paper_size_index: int = 0          # Index into PAPER_SIZES
    margin_inches: float = 0.25        # Margin on all sides, in inches
    cut_line_style: int = 0            # Index into CUT_LINE_STYLES (default: None)
    street_width_inches: float = 0.125  # Min gap between images, in inches (1/8")

    @property
    def page_width(self) -> int:
        _, w, _ = PAPER_SIZES[self.paper_size_index]
        return int(w * DPI)

    @property
    def page_height(self) -> int:
        _, _, h = PAPER_SIZES[self.paper_size_index]
        return int(h * DPI)

    @property
    def margin(self) -> int:
        return int(self.margin_inches * DPI)

    @property
    def cut_gap(self) -> int:
        return max(1, int(self.street_width_inches * DPI))

    @property
    def printable_width(self) -> int:
        return self.page_width - 2 * self.margin

    @property
    def printable_height(self) -> int:
        return self.page_height - 2 * self.margin

    def __getattr__(self, name):
        # Backward compat for old pickled PageSettings that may lack new fields
        defaults = {
            'paper_size_index': 0,
            'margin_inches': 0.25,
            'cut_line_style': 1,
            'street_width_inches': 0.125,
        }
        if name in defaults:
            return defaults[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

@dataclass
class StickerImage:
    """A single image stored as normalized PNG bytes."""
    png_data: bytes
    pixel_width: int
    pixel_height: int
    scale_step: int = 0  # Per-image scaling: each step ~ 20% size change
    mask: tuple[int, int, int, int] | None = None  # (x, y, w, h) crop rect in pixel coords

    @property
    def effective_width(self) -> int:
        """Width after mask is applied. Used by tiler for layout."""
        if self.mask is not None:
            return self.mask[2]
        return self.pixel_width

    @property
    def effective_height(self) -> int:
        """Height after mask is applied. Used by tiler for layout."""
        if self.mask is not None:
            return self.mask[3]
        return self.pixel_height

    def __getattr__(self, name):
        # Backward compat: old pickled instances may lack these fields
        defaults = {'scale_step': 0, 'mask': None}
        if name in defaults:
            return defaults[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


@dataclass
class PlacedImage:
    """An image with its computed placement on the page (300 DPI coords)."""
    image_index: int
    x: int
    y: int
    width: int
    height: int


@dataclass
class LayoutRow:
    """A row of images sharing the same quantized height."""
    y: int
    height: int
    placements: list[PlacedImage] = field(default_factory=list)


@dataclass
class LayoutResult:
    """Complete layout: rows of placed images."""
    rows: list[LayoutRow] = field(default_factory=list)

    @property
    def placements(self):
        return [p for row in self.rows for p in row.placements]


@dataclass
class StickerProject:
    """Full project state. Pickle-serializable."""
    images: list[StickerImage] = field(default_factory=list)
    layout: LayoutResult = field(default_factory=LayoutResult)
    settings: PageSettings = field(default_factory=PageSettings)

    def __getattr__(self, name):
        # Backward compat: old pickled instances lack settings
        if name == 'settings':
            return PageSettings()
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
