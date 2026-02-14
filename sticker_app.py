#!/usr/bin/env python3
"""Sticker Sheet Maker - Tile pasted images onto a US Letter page for sticker printing."""

import io
import math
import pickle
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from PIL import Image
from PySide6.QtCore import Qt, QRectF, QPointF, QByteArray, QBuffer, QIODevice, QEvent, Signal
from PySide6.QtGui import (
    QAction, QImage, QPixmap, QPainter, QPen, QColor, QKeySequence, QUndoStack, QUndoCommand,
)
from PySide6.QtPrintSupport import QPrintDialog, QPrinter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStatusBar, QFileDialog, QMessageBox, QMenu,
    QDialog, QDialogButtonBox, QFormLayout, QComboBox, QDoubleSpinBox, QVBoxLayout, QGroupBox,
)


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
    ("US Letter (8.5 × 11 in)", 8.5, 11.0),
    ("US Legal (8.5 × 14 in)", 8.5, 14.0),
    ("Tabloid (11 × 17 in)", 11.0, 17.0),
    ("A4 (210 × 297 mm)", 8.267, 11.692),
    ("A3 (297 × 420 mm)", 11.692, 16.535),
]

CUT_LINE_STYLES = ["None", "Dashed", "Dotted", "Solid"]


# === Data Model ===

@dataclass
class PageSettings:
    """Configurable page layout settings."""
    paper_size_index: int = 0          # Index into PAPER_SIZES
    margin_inches: float = 0.25        # Margin on all sides, in inches
    cut_line_style: int = 1            # Index into CUT_LINE_STYLES (default: Dashed)
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
    scale_step: int = 0  # Per-image scaling: each step ≈ 20% size change

    def __getattr__(self, name):
        # Backward compat: old pickled instances lack scale_step
        if name == 'scale_step':
            return 0
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


# === Tiler: log-scale sizing, row quantization, row packing, scale-to-fit ===

class Tiler:
    """Layout engine implementing the row-based tiling algorithm."""

    def __init__(self, settings: PageSettings | None = None):
        s = settings or PageSettings()
        self.pw = s.printable_width
        self.ph = s.printable_height
        self.gap = s.cut_gap
        self.margin = s.margin

    def layout(self, images: list[StickerImage]) -> LayoutResult:
        """Run the full layout pipeline on the given images."""
        if not images:
            return LayoutResult()

        # Step 1: Log-scale sizing — dampen resolution differences
        raw_ideal = []
        for img in images:
            iw = math.log2(img.pixel_width + 1)
            ih = math.log2(img.pixel_height + 1)
            raw_ideal.append((iw, ih))

        # Step 2: Quantize heights into row bins (from UNSCALED ideals for stability)
        n_bins = self._bin_count(len(images))
        bins = self._compute_bins([ih for _, ih in raw_ideal], n_bins)

        # Step 3: Apply per-image scale_step, then snap to nearest bin.
        # Each step ≈ 1.2x size change. This naturally bumps images into
        # adjacent bins when the scaling crosses a bin boundary.
        groups: dict[float, list[tuple[float, int]]] = {}
        for i, (iw, ih) in enumerate(raw_ideal):
            step = getattr(images[i], 'scale_step', 0)
            if step != 0:
                factor = 1.2 ** step
                iw *= factor
                ih *= factor
            best_bin = min(bins, key=lambda b: abs(b - ih))
            scaled_w = iw * (best_bin / ih) if ih > 0 else iw
            groups.setdefault(best_bin, []).append((scaled_w, i))

        # Step 4: Pack rows with scale-to-fit via binary search
        return self._scale_to_fit(groups)

    def _bin_count(self, n: int) -> int:
        """Choose number of height bins based on image count."""
        if n <= 2:
            return 1
        if n <= 5:
            return 2
        if n <= 10:
            return 3
        if n <= 20:
            return 4
        return 5

    def _compute_bins(self, heights: list[float], n_bins: int) -> list[float]:
        """Split sorted heights into n_bins groups, return median of each."""
        heights = sorted(heights)
        if n_bins >= len(heights):
            result = sorted(set(heights))
            return result if result else [1.0]

        group_size = len(heights) / n_bins
        bins = []
        for i in range(n_bins):
            start = int(i * group_size)
            end = int((i + 1) * group_size)
            group = heights[start:end]
            if group:
                bins.append(group[len(group) // 2])

        result = sorted(set(bins))
        return result if result else [heights[len(heights) // 2]]

    def _scale_to_fit(self, groups: dict) -> LayoutResult:
        """Binary search for the largest scale factor that fits the page."""
        max_bin = max(groups.keys())
        scale_hi = (self.ph * 0.6) / max_bin if max_bin > 0 else 100.0
        scale_lo = 0.1

        best = None
        for _ in range(40):
            mid = (scale_hi + scale_lo) / 2
            result = self._try_pack(groups, mid)
            if result is not None:
                best = result
                scale_lo = mid
            else:
                scale_hi = mid

        if best is None:
            best = self._try_pack(groups, scale_lo) or LayoutResult()
        return best

    def _try_pack(self, groups: dict, scale: float) -> LayoutResult | None:
        """Attempt to pack all images at the given scale. Returns None if overflow."""
        rows = []

        # Pack each bin's images into rows, largest bins first
        for bin_h in sorted(groups.keys(), reverse=True):
            row_h = max(1, int(bin_h * scale))

            current: list[tuple[int, int, int]] = []  # (x, width, image_index)
            cx = 0

            for ideal_w, idx in groups[bin_h]:
                w = max(1, int(ideal_w * scale))
                if w > self.pw:
                    w = self.pw

                needed = cx + (self.gap if current else 0) + w
                if needed > self.pw and current:
                    rows.append((row_h, current))
                    current = []
                    cx = 0

                if current:
                    cx += self.gap
                current.append((cx, w, idx))
                cx += w

            if current:
                rows.append((row_h, current))

        # Check vertical fit
        total = sum(h for h, _ in rows) + self.gap * max(0, len(rows) - 1)
        if total > self.ph:
            return None

        # Build final layout with absolute positions
        layout_rows = []
        y = self.margin
        for row_h, items in rows:
            placements = [
                PlacedImage(image_index=idx, x=self.margin + x, y=y, width=w, height=row_h)
                for x, w, idx in items
            ]
            layout_rows.append(LayoutRow(y=y, height=row_h, placements=placements))
            y += row_h + self.gap

        return LayoutResult(rows=layout_rows)


# === PageWidget: WYSIWYG page view ===

class PageWidget(QWidget):
    """Draws the US Letter page with tiled sticker images and cut lines."""

    images_changed = Signal()
    selection_changed = Signal()
    zoom_changed = Signal()

    def __init__(self, project: StickerProject, parent=None):
        super().__init__(parent)
        self.project = project
        self._pixmap_cache: dict[int, QPixmap] = {}
        self.selected_index: int | None = None
        self._zoom: float = 1.0
        self._pan = QPointF(0, 0)
        self._pan_start: QPointF | None = None
        self._pan_start_offset = QPointF(0, 0)
        self.setMinimumSize(400, 500)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def invalidate_cache(self):
        self._pixmap_cache.clear()

    def _get_pixmap(self, index: int) -> QPixmap:
        if index not in self._pixmap_cache:
            qimg = QImage()
            qimg.loadFromData(self.project.images[index].png_data)
            self._pixmap_cache[index] = QPixmap.fromImage(qimg)
        return self._pixmap_cache[index]

    def _page_w(self) -> int:
        return self.project.settings.page_width

    def _page_h(self) -> int:
        return self.project.settings.page_height

    def _base_scale(self) -> float:
        """Scale factor to fit the page into the widget at zoom=1."""
        padding = 20
        w = max(1, self.width() - 2 * padding)
        h = max(1, self.height() - 2 * padding)
        return min(w / self._page_w(), h / self._page_h())

    def page_scale(self) -> float:
        """Compute the scale factor from 300 DPI page coords to screen pixels."""
        return self._base_scale() * self._zoom

    def _page_origin(self) -> tuple[float, float]:
        """Top-left corner of the page on screen, centered in widget with pan offset."""
        scale = self.page_scale()
        pw = self._page_w() * scale
        ph = self._page_h() * scale
        return (self.width() - pw) / 2 + self._pan.x(), (self.height() - ph) / 2 + self._pan.y()

    def reset_view(self):
        """Reset zoom and pan to defaults."""
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.zoom_changed.emit()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        s = self.project.settings
        page_w = s.page_width
        page_h = s.page_height

        # Widget background
        painter.fillRect(self.rect(), QColor(200, 200, 200))

        scale = self.page_scale()
        ox, oy = self._page_origin()

        # Page shadow
        painter.fillRect(
            QRectF(ox + 3, oy + 3, page_w * scale, page_h * scale),
            QColor(150, 150, 150),
        )
        # White page
        painter.fillRect(
            QRectF(ox, oy, page_w * scale, page_h * scale),
            QColor(255, 255, 255),
        )
        # Margin guides (faint dashed rectangle)
        painter.setPen(QPen(QColor(230, 230, 230), 1, Qt.PenStyle.DashLine))
        m = s.margin * scale
        painter.drawRect(QRectF(ox + m, oy + m,
                                s.printable_width * scale, s.printable_height * scale))

        layout = self.project.layout
        if layout and layout.rows:
            self._paint_images(painter, layout, scale, ox, oy)
            self._paint_cut_lines(painter, layout, scale, ox, oy)
            self._paint_selection(painter, layout, scale, ox, oy)

        painter.end()

    def _paint_images(self, painter, layout, scale, ox, oy):
        for placed in layout.placements:
            pix = self._get_pixmap(placed.image_index)
            dest = QRectF(ox + placed.x * scale, oy + placed.y * scale,
                          placed.width * scale, placed.height * scale)
            painter.drawPixmap(dest.toRect(), pix)

    def _paint_cut_lines(self, painter, layout, scale, ox, oy):
        s = self.project.settings
        style_name = CUT_LINE_STYLES[s.cut_line_style]
        if style_name == "None":
            return
        pen_style = {
            "Dashed": Qt.PenStyle.DashLine,
            "Dotted": Qt.PenStyle.DotLine,
            "Solid": Qt.PenStyle.SolidLine,
        }[style_name]
        painter.setPen(QPen(QColor(180, 180, 180), 1, pen_style))

        margin = s.margin
        left = ox + margin * scale
        right = ox + (s.page_width - margin) * scale

        # Horizontal cut lines between rows (span full printable width)
        for i in range(1, len(layout.rows)):
            prev = layout.rows[i - 1]
            curr = layout.rows[i]
            y_mid = (prev.y + prev.height + curr.y) / 2
            ys = oy + y_mid * scale
            painter.drawLine(QPointF(left, ys), QPointF(right, ys))

        # Vertical cut lines between images within each row
        for row in layout.rows:
            sorted_p = sorted(row.placements, key=lambda p: p.x)
            for j in range(len(sorted_p) - 1):
                a, b = sorted_p[j], sorted_p[j + 1]
                x_mid = (a.x + a.width + b.x) / 2
                xs = ox + x_mid * scale
                top = oy + row.y * scale
                bot = oy + (row.y + row.height) * scale
                painter.drawLine(QPointF(xs, top), QPointF(xs, bot))

    def _paint_selection(self, painter, layout, scale, ox, oy):
        """Draw a highlight border around the selected image."""
        if self.selected_index is None:
            return
        for placed in layout.placements:
            if placed.image_index == self.selected_index:
                rect = QRectF(ox + placed.x * scale, oy + placed.y * scale,
                              placed.width * scale, placed.height * scale)
                pen = QPen(QColor(0, 120, 215), 3)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(rect)
                break

    def _hit_test(self, pos) -> int | None:
        """Return the image_index of the image under the given widget position, or None."""
        layout = self.project.layout
        if not layout or not layout.rows:
            return None
        scale = self.page_scale()
        ox, oy = self._page_origin()
        # Check in reverse draw order so topmost image wins
        for placed in reversed(layout.placements):
            rect = QRectF(ox + placed.x * scale, oy + placed.y * scale,
                          placed.width * scale, placed.height * scale)
            if rect.contains(QPointF(pos.x(), pos.y())):
                return placed.image_index
        return None

    def wheelEvent(self, event):
        """Cmd+scroll or Shift+scroll to zoom toward cursor."""
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier or mods & Qt.KeyboardModifier.ShiftModifier:
            old_scale = self.page_scale()
            delta = event.angleDelta().y()
            if delta == 0:
                event.ignore()
                return
            factor = 1.15 if delta > 0 else 1 / 1.15
            new_zoom = max(0.25, min(self._zoom * factor, 8.0))

            if new_zoom != self._zoom:
                mouse_pos = event.position()
                ox, oy = self._page_origin()
                # Point on the page under the cursor
                rel_x = mouse_pos.x() - ox
                rel_y = mouse_pos.y() - oy

                self._zoom = new_zoom
                new_scale = self.page_scale()
                scale_ratio = new_scale / old_scale

                # Adjust pan so the point under cursor stays fixed
                new_ox = mouse_pos.x() - rel_x * scale_ratio
                new_oy = mouse_pos.y() - rel_y * scale_ratio
                base_ox = (self.width() - self._page_w() * new_scale) / 2
                base_oy = (self.height() - self._page_h() * new_scale) / 2
                self._pan = QPointF(new_ox - base_ox, new_oy - base_oy)

                self.zoom_changed.emit()
                self.update()
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = event.position()
            self._pan_start_offset = QPointF(self._pan)
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            hit = self._hit_test(event.position())
            old = self.selected_index
            self.selected_index = hit
            if old != hit:
                self.selection_changed.emit()
                self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._pan = self._pan_start_offset + delta
            self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_start is not None:
            self._pan_start = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    # --- Drag and drop ---

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        mime = event.mimeData()
        added = False
        if mime.hasUrls():
            for url in mime.urls():
                path = url.toLocalFile()
                if path:
                    if self._load_file(path):
                        added = True
                elif url.scheme() in ("http", "https"):
                    if self._load_url(url.toString()):
                        added = True
        elif mime.hasImage():
            qimg = QImage(mime.imageData())
            if not qimg.isNull():
                self._add_qimage(qimg)
                added = True
        elif mime.hasText():
            text = mime.text().strip()
            if text.startswith(("http://", "https://")):
                if self._load_url(text):
                    added = True
        if added:
            self.images_changed.emit()
        event.acceptProposedAction()

    def dragEnterEvent(self, event):
        mime = event.mimeData()
        if mime.hasImage() or mime.hasUrls() or mime.hasText():
            event.acceptProposedAction()

    def _load_file(self, path: str) -> bool:
        try:
            img = Image.open(path)
            img.load()
            self._add_pil(img)
            return True
        except Exception:
            return False

    @staticmethod
    def _upgrade_url(url: str) -> str:
        """Attempt to convert a thumbnail/proxy URL to a higher-resolution original.

        Supported sites:
        - DuckDuckGo image proxy: extracts the original URL from the 'u' query param.
        - Wikipedia/Wikimedia thumbnails: strips /thumb/ and the size suffix to get
          the full-resolution original.

        Returns the upgraded URL, or the original URL unchanged if no pattern matches.
        """
        parsed = urllib.parse.urlparse(url)

        # DuckDuckGo image proxy: external-content.duckduckgo.com/iu/?u=<original>
        if (parsed.hostname and parsed.hostname.endswith("duckduckgo.com")
                and parsed.path == "/iu/"):
            params = urllib.parse.parse_qs(parsed.query)
            if "u" in params:
                return params["u"][0]

        # Wikipedia / Wikimedia thumbnails:
        #   .../thumb/{h1}/{h2}/{Filename}/{width}px-{Filename}
        # becomes:
        #   .../{h1}/{h2}/{Filename}
        if (parsed.hostname and parsed.hostname.endswith("wikimedia.org")
                and "/thumb/" in parsed.path):
            path = parsed.path.replace("/thumb/", "/", 1)
            path = path.rsplit("/", 1)[0]  # drop the sized filename suffix
            return urllib.parse.urlunparse(parsed._replace(path=path))

        return url

    def _load_url(self, url: str) -> bool:
        """Download an image from a URL and add it to the project.

        Tries to upgrade thumbnail/proxy URLs to full-resolution originals first
        (DuckDuckGo, Wikipedia). Falls back to the original URL on failure.
        """
        upgraded = self._upgrade_url(url)
        for attempt_url in ([upgraded, url] if upgraded != url else [url]):
            try:
                req = urllib.request.Request(
                    attempt_url, headers={"User-Agent": "StickerSheet/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read(10 * 1024 * 1024)  # 10 MB limit
                img = Image.open(io.BytesIO(data))
                img.load()
                self._add_pil(img)
                return True
            except Exception:
                continue
        return False

    def _qimage_to_sticker(self, qimage: QImage) -> StickerImage:
        """Convert QImage to a StickerImage via Pillow normalization."""
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        qimage.save(buf, "PNG")
        buf.close()
        img = Image.open(io.BytesIO(bytes(ba.data())))
        return self._pil_to_sticker(img)

    def _pil_to_sticker(self, img: Image.Image) -> StickerImage:
        """Normalize to RGBA PNG and return a StickerImage."""
        img = img.convert("RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return StickerImage(png_data=buf.getvalue(),
                            pixel_width=img.width,
                            pixel_height=img.height)

    def _add_qimage(self, qimage: QImage):
        """Convert QImage to PNG bytes and store in project."""
        self.project.images.append(self._qimage_to_sticker(qimage))

    def _add_pil(self, img: Image.Image):
        """Normalize to RGBA PNG and store in project."""
        self.project.images.append(self._pil_to_sticker(img))


# === Undo Commands (Phase 6) ===

class PasteImageCommand(QUndoCommand):
    """Undoable command: paste one image."""

    def __init__(self, window: "MainWindow", sticker_image: StickerImage):
        super().__init__("Paste Image")
        self._window = window
        self._image = sticker_image

    def redo(self):
        self._window.project.images.append(self._image)
        self._window._retile()

    def undo(self):
        self._window.project.images.pop()
        self._window.page_widget.selected_index = None
        self._window._retile()


class ScaleImageCommand(QUndoCommand):
    """Undoable command: change an image's scale_step."""

    def __init__(self, window: "MainWindow", index: int, old_step: int, new_step: int):
        super().__init__("Scale Image")
        self._window = window
        self._index = index
        self._old_step = old_step
        self._new_step = new_step

    def redo(self):
        self._window.project.images[self._index].scale_step = self._new_step
        self._window._retile()

    def undo(self):
        self._window.project.images[self._index].scale_step = self._old_step
        self._window._retile()


class DeleteImageCommand(QUndoCommand):
    """Undoable command: delete one image by index."""

    def __init__(self, window: "MainWindow", index: int):
        super().__init__("Delete Image")
        self._window = window
        self._index = index
        self._image = window.project.images[index]

    def redo(self):
        self._window.project.images.pop(self._index)
        self._window.page_widget.selected_index = None
        self._window._retile()

    def undo(self):
        self._window.project.images.insert(self._index, self._image)
        self._window._retile()


# === Settings Dialog ===

class SettingsDialog(QDialog):
    """Dialog for configuring page layout settings."""

    def __init__(self, settings: PageSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(350)

        layout = QVBoxLayout(self)

        # --- Page group ---
        page_group = QGroupBox("Page")
        page_form = QFormLayout(page_group)

        self._paper_combo = QComboBox()
        for name, _, _ in PAPER_SIZES:
            self._paper_combo.addItem(name)
        self._paper_combo.setCurrentIndex(settings.paper_size_index)
        page_form.addRow("Paper size:", self._paper_combo)

        self._margin_spin = QDoubleSpinBox()
        self._margin_spin.setRange(0.0, 2.0)
        self._margin_spin.setSingleStep(0.125)
        self._margin_spin.setDecimals(3)
        self._margin_spin.setSuffix(" in")
        self._margin_spin.setValue(settings.margin_inches)
        page_form.addRow("Margins:", self._margin_spin)

        layout.addWidget(page_group)

        # --- Cutting group ---
        cut_group = QGroupBox("Cutting")
        cut_form = QFormLayout(cut_group)

        self._street_spin = QDoubleSpinBox()
        self._street_spin.setRange(0.0, 1.0)
        self._street_spin.setSingleStep(0.0625)
        self._street_spin.setDecimals(4)
        self._street_spin.setSuffix(" in")
        self._street_spin.setValue(settings.street_width_inches)
        cut_form.addRow("Street width:", self._street_spin)

        self._cut_style_combo = QComboBox()
        for style in CUT_LINE_STYLES:
            self._cut_style_combo.addItem(style)
        self._cut_style_combo.setCurrentIndex(settings.cut_line_style)
        cut_form.addRow("Cut line style:", self._cut_style_combo)

        layout.addWidget(cut_group)

        # --- Buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result_settings(self) -> PageSettings:
        """Return a PageSettings reflecting the dialog's current values."""
        return PageSettings(
            paper_size_index=self._paper_combo.currentIndex(),
            margin_inches=self._margin_spin.value(),
            cut_line_style=self._cut_style_combo.currentIndex(),
            street_width_inches=self._street_spin.value(),
        )


# === MainWindow ===

class MainWindow(QMainWindow):
    """Top-level window: menu bar, page widget, status bar."""

    def __init__(self):
        super().__init__()
        self.project = StickerProject()
        self.tiler = Tiler(self.project.settings)
        self._file_path: str | None = None
        self._dirty = False
        self._undo_stack = QUndoStack(self)
        self._undo_stack.cleanChanged.connect(self._on_clean_changed)

        self.setWindowTitle("Sticker Sheet Maker")
        self.resize(800, 1000)

        self.page_widget = PageWidget(self.project)
        self.setCentralWidget(self.page_widget)
        self.page_widget.images_changed.connect(self._on_drop_images)
        self.page_widget.selection_changed.connect(self._on_selection_changed)
        self.page_widget.zoom_changed.connect(self._update_status)
        self.page_widget.customContextMenuRequested.connect(self._show_context_menu)

        self._build_menus()
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._update_title()
        self._update_status()

    def _build_menus(self):
        mb = self.menuBar()

        # File menu
        file_menu = mb.addMenu("&File")

        act = QAction("&New", self)
        act.setShortcut(QKeySequence.StandardKey.New)
        act.triggered.connect(self._new_project)
        file_menu.addAction(act)

        act = QAction("&Open...", self)
        act.setShortcut(QKeySequence.StandardKey.Open)
        act.triggered.connect(self._open)
        file_menu.addAction(act)

        file_menu.addSeparator()

        act = QAction("&Save", self)
        act.setShortcut(QKeySequence.StandardKey.Save)
        act.triggered.connect(self._save)
        file_menu.addAction(act)

        act = QAction("Save &As...", self)
        act.setShortcut(QKeySequence.StandardKey.SaveAs)
        act.triggered.connect(self._save_as)
        file_menu.addAction(act)

        file_menu.addSeparator()

        act = QAction("&Print...", self)
        act.setShortcut(QKeySequence.StandardKey.Print)
        act.triggered.connect(self._print)
        file_menu.addAction(act)

        file_menu.addSeparator()

        act = QAction("&Quit", self)
        act.setShortcut(QKeySequence.StandardKey.Quit)
        act.triggered.connect(self.close)
        file_menu.addAction(act)

        # Edit menu
        edit_menu = mb.addMenu("&Edit")

        self._undo_action = self._undo_stack.createUndoAction(self, "&Undo")
        self._undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        edit_menu.addAction(self._undo_action)

        self._redo_action = self._undo_stack.createRedoAction(self, "&Redo")
        self._redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        edit_menu.addAction(self._redo_action)

        edit_menu.addSeparator()

        act = QAction("&Paste", self)
        act.setShortcut(QKeySequence.StandardKey.Paste)
        act.triggered.connect(self._paste)
        edit_menu.addAction(act)

        act = QAction("&Copy", self)
        act.setShortcut(QKeySequence.StandardKey.Copy)
        act.triggered.connect(self._copy_selected)
        edit_menu.addAction(act)

        self._delete_action = QAction("&Delete Selected", self)
        self._delete_action.setShortcut(QKeySequence.StandardKey.Delete)
        self._delete_action.triggered.connect(self._delete_selected)
        self._delete_action.setEnabled(False)
        edit_menu.addAction(self._delete_action)

        edit_menu.addSeparator()

        act = QAction("Clear &All", self)
        act.triggered.connect(self._clear_all)
        edit_menu.addAction(act)

        edit_menu.addSeparator()

        act = QAction("Se&ttings...", self)
        act.triggered.connect(self._show_settings)
        edit_menu.addAction(act)

    # --- Dirty state ---

    def _mark_dirty(self):
        self._dirty = True
        self._update_title()

    def _mark_clean(self):
        self._dirty = False
        self._update_title()

    def _on_clean_changed(self, clean: bool):
        self._dirty = not clean
        self._update_title()

    def _update_title(self):
        name = self._file_path.rsplit("/", 1)[-1] if self._file_path else "Untitled"
        dirty = " *" if self._dirty else ""
        self.setWindowTitle(f"{name}{dirty} — Sticker Sheet Maker")

    def _check_unsaved(self) -> bool:
        """Return True if it's safe to proceed (saved or discarded). False = cancelled."""
        if not self._dirty:
            return True
        reply = QMessageBox.question(
            self, "Unsaved Changes",
            "You have unsaved changes. Do you want to save before continuing?",
            QMessageBox.StandardButton.Save |
            QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self._save()
        return reply == QMessageBox.StandardButton.Discard

    # --- Selection ---

    def _on_selection_changed(self):
        has_sel = self.page_widget.selected_index is not None
        self._delete_action.setEnabled(has_sel)
        self._update_status()

    # --- Actions ---

    def _paste(self):
        """Paste image from clipboard via undo stack."""
        cb = QApplication.clipboard()
        mime = cb.mimeData()

        sticker = None
        if mime.hasImage():
            qimg = cb.image()
            if not qimg.isNull():
                sticker = self.page_widget._qimage_to_sticker(qimg)

        if sticker is None:
            for fmt in mime.formats():
                if "image" in fmt.lower():
                    data = mime.data(fmt)
                    if data:
                        try:
                            img = Image.open(io.BytesIO(bytes(data)))
                            sticker = self.page_widget._pil_to_sticker(img)
                            break
                        except Exception:
                            continue

        if sticker:
            self._undo_stack.push(PasteImageCommand(self, sticker))

    def _on_drop_images(self):
        """Handle images added via drag-and-drop (already in project.images)."""
        self._undo_stack.clear()  # can't undo drops; reset baseline
        self._retile()
        self._mark_dirty()

    def _delete_selected(self):
        idx = self.page_widget.selected_index
        if idx is not None and 0 <= idx < len(self.project.images):
            self._undo_stack.push(DeleteImageCommand(self, idx))

    def _copy_selected(self):
        """Copy the selected image back to clipboard."""
        idx = self.page_widget.selected_index
        if idx is None or idx >= len(self.project.images):
            return
        data = self.project.images[idx].png_data
        qimg = QImage()
        qimg.loadFromData(data)
        if not qimg.isNull():
            QApplication.clipboard().setImage(qimg)

    def _scale_selected(self, delta: int):
        """Adjust the selected image's scale_step by delta (+1 or -1)."""
        idx = self.page_widget.selected_index
        if idx is None or idx >= len(self.project.images):
            return
        img = self.project.images[idx]
        old_step = getattr(img, 'scale_step', 0)
        new_step = old_step + delta
        self._undo_stack.push(ScaleImageCommand(self, idx, old_step, new_step))

    def _show_context_menu(self, pos):
        idx = self.page_widget._hit_test(pos)
        if idx is None:
            return
        # Select the right-clicked image
        self.page_widget.selected_index = idx
        self.page_widget.selection_changed.emit()
        self.page_widget.update()

        menu = QMenu(self)
        copy_act = menu.addAction("Copy")
        menu.addSeparator()
        scale_up_act = menu.addAction("Scale Up  ]")
        scale_down_act = menu.addAction("Scale Down  [")
        img = self.project.images[idx]
        step = getattr(img, 'scale_step', 0)
        if step != 0:
            reset_scale_act = menu.addAction("Reset Scale")
        else:
            reset_scale_act = None
        menu.addSeparator()
        delete_act = menu.addAction("Delete")
        chosen = menu.exec(self.page_widget.mapToGlobal(pos))
        if chosen == copy_act:
            self._copy_selected()
        elif chosen == scale_up_act:
            self._scale_selected(1)
        elif chosen == scale_down_act:
            self._scale_selected(-1)
        elif chosen == reset_scale_act:
            self._scale_selected(-step)
        elif chosen == delete_act:
            self._delete_selected()

    def _new_project(self):
        if not self._check_unsaved():
            return
        self.project.images.clear()
        self.project.layout = LayoutResult()
        self.project.settings = PageSettings()
        self.tiler = Tiler(self.project.settings)
        self._file_path = None
        self._undo_stack.clear()
        self._undo_stack.setClean()
        self.page_widget.selected_index = None
        self.page_widget.invalidate_cache()
        self.page_widget.update()
        self._mark_clean()
        self._update_status()

    def _clear_all(self):
        if not self.project.images:
            return
        if not self._check_unsaved():
            return
        self.project.images.clear()
        self.project.layout = LayoutResult()
        self._undo_stack.clear()
        self._undo_stack.setClean()
        self.page_widget.selected_index = None
        self.page_widget.invalidate_cache()
        self.page_widget.update()
        self._mark_dirty()
        self._update_status()

    # --- Settings ---

    def _show_settings(self):
        dlg = SettingsDialog(self.project.settings, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.project.settings = dlg.result_settings()
            self._apply_settings()
            self._mark_dirty()

    def _apply_settings(self):
        """Rebuild tiler from current settings and re-layout."""
        self.tiler = Tiler(self.project.settings)
        self._retile()

    # --- Save / Load (Phase 5) ---

    def _save(self) -> bool:
        if self._file_path:
            return self._write_file(self._file_path)
        return self._save_as()

    def _save_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Save Sticker Sheet", "", STICKER_FILTER)
        if not path:
            return False
        if not path.endswith(".sticker"):
            path += ".sticker"
        return self._write_file(path)

    def _write_file(self, path: str) -> bool:
        try:
            with open(path, "wb") as f:
                pickle.dump(self.project, f)
            self._file_path = path
            self._undo_stack.setClean()
            self._mark_clean()
            return True
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save file:\n{e}")
            return False

    def _open(self):
        if not self._check_unsaved():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open Sticker Sheet", "", STICKER_FILTER)
        if not path:
            return
        self.open_file(path)

    def open_file(self, path: str):
        """Open a .sticker file by path. Used by File>Open, argv, and macOS QFileOpenEvent."""
        try:
            with open(path, "rb") as f:
                proj = pickle.load(f)  # noqa: S301
            if not isinstance(proj, StickerProject):
                raise TypeError("Not a valid sticker project")
        except Exception as e:
            QMessageBox.critical(self, "Open Error", f"Could not open file:\n{e}")
            return
        self.project.images = proj.images
        self.project.layout = proj.layout
        self.project.settings = proj.settings
        self.tiler = Tiler(self.project.settings)
        self._file_path = path
        self._undo_stack.clear()
        self._undo_stack.setClean()
        self.page_widget.selected_index = None
        self.page_widget.invalidate_cache()
        self._retile()
        self._mark_clean()

    # --- Print (Phase 4) ---

    def _print(self):
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setPageSize(printer.pageLayout().pageSize())  # default to system
        dialog = QPrintDialog(printer, self)
        if dialog.exec() != QPrintDialog.DialogCode.Accepted:
            return

        painter = QPainter()
        if not painter.begin(printer):
            return

        # Render at native printer resolution with the same layout
        layout = self.project.layout
        if not layout or not layout.rows:
            painter.end()
            return

        # Scale from our 300 DPI coordinate system to printer DPI
        printer_dpi_x = printer.logicalDpiX()
        printer_dpi_y = printer.logicalDpiY()
        sx = printer_dpi_x / DPI
        sy = printer_dpi_y / DPI
        painter.scale(sx, sy)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        for placed in layout.placements:
            qimg = QImage()
            qimg.loadFromData(self.project.images[placed.image_index].png_data)
            dest = QRectF(placed.x, placed.y, placed.width, placed.height)
            painter.drawImage(dest, qimg)

        painter.end()

    # --- Retile & status ---

    def _retile(self):
        """Re-run the layout algorithm and repaint."""
        self.project.layout = self.tiler.layout(self.project.images)
        self.page_widget.invalidate_cache()
        self.page_widget.update()
        self._update_status()

    def _update_status(self):
        n = len(self.project.images)
        zoom = int(self.page_widget._zoom * 100)
        sel = self.page_widget.selected_index
        if n == 0:
            self._status.showMessage(
                f"No images \u2014 Paste (Ctrl+V) or drag images to add | Zoom: {zoom}%")
        else:
            sel_text = ""
            if sel is not None and sel < n:
                step = getattr(self.project.images[sel], 'scale_step', 0)
                scale_text = f" (scale: {'+' if step > 0 else ''}{step})" if step != 0 else ""
                sel_text = f" | Selected: #{sel + 1}{scale_text}"
            self._status.showMessage(
                f"{n} image{'s' if n != 1 else ''}{sel_text} | Zoom: {zoom}%")

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()

        # Cmd+0 / Ctrl+0: reset zoom and pan
        if key == Qt.Key.Key_0 and mods & Qt.KeyboardModifier.ControlModifier:
            self.page_widget.reset_view()
            event.accept()
            return

        # ] or +/= : scale up selected image
        if key in (Qt.Key.Key_BracketRight, Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            if not mods & Qt.KeyboardModifier.ControlModifier:
                self._scale_selected(1)
                event.accept()
                return

        # [ or - : scale down selected image
        if key in (Qt.Key.Key_BracketLeft, Qt.Key.Key_Minus):
            if not mods & Qt.KeyboardModifier.ControlModifier:
                self._scale_selected(-1)
                event.accept()
                return

        super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_status()

    def closeEvent(self, event):
        if self._check_unsaved():
            event.accept()
        else:
            event.ignore()


# === StickerApp: custom QApplication for macOS file open events ===

class StickerApp(QApplication):
    """QApplication subclass that handles macOS QFileOpenEvent."""

    file_open_requested = Signal(str)

    def event(self, event):
        if event.type() == QEvent.Type.FileOpen:
            self.file_open_requested.emit(event.file())
            return True
        return super().event(event)


# === Entry Point ===

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
