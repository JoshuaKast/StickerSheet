#!/usr/bin/env python3
"""Sticker Sheet Maker - Tile pasted images onto a US Letter page for sticker printing."""

import io
import math
import sys
from dataclasses import dataclass, field

from PIL import Image
from PySide6.QtCore import Qt, QRectF, QPointF, QByteArray, QBuffer, QIODevice, Signal
from PySide6.QtGui import (
    QAction, QImage, QPixmap, QPainter, QPen, QColor, QKeySequence,
)
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QStatusBar


# === Constants (all layout math in 300 DPI pixel space) ===
DPI = 300
PAGE_WIDTH = int(8.5 * DPI)    # 2550
PAGE_HEIGHT = int(11.0 * DPI)  # 3300
MARGIN = int(0.25 * DPI)       # 75
CUT_GAP = 4                    # ~4px at 300 DPI between images

PRINTABLE_WIDTH = PAGE_WIDTH - 2 * MARGIN    # 2400
PRINTABLE_HEIGHT = PAGE_HEIGHT - 2 * MARGIN  # 3150


# === Data Model ===

@dataclass
class StickerImage:
    """A single image stored as normalized PNG bytes."""
    png_data: bytes
    pixel_width: int
    pixel_height: int


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


# === Tiler: log-scale sizing, row quantization, row packing, scale-to-fit ===

class Tiler:
    """Layout engine implementing the row-based tiling algorithm."""

    def __init__(self):
        self.pw = PRINTABLE_WIDTH
        self.ph = PRINTABLE_HEIGHT
        self.gap = CUT_GAP

    def layout(self, images: list[StickerImage]) -> LayoutResult:
        """Run the full layout pipeline on the given images."""
        if not images:
            return LayoutResult()

        # Step 1: Log-scale sizing — dampen resolution differences
        ideal = []
        for img in images:
            iw = math.log2(img.pixel_width + 1)
            ih = math.log2(img.pixel_height + 1)
            ideal.append((iw, ih))

        # Step 2: Quantize heights into row bins
        n_bins = self._bin_count(len(images))
        bins = self._compute_bins([ih for _, ih in ideal], n_bins)

        # Snap each image to nearest bin height, scale width proportionally
        groups: dict[float, list[tuple[float, int]]] = {}
        for i, (iw, ih) in enumerate(ideal):
            best_bin = min(bins, key=lambda b: abs(b - ih))
            scaled_w = iw * (best_bin / ih) if ih > 0 else iw
            groups.setdefault(best_bin, []).append((scaled_w, i))

        # Step 3: Pack rows with scale-to-fit via binary search
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
        y = MARGIN
        for row_h, items in rows:
            placements = [
                PlacedImage(image_index=idx, x=MARGIN + x, y=y, width=w, height=row_h)
                for x, w, idx in items
            ]
            layout_rows.append(LayoutRow(y=y, height=row_h, placements=placements))
            y += row_h + self.gap

        return LayoutResult(rows=layout_rows)


# === PageWidget: WYSIWYG page view ===

class PageWidget(QWidget):
    """Draws the US Letter page with tiled sticker images and cut lines."""

    images_changed = Signal()

    def __init__(self, project: StickerProject, parent=None):
        super().__init__(parent)
        self.project = project
        self._pixmap_cache: dict[int, QPixmap] = {}
        self.setMinimumSize(400, 500)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def invalidate_cache(self):
        self._pixmap_cache.clear()

    def _get_pixmap(self, index: int) -> QPixmap:
        if index not in self._pixmap_cache:
            qimg = QImage()
            qimg.loadFromData(self.project.images[index].png_data)
            self._pixmap_cache[index] = QPixmap.fromImage(qimg)
        return self._pixmap_cache[index]

    def page_scale(self) -> float:
        """Compute the scale factor from 300 DPI page coords to screen pixels."""
        padding = 20
        w = max(1, self.width() - 2 * padding)
        h = max(1, self.height() - 2 * padding)
        return min(w / PAGE_WIDTH, h / PAGE_HEIGHT)

    def _page_origin(self) -> tuple[float, float]:
        """Top-left corner of the page on screen, centered in widget."""
        scale = self.page_scale()
        pw = PAGE_WIDTH * scale
        ph = PAGE_HEIGHT * scale
        return (self.width() - pw) / 2, (self.height() - ph) / 2

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Widget background
        painter.fillRect(self.rect(), QColor(200, 200, 200))

        scale = self.page_scale()
        ox, oy = self._page_origin()

        # Page shadow
        painter.fillRect(
            QRectF(ox + 3, oy + 3, PAGE_WIDTH * scale, PAGE_HEIGHT * scale),
            QColor(150, 150, 150),
        )
        # White page
        painter.fillRect(
            QRectF(ox, oy, PAGE_WIDTH * scale, PAGE_HEIGHT * scale),
            QColor(255, 255, 255),
        )
        # Margin guides (faint dashed rectangle)
        painter.setPen(QPen(QColor(230, 230, 230), 1, Qt.PenStyle.DashLine))
        m = MARGIN * scale
        painter.drawRect(QRectF(ox + m, oy + m,
                                PRINTABLE_WIDTH * scale, PRINTABLE_HEIGHT * scale))

        layout = self.project.layout
        if layout and layout.rows:
            self._paint_images(painter, layout, scale, ox, oy)
            self._paint_cut_lines(painter, layout, scale, ox, oy)

        painter.end()

    def _paint_images(self, painter, layout, scale, ox, oy):
        for placed in layout.placements:
            pix = self._get_pixmap(placed.image_index)
            dest = QRectF(ox + placed.x * scale, oy + placed.y * scale,
                          placed.width * scale, placed.height * scale)
            painter.drawPixmap(dest.toRect(), pix)

    def _paint_cut_lines(self, painter, layout, scale, ox, oy):
        painter.setPen(QPen(QColor(180, 180, 180), 1, Qt.PenStyle.DashLine))

        left = ox + MARGIN * scale
        right = ox + (PAGE_WIDTH - MARGIN) * scale

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

    # --- Drag and drop ---

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage() or event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        mime = event.mimeData()
        added = False
        if mime.hasUrls():
            for url in mime.urls():
                path = url.toLocalFile()
                if path and self._load_file(path):
                    added = True
        elif mime.hasImage():
            qimg = QImage(mime.imageData())
            if not qimg.isNull():
                self._add_qimage(qimg)
                added = True
        if added:
            self.images_changed.emit()
        event.acceptProposedAction()

    def _load_file(self, path: str) -> bool:
        try:
            img = Image.open(path)
            img.load()
            self._add_pil(img)
            return True
        except Exception:
            return False

    def _add_qimage(self, qimage: QImage):
        """Convert QImage to PNG bytes via Pillow normalization."""
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        qimage.save(buf, "PNG")
        buf.close()
        img = Image.open(io.BytesIO(bytes(ba.data())))
        self._add_pil(img)

    def _add_pil(self, img: Image.Image):
        """Normalize to RGBA PNG and store in project."""
        img = img.convert("RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.project.images.append(
            StickerImage(png_data=buf.getvalue(),
                         pixel_width=img.width,
                         pixel_height=img.height)
        )


# === MainWindow ===

class MainWindow(QMainWindow):
    """Top-level window: menu bar, page widget, status bar."""

    def __init__(self):
        super().__init__()
        self.project = StickerProject()
        self.tiler = Tiler()

        self.setWindowTitle("Sticker Sheet Maker")
        self.resize(800, 1000)

        self.page_widget = PageWidget(self.project)
        self.setCentralWidget(self.page_widget)
        self.page_widget.images_changed.connect(self._retile)

        self._build_menus()
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._update_status()

    def _build_menus(self):
        mb = self.menuBar()

        # File menu
        file_menu = mb.addMenu("&File")

        act = QAction("&New", self)
        act.setShortcut(QKeySequence.StandardKey.New)
        act.triggered.connect(self._new_project)
        file_menu.addAction(act)

        file_menu.addSeparator()

        act = QAction("&Print...", self)
        act.setShortcut(QKeySequence.StandardKey.Print)
        act.setEnabled(False)  # Stub — Phase 4
        file_menu.addAction(act)

        file_menu.addSeparator()

        act = QAction("&Quit", self)
        act.setShortcut(QKeySequence.StandardKey.Quit)
        act.triggered.connect(self.close)
        file_menu.addAction(act)

        # Edit menu
        edit_menu = mb.addMenu("&Edit")

        act = QAction("&Paste", self)
        act.setShortcut(QKeySequence.StandardKey.Paste)
        act.triggered.connect(self._paste)
        edit_menu.addAction(act)

        edit_menu.addSeparator()

        act = QAction("Clear &All", self)
        act.triggered.connect(self._clear_all)
        edit_menu.addAction(act)

    # --- Actions ---

    def _paste(self):
        """Paste image from clipboard."""
        cb = QApplication.clipboard()
        mime = cb.mimeData()

        if mime.hasImage():
            qimg = cb.image()
            if not qimg.isNull():
                self.page_widget._add_qimage(qimg)
                self._retile()
                return

        # Fallback: try raw image data from clipboard formats
        for fmt in mime.formats():
            if "image" in fmt.lower():
                data = mime.data(fmt)
                if data:
                    try:
                        img = Image.open(io.BytesIO(bytes(data)))
                        self.page_widget._add_pil(img)
                        self._retile()
                        return
                    except Exception:
                        continue

    def _new_project(self):
        self.project.images.clear()
        self.project.layout = LayoutResult()
        self.page_widget.invalidate_cache()
        self.page_widget.update()
        self._update_status()

    def _clear_all(self):
        self._new_project()

    def _retile(self):
        """Re-run the layout algorithm and repaint."""
        self.project.layout = self.tiler.layout(self.project.images)
        self.page_widget.invalidate_cache()
        self.page_widget.update()
        self._update_status()

    def _update_status(self):
        n = len(self.project.images)
        zoom = int(self.page_widget.page_scale() * 100)
        if n == 0:
            self._status.showMessage(
                f"No images \u2014 Paste (Ctrl+V) or drag images to add | Zoom: {zoom}%")
        else:
            self._status.showMessage(
                f"{n} image{'s' if n != 1 else ''} | Zoom: {zoom}%")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_status()


# === Entry Point ===

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sticker Sheet Maker")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    app = QApplication(sys.argv)
    app.setApplicationName("Sticker Sheet Maker")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
