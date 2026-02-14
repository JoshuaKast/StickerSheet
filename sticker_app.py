#!/usr/bin/env python3
"""Sticker Sheet Maker - Tile pasted images onto a US Letter page for sticker printing."""

import io
import math
import pickle
import sys
from dataclasses import dataclass, field

from PIL import Image
from PySide6.QtCore import Qt, QRectF, QPointF, QByteArray, QBuffer, QIODevice, Signal
from PySide6.QtGui import (
    QAction, QImage, QPixmap, QPainter, QPen, QColor, QKeySequence, QUndoStack, QUndoCommand,
)
from PySide6.QtPrintSupport import QPrintDialog, QPrinter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStatusBar, QFileDialog, QMessageBox, QMenu,
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
    selection_changed = Signal()

    def __init__(self, project: StickerProject, parent=None):
        super().__init__(parent)
        self.project = project
        self._pixmap_cache: dict[int, QPixmap] = {}
        self.selected_index: int | None = None
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
            self._paint_selection(painter, layout, scale, ox, oy)

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

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            hit = self._hit_test(event.position())
            old = self.selected_index
            self.selected_index = hit
            if old != hit:
                self.selection_changed.emit()
                self.update()
        super().mousePressEvent(event)

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


# === MainWindow ===

class MainWindow(QMainWindow):
    """Top-level window: menu bar, page widget, status bar."""

    def __init__(self):
        super().__init__()
        self.project = StickerProject()
        self.tiler = Tiler()
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
        delete_act = menu.addAction("Delete")
        chosen = menu.exec(self.page_widget.mapToGlobal(pos))
        if chosen == copy_act:
            self._copy_selected()
        elif chosen == delete_act:
            self._delete_selected()

    def _new_project(self):
        if not self._check_unsaved():
            return
        self.project.images.clear()
        self.project.layout = LayoutResult()
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
        zoom = int(self.page_widget.page_scale() * 100)
        sel = self.page_widget.selected_index
        if n == 0:
            self._status.showMessage(
                f"No images \u2014 Paste (Ctrl+V) or drag images to add | Zoom: {zoom}%")
        else:
            sel_text = f" | Selected: #{sel + 1}" if sel is not None else ""
            self._status.showMessage(
                f"{n} image{'s' if n != 1 else ''}{sel_text} | Zoom: {zoom}%")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_status()

    def closeEvent(self, event):
        if self._check_unsaved():
            event.accept()
        else:
            event.ignore()


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
