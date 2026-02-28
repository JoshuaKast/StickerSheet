"""View layer: Qt widgets for display and interaction.

Contains PageWidget (WYSIWYG page view), SettingsDialog, and MaskEditorDialog.
"""

import io
import re
import urllib.parse
import urllib.request

from PIL import Image
from PySide6.QtCore import Qt, QRectF, QPointF, QByteArray, QBuffer, QIODevice, QEvent, QRect, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QWidget, QDialog, QDialogButtonBox, QFormLayout, QComboBox, QDoubleSpinBox,
    QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QMenu,
)

from models import (
    StickerImage, StickerProject, PageSettings,
    CUT_LINE_STYLES, PAPER_SIZES, DPI,
)


# === PageWidget: WYSIWYG page view ===

class PageWidget(QWidget):
    """Draws the US Letter page with tiled sticker images and cut lines."""

    images_changed = Signal(list)  # emits list of StickerImage objects from drop
    selection_changed = Signal()
    zoom_changed = Signal()
    mask_edit_requested = Signal(int)  # emits image index

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

    def event(self, event):
        """Handle native gesture events (macOS trackpad pinch-to-zoom)."""
        if event.type() == QEvent.Type.NativeGesture:
            try:
                if event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                    self._handle_pinch_zoom(event)
                    return True
            except AttributeError:
                pass  # Platform doesn't support NativeGestureType
        return super().event(event)

    def _handle_pinch_zoom(self, event):
        """Zoom toward the pinch center point on trackpad pinch gestures."""
        factor = 1.0 + event.value()
        new_zoom = max(0.25, min(self._zoom * factor, 8.0))
        if new_zoom == self._zoom:
            return

        old_scale = self.page_scale()
        mouse_pos = event.position()
        ox, oy = self._page_origin()
        rel_x = mouse_pos.x() - ox
        rel_y = mouse_pos.y() - oy

        self._zoom = new_zoom
        new_scale = self.page_scale()
        scale_ratio = new_scale / old_scale

        new_ox = mouse_pos.x() - rel_x * scale_ratio
        new_oy = mouse_pos.y() - rel_y * scale_ratio
        base_ox = (self.width() - self._page_w() * new_scale) / 2
        base_oy = (self.height() - self._page_h() * new_scale) / 2
        self._pan = QPointF(new_ox - base_ox, new_oy - base_oy)

        self.zoom_changed.emit()
        self.update()

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

    @staticmethod
    def _fit_rect(cell: QRectF, src_w: float, src_h: float) -> QRectF:
        """Return the largest rect with src aspect ratio that fits inside cell, centered."""
        if src_w <= 0 or src_h <= 0:
            return cell
        src_aspect = src_w / src_h
        cell_aspect = cell.width() / cell.height() if cell.height() > 0 else 1
        if src_aspect > cell_aspect:
            # Image is wider than cell -- fit to width
            w = cell.width()
            h = w / src_aspect
        else:
            # Image is taller (or equal) -- fit to height
            h = cell.height()
            w = h * src_aspect
        x = cell.x() + (cell.width() - w) / 2
        y = cell.y() + (cell.height() - h) / 2
        return QRectF(x, y, w, h)

    def _paint_images(self, painter, layout, scale, ox, oy):
        for placed in layout.placements:
            pix = self._get_pixmap(placed.image_index)
            img = self.project.images[placed.image_index]
            mask = getattr(img, 'mask', None)
            cell = QRectF(ox + placed.x * scale, oy + placed.y * scale,
                          placed.width * scale, placed.height * scale)
            if mask is not None:
                mx, my, mw, mh = mask
                source = QRectF(mx, my, mw, mh)
                dest = self._fit_rect(cell, mw, mh)
                painter.drawPixmap(dest, pix, source)
            else:
                dest = self._fit_rect(cell, pix.width(), pix.height())
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
            # Unmodified scroll: pan (natural two-finger trackpad scrolling on macOS)
            pd = event.pixelDelta()
            if not pd.isNull():
                # pixelDelta is precise on macOS trackpads
                self._pan += QPointF(pd.x(), pd.y())
            else:
                # Fallback for mouse wheels
                self._pan += QPointF(event.angleDelta().x() / 2,
                                     event.angleDelta().y() / 2)
            self.update()
            event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = event.position()
            self._pan_start_offset = QPointF(self._pan)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            hit = self._hit_test(event.position())
            old = self.selected_index
            self.selected_index = hit
            if old != hit:
                self.selection_changed.emit()
                self.update()
            if hit is None:
                # Clicked empty space -- allow drag-to-pan
                self._pan_start = event.position()
                self._pan_start_offset = QPointF(self._pan)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._pan = self._pan_start_offset + delta
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pan_start is not None and event.button() in (
            Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton
        ):
            self._pan_start = None
            self.unsetCursor()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            hit = self._hit_test(event.position())
            if hit is not None:
                self.mask_edit_requested.emit(hit)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    # --- Drag and drop ---

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        mime = event.mimeData()
        stickers: list[StickerImage] = []

        # 1. Try image data first (some browsers include the actual bitmap)
        if mime.hasImage():
            qimg = QImage(mime.imageData())
            if not qimg.isNull():
                stickers.append(self._qimage_to_sticker(qimg))

        # 2. Try URLs (local files or remote image URLs)
        if not stickers and mime.hasUrls():
            for url in mime.urls():
                path = url.toLocalFile()
                if path:
                    s = self._load_file_as_sticker(path)
                    if s:
                        stickers.append(s)
                elif url.scheme() in ("http", "https"):
                    s = self._load_url_as_sticker(url.toString())
                    if s:
                        stickers.append(s)

        # 3. Try HTML -- browsers often provide text/html with <img src="...">
        if not stickers and mime.hasHtml():
            for img_url in self._extract_img_urls(mime.html()):
                s = self._load_url_as_sticker(img_url)
                if s:
                    stickers.append(s)
                    break

        # 4. Try plain text as a URL
        if not stickers and mime.hasText():
            text = mime.text().strip()
            if text.startswith(("http://", "https://")):
                s = self._load_url_as_sticker(text)
                if s:
                    stickers.append(s)

        if stickers:
            self.images_changed.emit(stickers)
        event.acceptProposedAction()

    def dragEnterEvent(self, event):
        mime = event.mimeData()
        if mime.hasImage() or mime.hasUrls() or mime.hasHtml() or mime.hasText():
            event.acceptProposedAction()

    def _load_file_as_sticker(self, path: str) -> StickerImage | None:
        try:
            img = Image.open(path)
            img.load()
            return self._pil_to_sticker(img)
        except Exception:
            return None

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

    @staticmethod
    def _extract_img_urls(html: str) -> list[str]:
        """Extract image URLs from HTML img tags (e.g. browser drag MIME data)."""
        urls = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE)
        # Also check srcset for higher-res variants
        for match in re.finditer(r'<img[^>]+srcset=["\']([^"\']+)["\']', html, re.IGNORECASE):
            # srcset format: "url1 1x, url2 2x" -- take the last (highest-res) entry
            entries = [e.strip().split()[0] for e in match.group(1).split(",") if e.strip()]
            if entries:
                urls.insert(0, entries[-1])  # prefer highest-res
        return [u for u in urls if u.startswith(("http://", "https://"))]

    def _load_url_as_sticker(self, url: str) -> StickerImage | None:
        """Download an image from a URL and return as StickerImage.

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
                return self._pil_to_sticker(img)
            except Exception:
                continue
        return None

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


# === Mask Editor Dialog ===

_HANDLE_SIZE = 8  # pixels, half-width of resize handles
_MIN_MASK = 10    # minimum mask dimension in image pixels


class _MaskCanvas(QWidget):
    """Interactive canvas for drawing a crop rectangle over an image."""

    def __init__(self, pixmap: QPixmap, mask_rect: QRect, parent=None):
        super().__init__(parent)
        self._pixmap = pixmap
        self._mask = QRect(mask_rect)
        self._drag_mode: str | None = None
        self._drag_start = QPointF()
        self._drag_start_mask = QRect()
        self.setMinimumSize(300, 300)
        self.setMouseTracking(True)

    def mask_rect(self) -> QRect:
        return QRect(self._mask)

    def set_mask(self, r: QRect):
        self._mask = QRect(r)
        self.update()

    # --- Coordinate mapping ---

    def _image_to_widget(self, ix: float, iy: float) -> QPointF:
        s, ox, oy = self._transform()
        return QPointF(ox + ix * s, oy + iy * s)

    def _widget_to_image(self, wx: float, wy: float) -> QPointF:
        s, ox, oy = self._transform()
        return QPointF((wx - ox) / s, (wy - oy) / s)

    def _transform(self) -> tuple[float, float, float]:
        """Return (scale, offset_x, offset_y) to fit image into widget."""
        padding = 10
        w = max(1, self.width() - 2 * padding)
        h = max(1, self.height() - 2 * padding)
        sx = w / max(1, self._pixmap.width())
        sy = h / max(1, self._pixmap.height())
        s = min(sx, sy)
        ox = padding + (w - self._pixmap.width() * s) / 2
        oy = padding + (h - self._pixmap.height() * s) / 2
        return s, ox, oy

    # --- Painting ---

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Background
        p.fillRect(self.rect(), QColor(60, 60, 60))

        s, ox, oy = self._transform()
        img_rect = QRectF(ox, oy, self._pixmap.width() * s, self._pixmap.height() * s)

        # Draw full image
        p.drawPixmap(img_rect.toRect(), self._pixmap)

        # Dimmed overlay outside mask
        mask_screen = QRectF(
            ox + self._mask.x() * s, oy + self._mask.y() * s,
            self._mask.width() * s, self._mask.height() * s,
        )
        dim = QColor(0, 0, 0, 120)
        # Top
        p.fillRect(QRectF(img_rect.x(), img_rect.y(),
                          img_rect.width(), mask_screen.y() - img_rect.y()), dim)
        # Bottom
        p.fillRect(QRectF(img_rect.x(), mask_screen.bottom(),
                          img_rect.width(), img_rect.bottom() - mask_screen.bottom()), dim)
        # Left
        p.fillRect(QRectF(img_rect.x(), mask_screen.y(),
                          mask_screen.x() - img_rect.x(), mask_screen.height()), dim)
        # Right
        p.fillRect(QRectF(mask_screen.right(), mask_screen.y(),
                          img_rect.right() - mask_screen.right(), mask_screen.height()), dim)

        # Mask border
        p.setPen(QPen(QColor(255, 255, 255), 2, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(mask_screen)

        # Resize handles
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255))
        for hx, hy in self._handle_positions(mask_screen):
            p.drawRect(QRectF(hx - _HANDLE_SIZE / 2, hy - _HANDLE_SIZE / 2,
                              _HANDLE_SIZE, _HANDLE_SIZE))

        p.end()

    def _handle_positions(self, r: QRectF) -> list[tuple[float, float]]:
        """Return screen positions for the 8 resize handles."""
        cx, cy = r.center().x(), r.center().y()
        return [
            (r.left(), r.top()), (cx, r.top()), (r.right(), r.top()),
            (r.left(), cy), (r.right(), cy),
            (r.left(), r.bottom()), (cx, r.bottom()), (r.right(), r.bottom()),
        ]

    _HANDLE_NAMES = ["nw", "n", "ne", "w", "e", "sw", "s", "se"]
    _HANDLE_CURSORS = {
        "nw": Qt.CursorShape.SizeFDiagCursor, "se": Qt.CursorShape.SizeFDiagCursor,
        "ne": Qt.CursorShape.SizeBDiagCursor, "sw": Qt.CursorShape.SizeBDiagCursor,
        "n": Qt.CursorShape.SizeVerCursor, "s": Qt.CursorShape.SizeVerCursor,
        "w": Qt.CursorShape.SizeHorCursor, "e": Qt.CursorShape.SizeHorCursor,
        "move": Qt.CursorShape.SizeAllCursor,
    }

    def _hit_handle(self, pos: QPointF) -> str | None:
        """Return handle name or 'move' if inside mask, else None."""
        s, ox, oy = self._transform()
        mask_screen = QRectF(
            ox + self._mask.x() * s, oy + self._mask.y() * s,
            self._mask.width() * s, self._mask.height() * s,
        )
        handles = self._handle_positions(mask_screen)
        for i, (hx, hy) in enumerate(handles):
            if abs(pos.x() - hx) <= _HANDLE_SIZE and abs(pos.y() - hy) <= _HANDLE_SIZE:
                return self._HANDLE_NAMES[i]
        if mask_screen.contains(pos):
            return "move"
        return None

    # --- Mouse interaction ---

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            mode = self._hit_handle(event.position())
            if mode is not None:
                self._drag_mode = mode
                self._drag_start = event.position()
                self._drag_start_mask = QRect(self._mask)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_mode is None:
            # Update cursor based on hover
            mode = self._hit_handle(event.position())
            if mode and mode in self._HANDLE_CURSORS:
                self.setCursor(self._HANDLE_CURSORS[mode])
            else:
                self.unsetCursor()
            return

        s, ox, oy = self._transform()
        if s <= 0:
            return

        # Compute delta in image pixel coordinates
        dx = (event.position().x() - self._drag_start.x()) / s
        dy = (event.position().y() - self._drag_start.y()) / s
        r = QRect(self._drag_start_mask)
        iw, ih = self._pixmap.width(), self._pixmap.height()

        if self._drag_mode == "move":
            nx = int(r.x() + dx)
            ny = int(r.y() + dy)
            nx = max(0, min(nx, iw - r.width()))
            ny = max(0, min(ny, ih - r.height()))
            self._mask = QRect(nx, ny, r.width(), r.height())
        else:
            x1, y1, x2, y2 = r.x(), r.y(), r.x() + r.width(), r.y() + r.height()
            mode = self._drag_mode
            if "w" in mode:
                x1 = max(0, min(int(r.x() + dx), x2 - _MIN_MASK))
            if "e" in mode:
                x2 = min(iw, max(int(r.x() + r.width() + dx), x1 + _MIN_MASK))
            if "n" in mode:
                y1 = max(0, min(int(r.y() + dy), y2 - _MIN_MASK))
            if "s" in mode:
                y2 = min(ih, max(int(r.y() + r.height() + dy), y1 + _MIN_MASK))
            self._mask = QRect(x1, y1, x2 - x1, y2 - y1)

        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag_mode is not None:
            self._drag_mode = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class MaskEditorDialog(QDialog):
    """Modal dialog for editing an image's crop mask."""

    def __init__(self, png_data: bytes, img_w: int, img_h: int,
                 current_mask: tuple[int, int, int, int] | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Mask")
        self._img_w = img_w
        self._img_h = img_h

        pix = QPixmap()
        pix.loadFromData(png_data)

        if current_mask is not None:
            init_rect = QRect(*current_mask)
        else:
            init_rect = QRect(0, 0, img_w, img_h)

        layout = QVBoxLayout(self)

        self._canvas = _MaskCanvas(pix, init_rect)
        layout.addWidget(self._canvas, 1)

        btn_row = QHBoxLayout()
        reset_btn = QPushButton("Reset (Full Image)")
        reset_btn.clicked.connect(self._reset_mask)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_row.addWidget(buttons)
        layout.addLayout(btn_row)

        # Size dialog to show image comfortably
        max_w, max_h = 700, 700
        aspect = img_w / img_h if img_h > 0 else 1.0
        if aspect > 1:
            dw = min(max_w, max(400, img_w))
            dh = int(dw / aspect) + 80
        else:
            dh = min(max_h, max(400, img_h))
            dw = int(dh * aspect) + 40
        self.resize(max(400, dw), max(400, dh))

    def _reset_mask(self):
        self._canvas.set_mask(QRect(0, 0, self._img_w, self._img_h))

    def result_mask(self) -> tuple[int, int, int, int] | None:
        """Return the mask, or None if it covers the full image."""
        r = self._canvas.mask_rect()
        if r.x() == 0 and r.y() == 0 and r.width() == self._img_w and r.height() == self._img_h:
            return None
        return (r.x(), r.y(), r.width(), r.height())
