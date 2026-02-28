"""Controller layer: MainWindow, undo commands, and StickerApp.

Orchestrates the model, tiler, and views.
"""

import io
import pickle
import urllib.request

from PIL import Image
from PySide6.QtCore import Qt, QEvent, Signal
from PySide6.QtGui import (
    QAction, QImage, QKeySequence, QUndoStack, QUndoCommand, QPainter,
)
from PySide6.QtPrintSupport import QPrintDialog, QPrinter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QStatusBar, QFileDialog, QMessageBox, QMenu, QDialog,
)

from models import (
    StickerProject, StickerImage, PageSettings, LayoutResult,
    DPI, STICKER_FILTER,
)
from tiler import Tiler
from views import PageWidget, SettingsDialog, MaskEditorDialog


# === Undo Commands ===

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


class MaskImageCommand(QUndoCommand):
    """Undoable command: change an image's crop mask."""

    def __init__(self, window: "MainWindow", index: int,
                 old_mask: tuple[int, int, int, int] | None,
                 new_mask: tuple[int, int, int, int] | None):
        super().__init__("Edit Mask")
        self._window = window
        self._index = index
        self._old_mask = old_mask
        self._new_mask = new_mask

    def redo(self):
        self._window.project.images[self._index].mask = self._new_mask
        self._window._retile()

    def undo(self):
        self._window.project.images[self._index].mask = self._old_mask
        self._window._retile()


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
        self.page_widget.mask_edit_requested.connect(self._edit_mask)

        self._build_menus()
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._update_title()
        self._update_status()

    def _build_menus(self):
        mb = self.menuBar()

        # --- About action (macOS places this in the app menu automatically) ---
        about_act = QAction("&About Sticker Sheet Maker", self)
        about_act.setMenuRole(QAction.MenuRole.AboutRole)
        about_act.triggered.connect(self._show_about)

        # --- File menu ---
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

        act = QAction("&Close Window", self)
        act.setShortcut(QKeySequence.StandardKey.Close)
        act.triggered.connect(self.close)
        file_menu.addAction(act)

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

        # --- Edit menu ---
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
        # On macOS, the key labeled "delete" sends Backspace, not forward-delete.
        # Map both so the action works with either key.
        self._delete_action.setShortcuts([
            QKeySequence.StandardKey.Delete,
            QKeySequence(Qt.Key.Key_Backspace),
        ])
        self._delete_action.triggered.connect(self._delete_selected)
        self._delete_action.setEnabled(False)
        edit_menu.addAction(self._delete_action)

        self._mask_action = QAction("Edit &Mask...", self)
        self._mask_action.setShortcut(QKeySequence("Ctrl+M"))
        self._mask_action.triggered.connect(self._edit_mask_selected)
        self._mask_action.setEnabled(False)
        edit_menu.addAction(self._mask_action)

        edit_menu.addSeparator()

        act = QAction("Clear &All", self)
        act.triggered.connect(self._clear_all)
        edit_menu.addAction(act)

        edit_menu.addSeparator()

        # Preferences -- macOS auto-moves this to the app menu via PreferencesRole
        act = QAction("&Preferences...", self)
        act.setShortcut(QKeySequence("Ctrl+,"))
        act.setMenuRole(QAction.MenuRole.PreferencesRole)
        act.triggered.connect(self._show_settings)
        edit_menu.addAction(act)

        # About must be added to a menu for macOS role handling to pick it up
        edit_menu.addAction(about_act)

        # --- View menu ---
        view_menu = mb.addMenu("&View")

        act = QAction("Zoom &In", self)
        act.setShortcut(QKeySequence.StandardKey.ZoomIn)
        act.triggered.connect(lambda: self._zoom_view(1.25))
        view_menu.addAction(act)

        act = QAction("Zoom &Out", self)
        act.setShortcut(QKeySequence.StandardKey.ZoomOut)
        act.triggered.connect(lambda: self._zoom_view(1 / 1.25))
        view_menu.addAction(act)

        act = QAction("Zoom to &Fit", self)
        act.setShortcut(QKeySequence("Ctrl+0"))
        act.triggered.connect(self.page_widget.reset_view)
        view_menu.addAction(act)

        view_menu.addSeparator()

        self._fullscreen_action = QAction("Enter Full Screen", self)
        self._fullscreen_action.setShortcut(QKeySequence.StandardKey.FullScreen)
        self._fullscreen_action.triggered.connect(self._toggle_full_screen)
        view_menu.addAction(self._fullscreen_action)

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
        self.setWindowTitle(f"{name}{dirty} \u2014 Sticker Sheet Maker")
        # macOS shows a proxy icon in the title bar when windowFilePath is set;
        # Cmd-clicking the title reveals the file's path in Finder.
        self.setWindowFilePath(self._file_path or "")

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
        self._mask_action.setEnabled(has_sel)
        self._update_status()

    # --- Actions ---

    def _paste(self):
        """Paste image from clipboard via undo stack.

        Tries in order: image data, raw image MIME formats, HTML with <img> tags,
        plain text URL.
        """
        cb = QApplication.clipboard()
        mime = cb.mimeData()

        sticker = None

        # 1. Direct image data
        if mime.hasImage():
            qimg = cb.image()
            if not qimg.isNull():
                sticker = self.page_widget._qimage_to_sticker(qimg)

        # 2. Raw image MIME formats (e.g. image/png bytes)
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

        # 3. HTML with <img src="..."> (e.g. copied image from browser)
        if sticker is None and mime.hasHtml():
            for img_url in PageWidget._extract_img_urls(mime.html()):
                try:
                    upgraded = PageWidget._upgrade_url(img_url)
                    for attempt_url in ([upgraded, img_url] if upgraded != img_url else [img_url]):
                        try:
                            req = urllib.request.Request(
                                attempt_url, headers={"User-Agent": "StickerSheet/1.0"})
                            with urllib.request.urlopen(req, timeout=10) as resp:
                                raw = resp.read(10 * 1024 * 1024)
                            pil_img = Image.open(io.BytesIO(raw))
                            pil_img.load()
                            sticker = self.page_widget._pil_to_sticker(pil_img)
                            break
                        except Exception:
                            continue
                    if sticker:
                        break
                except Exception:
                    continue

        # 4. Plain text URL (e.g. copied image address)
        if sticker is None and mime.hasText():
            text = mime.text().strip()
            if text.startswith(("http://", "https://")):
                upgraded = PageWidget._upgrade_url(text)
                for attempt_url in ([upgraded, text] if upgraded != text else [text]):
                    try:
                        req = urllib.request.Request(
                            attempt_url, headers={"User-Agent": "StickerSheet/1.0"})
                        with urllib.request.urlopen(req, timeout=10) as resp:
                            raw = resp.read(10 * 1024 * 1024)
                        pil_img = Image.open(io.BytesIO(raw))
                        pil_img.load()
                        sticker = self.page_widget._pil_to_sticker(pil_img)
                        break
                    except Exception:
                        continue

        if sticker:
            self._undo_stack.push(PasteImageCommand(self, sticker))

    def _on_drop_images(self, stickers: list):
        """Handle images added via drag-and-drop by pushing undo commands."""
        for sticker in stickers:
            self._undo_stack.push(PasteImageCommand(self, sticker))

    def _delete_selected(self):
        idx = self.page_widget.selected_index
        if idx is not None and 0 <= idx < len(self.project.images):
            self._undo_stack.push(DeleteImageCommand(self, idx))

    def _copy_selected(self):
        """Copy the selected image back to clipboard (masked region if cropped)."""
        idx = self.page_widget.selected_index
        if idx is None or idx >= len(self.project.images):
            return
        img = self.project.images[idx]
        qimg = QImage()
        qimg.loadFromData(img.png_data)
        mask = getattr(img, 'mask', None)
        if mask is not None:
            mx, my, mw, mh = mask
            qimg = qimg.copy(mx, my, mw, mh)
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

    def _edit_mask_selected(self):
        """Open the mask editor for the currently selected image."""
        idx = self.page_widget.selected_index
        if idx is not None:
            self._edit_mask(idx)

    def _edit_mask(self, index: int):
        """Open the mask editor dialog for the image at the given index."""
        if index < 0 or index >= len(self.project.images):
            return
        img = self.project.images[index]
        old_mask = getattr(img, 'mask', None)
        dlg = MaskEditorDialog(img.png_data, img.pixel_width, img.pixel_height, old_mask, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_mask = dlg.result_mask()
            if new_mask != old_mask:
                self._undo_stack.push(MaskImageCommand(self, index, old_mask, new_mask))

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
        mask_act = menu.addAction("Edit Mask...")
        mask = getattr(img, 'mask', None)
        if mask is not None:
            clear_mask_act = menu.addAction("Clear Mask")
        else:
            clear_mask_act = None
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
        elif chosen == mask_act:
            self._edit_mask(idx)
        elif chosen == clear_mask_act:
            old_mask = getattr(img, 'mask', None)
            if old_mask is not None:
                self._undo_stack.push(MaskImageCommand(self, idx, old_mask, None))
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

    # --- View actions ---

    def _show_about(self):
        QMessageBox.about(
            self,
            "About Sticker Sheet Maker",
            "Sticker Sheet Maker\n\n"
            "Tile pasted images onto a printable page\n"
            "for sticker sheets.",
        )

    def _zoom_view(self, factor: float):
        """Zoom the page view by the given factor."""
        pw = self.page_widget
        new_zoom = max(0.25, min(pw._zoom * factor, 8.0))
        if new_zoom != pw._zoom:
            pw._zoom = new_zoom
            pw.zoom_changed.emit()
            pw.update()

    def _toggle_full_screen(self):
        if self.isFullScreen():
            self.showNormal()
            self._fullscreen_action.setText("Enter Full Screen")
        else:
            self.showFullScreen()
            self._fullscreen_action.setText("Exit Full Screen")

    # --- Save / Load ---

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

    # --- Print ---

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

        from PySide6.QtCore import QRectF
        for placed in layout.placements:
            img = self.project.images[placed.image_index]
            qimg = QImage()
            qimg.loadFromData(img.png_data)
            cell = QRectF(placed.x, placed.y, placed.width, placed.height)
            mask = getattr(img, 'mask', None)
            if mask is not None:
                mx, my, mw, mh = mask
                source = QRectF(mx, my, mw, mh)
                dest = PageWidget._fit_rect(cell, mw, mh)
                painter.drawImage(dest, qimg, source)
            else:
                dest = PageWidget._fit_rect(cell, qimg.width(), qimg.height())
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
        paste_shortcut = QKeySequence(
            QKeySequence.StandardKey.Paste
        ).toString(QKeySequence.SequenceFormat.NativeText)
        if n == 0:
            self._status.showMessage(
                f"No images \u2014 Paste ({paste_shortcut}) or drag images to add | Zoom: {zoom}%")
        else:
            sel_text = ""
            if sel is not None and sel < n:
                img = self.project.images[sel]
                step = getattr(img, 'scale_step', 0)
                scale_text = f" (scale: {'+' if step > 0 else ''}{step})" if step != 0 else ""
                mask_text = " [cropped]" if getattr(img, 'mask', None) is not None else ""
                sel_text = f" | Selected: #{sel + 1}{scale_text}{mask_text}"
            self._status.showMessage(
                f"{n} image{'s' if n != 1 else ''}{sel_text} | Zoom: {zoom}%")

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()

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
