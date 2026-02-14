"""Tests for print/render functionality."""
import os

import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtPrintSupport import QPrinter

from sticker_app import (
    MainWindow, StickerImage, StickerProject, Tiler,
    DPI, PAGE_WIDTH, PAGE_HEIGHT,
)


def _png_to_qimage(png_bytes):
    qimg = QImage()
    qimg.loadFromData(png_bytes)
    return qimg


class TestPrintRender:
    """Test rendering the layout to a PDF via QPrinter."""

    def test_render_to_pdf(self, qapp, qtbot, sample_images, tmp_path):
        """Render a project with images to PDF, verify the file is created."""
        win = MainWindow()
        qtbot.addWidget(win)

        for png_bytes in sample_images:
            sticker = win.page_widget._qimage_to_sticker(_png_to_qimage(png_bytes))
            win.project.images.append(sticker)
        win._retile()

        # Render to PDF using QPrinter
        pdf_path = str(tmp_path / "test_output.pdf")
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(pdf_path)

        painter = QPainter()
        assert painter.begin(printer)

        layout = win.project.layout
        printer_dpi_x = printer.logicalDpiX()
        printer_dpi_y = printer.logicalDpiY()
        sx = printer_dpi_x / DPI
        sy = printer_dpi_y / DPI
        painter.scale(sx, sy)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        for placed in layout.placements:
            qimg = QImage()
            qimg.loadFromData(win.project.images[placed.image_index].png_data)
            dest = QRectF(placed.x, placed.y, placed.width, placed.height)
            painter.drawImage(dest, qimg)

        painter.end()

        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0

    def test_render_empty_project(self, qapp, qtbot, tmp_path):
        """Rendering an empty project should still produce a valid (small) PDF."""
        win = MainWindow()
        qtbot.addWidget(win)

        pdf_path = str(tmp_path / "empty_output.pdf")
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(pdf_path)

        painter = QPainter()
        assert painter.begin(printer)
        # Nothing to render — just end
        painter.end()

        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0
