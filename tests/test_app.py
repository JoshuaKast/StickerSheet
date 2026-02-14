"""Integration tests for the Sticker Sheet app via pytest-qt."""
import io
import os
import pickle

import pytest
from PIL import Image
from PySide6.QtGui import QImage

from sticker_app import MainWindow, StickerProject, StickerImage, Tiler


@pytest.fixture
def main_window(qapp, qtbot):
    """Create a MainWindow managed by qtbot."""
    win = MainWindow()
    # Override closeEvent to avoid unsaved-changes dialog during test teardown
    win.closeEvent = lambda event: event.accept()
    qtbot.addWidget(win)
    win.show()
    return win


def _png_to_qimage(png_bytes):
    """Convert PNG bytes to a QImage."""
    qimg = QImage()
    qimg.loadFromData(png_bytes)
    return qimg


class TestPasteWorkflow:
    """Simulate pasting images and verify the layout updates."""

    def test_paste_single_image(self, main_window, circle_image):
        """Paste one image, verify it appears in the project."""
        sticker = main_window.page_widget._qimage_to_sticker(_png_to_qimage(circle_image))
        main_window.project.images.append(sticker)
        main_window._retile()

        assert len(main_window.project.images) == 1
        assert len(main_window.project.layout.placements) == 1

    def test_paste_multiple_images(self, main_window, sample_images):
        """Paste several images, verify layout has all of them."""
        for png_bytes in sample_images:
            sticker = main_window.page_widget._qimage_to_sticker(_png_to_qimage(png_bytes))
            main_window.project.images.append(sticker)

        main_window._retile()

        assert len(main_window.project.images) == 5
        assert len(main_window.project.layout.placements) == 5

    def test_status_bar_updates(self, main_window, circle_image):
        """Status bar should reflect image count."""
        assert "No images" in main_window._status.currentMessage()

        sticker = main_window.page_widget._qimage_to_sticker(_png_to_qimage(circle_image))
        main_window.project.images.append(sticker)
        main_window._retile()

        assert "1 image" in main_window._status.currentMessage()


class TestDeleteWorkflow:
    """Test image deletion."""

    def test_delete_image(self, main_window, sample_images):
        """Add images, delete one, verify count decreases."""
        for png_bytes in sample_images[:3]:
            sticker = main_window.page_widget._qimage_to_sticker(_png_to_qimage(png_bytes))
            main_window.project.images.append(sticker)
        main_window._retile()
        assert len(main_window.project.images) == 3

        # Select and delete the first image
        main_window.page_widget.selected_index = 0
        main_window._delete_selected()

        assert len(main_window.project.images) == 2


class TestSaveLoad:
    """Test File > Save and File > Open workflows."""

    def test_save_creates_file(self, main_window, circle_image, tmp_path):
        """Saving should create a .sticker file."""
        sticker = main_window.page_widget._qimage_to_sticker(_png_to_qimage(circle_image))
        main_window.project.images.append(sticker)
        main_window._retile()

        filepath = str(tmp_path / "test_save.sticker")
        result = main_window._write_file(filepath)

        assert result is True
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0

    def test_load_restores_images(self, main_window, sample_images, tmp_path):
        """Save a project, create new window, load — images should match."""
        # Build and save a project
        for png_bytes in sample_images:
            sticker = main_window.page_widget._qimage_to_sticker(_png_to_qimage(png_bytes))
            main_window.project.images.append(sticker)
        main_window._retile()

        filepath = str(tmp_path / "test_load.sticker")
        main_window._write_file(filepath)

        # Load into the same window (simulating File > Open)
        main_window._new_project()
        assert len(main_window.project.images) == 0

        main_window.open_file(filepath)
        assert len(main_window.project.images) == 5
        assert len(main_window.project.layout.placements) == 5


class TestNewProject:
    """Test File > New workflow."""

    def test_new_clears_project(self, main_window, circle_image):
        sticker = main_window.page_widget._qimage_to_sticker(_png_to_qimage(circle_image))
        main_window.project.images.append(sticker)
        main_window._retile()
        assert len(main_window.project.images) == 1

        # Mark as clean so _check_unsaved() returns True without dialog
        main_window._undo_stack.setClean()
        main_window._mark_clean()
        main_window._new_project()

        assert len(main_window.project.images) == 0
        assert len(main_window.project.layout.rows) == 0
