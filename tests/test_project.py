"""Tests for StickerProject save/load round-trip via pickle."""
import os
import pickle
import tempfile

from sticker_app import StickerProject, StickerImage, Tiler, LayoutResult


class TestStickerProject:
    """Basic StickerProject data model tests."""

    def test_empty_project(self):
        proj = StickerProject()
        assert len(proj.images) == 0
        assert len(proj.layout.rows) == 0

    def test_add_image(self, circle_image):
        proj = StickerProject()
        sticker = StickerImage(png_data=circle_image, pixel_width=200, pixel_height=200)
        proj.images.append(sticker)
        assert len(proj.images) == 1
        assert proj.images[0].pixel_width == 200


class TestPickleRoundTrip:
    """Save and load projects via pickle, verifying data integrity."""

    def test_empty_project_round_trip(self):
        proj = StickerProject()
        data = pickle.dumps(proj)
        loaded = pickle.loads(data)
        assert isinstance(loaded, StickerProject)
        assert len(loaded.images) == 0

    def test_project_with_images_round_trip(self, sample_images):
        proj = StickerProject()
        from PIL import Image
        import io
        for png_bytes in sample_images:
            img = Image.open(io.BytesIO(png_bytes))
            proj.images.append(StickerImage(
                png_data=png_bytes,
                pixel_width=img.width,
                pixel_height=img.height,
            ))

        # Run tiler to populate layout
        tiler = Tiler()
        proj.layout = tiler.layout(proj.images)

        data = pickle.dumps(proj)
        loaded = pickle.loads(data)

        assert len(loaded.images) == len(proj.images)
        for orig, copy in zip(proj.images, loaded.images):
            assert orig.png_data == copy.png_data
            assert orig.pixel_width == copy.pixel_width
            assert orig.pixel_height == copy.pixel_height

        assert len(loaded.layout.rows) == len(proj.layout.rows)
        assert len(loaded.layout.placements) == len(proj.layout.placements)

    def test_file_round_trip(self, sample_images, tmp_path):
        """Write to a .sticker file, read it back, verify data."""
        proj = StickerProject()
        from PIL import Image
        import io
        for png_bytes in sample_images:
            img = Image.open(io.BytesIO(png_bytes))
            proj.images.append(StickerImage(
                png_data=png_bytes,
                pixel_width=img.width,
                pixel_height=img.height,
            ))

        tiler = Tiler()
        proj.layout = tiler.layout(proj.images)

        filepath = str(tmp_path / "test.sticker")
        with open(filepath, "wb") as f:
            pickle.dump(proj, f)

        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0

        with open(filepath, "rb") as f:
            loaded = pickle.load(f)

        assert isinstance(loaded, StickerProject)
        assert len(loaded.images) == 5
        assert len(loaded.layout.placements) == 5
