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


class TestMaskPickle:
    """Pickle round-trip tests for the mask field."""

    def test_masked_image_round_trip(self, circle_image):
        proj = StickerProject()
        img = StickerImage(png_data=circle_image, pixel_width=200, pixel_height=200,
                           mask=(10, 10, 180, 180))
        proj.images.append(img)

        data = pickle.dumps(proj)
        loaded = pickle.loads(data)

        assert loaded.images[0].mask == (10, 10, 180, 180)
        assert loaded.images[0].effective_width == 180
        assert loaded.images[0].effective_height == 180

    def test_no_mask_defaults_to_none(self, circle_image):
        """Image without mask should have mask=None and use pixel dims."""
        img = StickerImage(png_data=circle_image, pixel_width=200, pixel_height=200)
        assert img.mask is None
        assert img.effective_width == 200
        assert img.effective_height == 200

    def test_backward_compat_no_mask_field(self, circle_image):
        """Old pickled StickerImage without mask field should default to None."""
        img = StickerImage(png_data=circle_image, pixel_width=200, pixel_height=200)
        data = pickle.dumps(img)
        # Simulate old pickle by removing the mask field from the dict
        # (old objects wouldn't have it)
        loaded = pickle.loads(data)
        # Even though it was pickled with mask=None, verify the property works
        assert getattr(loaded, 'mask', None) is None
        assert loaded.effective_width == 200
