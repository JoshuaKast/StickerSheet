"""Unit tests for the Tiler layout engine."""
import math

from sticker_app import Tiler, StickerImage, PlacedImage, MARGIN, PRINTABLE_WIDTH, PRINTABLE_HEIGHT, CUT_GAP, PageSettings


def _make_sticker(pw, ph, mask=None):
    """Create a StickerImage with given pixel dimensions (no actual image data needed for Tiler)."""
    return StickerImage(png_data=b'', pixel_width=pw, pixel_height=ph, mask=mask)


class TestLogScaleSizing:
    """Step 1: log-scale sizing dampens resolution differences."""

    def test_log_scale_compresses_sizes(self):
        """A 10x resolution difference should become ~1.5x after log2."""
        small = math.log2(150 + 1)
        large = math.log2(1500 + 1)
        ratio = large / small
        assert 1.2 < ratio < 1.8, f"Expected ratio ~1.4, got {ratio}"

    def test_aspect_ratio_preserved(self):
        """Log is applied independently to width and height, preserving aspect ratio direction."""
        w, h = 300, 150  # 2:1 aspect
        lw = math.log2(w + 1)
        lh = math.log2(h + 1)
        assert lw > lh, "Wide image should remain wider than tall after log-scale"


class TestKMeansBinning:
    """Height quantization via k-means clustering."""

    def test_single_bin(self):
        tiler = Tiler()
        heights = [5.0, 5.5, 6.0]
        bins = tiler._kmeans_1d(heights, 1)
        assert len(bins) == 1

    def test_multiple_bins(self):
        tiler = Tiler()
        heights = [3.0, 3.5, 7.0, 7.5, 11.0, 11.5]
        bins = tiler._kmeans_1d(heights, 3)
        assert len(bins) >= 2
        assert len(bins) <= 3

    def test_bins_are_sorted(self):
        tiler = Tiler()
        heights = [10.0, 2.0, 6.0, 1.0, 8.0]
        bins = tiler._kmeans_1d(heights, 3)
        assert bins == sorted(bins)

    def test_more_bins_than_values(self):
        tiler = Tiler()
        heights = [5.0, 10.0]
        bins = tiler._kmeans_1d(heights, 5)
        assert len(bins) == 2
        assert bins == sorted(bins)


class TestRowPacking:
    """Images are packed into rows without exceeding page width."""

    def test_single_image(self):
        tiler = Tiler()
        images = [_make_sticker(200, 200)]
        result = tiler.layout(images)
        assert len(result.rows) == 1
        assert len(result.placements) == 1

    def test_multiple_images_pack_into_rows(self):
        tiler = Tiler()
        images = [_make_sticker(200, 200) for _ in range(6)]
        result = tiler.layout(images)
        assert len(result.rows) >= 1
        assert len(result.placements) == 6

    def test_all_images_within_page_bounds(self):
        tiler = Tiler()
        images = [_make_sticker(w, h) for w, h in
                  [(200, 200), (300, 150), (150, 300), (400, 400), (100, 250)]]
        result = tiler.layout(images)
        for p in result.placements:
            assert p.x >= MARGIN
            assert p.y >= MARGIN
            assert p.x + p.width <= MARGIN + PRINTABLE_WIDTH
            assert p.y + p.height <= MARGIN + PRINTABLE_HEIGHT

    def test_images_within_same_row_share_height(self):
        tiler = Tiler()
        # All same-sized images should end up in rows with uniform height
        images = [_make_sticker(200, 200) for _ in range(4)]
        result = tiler.layout(images)
        for row in result.rows:
            heights = [p.height for p in row.placements]
            assert len(set(heights)) == 1, f"Heights in a row should be uniform, got {heights}"

    def test_gap_between_images_in_row(self):
        """Adjacent images in a row should have at least CUT_GAP pixels between them."""
        tiler = Tiler()
        images = [_make_sticker(200, 200) for _ in range(4)]
        result = tiler.layout(images)
        for row in result.rows:
            sorted_p = sorted(row.placements, key=lambda p: p.x)
            for i in range(len(sorted_p) - 1):
                gap = sorted_p[i + 1].x - (sorted_p[i].x + sorted_p[i].width)
                assert gap >= CUT_GAP - 1, f"Gap between images should be >= {CUT_GAP}, got {gap}"


class TestScaleToFit:
    """Scale-to-fit: if rows overflow, everything shrinks uniformly."""

    def test_empty_layout(self):
        tiler = Tiler()
        result = tiler.layout([])
        assert len(result.rows) == 0

    def test_many_images_still_fit(self):
        """Even with many images, scale-to-fit should keep everything on the page."""
        tiler = Tiler()
        images = [_make_sticker(500, 500) for _ in range(30)]
        result = tiler.layout(images)
        assert len(result.placements) == 30
        for p in result.placements:
            assert p.y + p.height <= MARGIN + PRINTABLE_HEIGHT + 1  # +1 for rounding

    def test_large_images_scale_down(self):
        """Very large images should be scaled to fit the page."""
        tiler = Tiler()
        images = [_make_sticker(5000, 5000) for _ in range(10)]
        result = tiler.layout(images)
        assert len(result.placements) == 10
        for p in result.placements:
            assert p.width <= PRINTABLE_WIDTH
            assert p.height <= PRINTABLE_HEIGHT


class TestRotation:
    """The tiler may rotate images 90 degrees to improve packing."""

    def test_rotated_flag_exists(self):
        """PlacedImage should have a rotated field."""
        tiler = Tiler()
        images = [_make_sticker(200, 200)]
        result = tiler.layout(images)
        p = result.placements[0]
        assert hasattr(p, 'rotated')

    def test_rotation_preserves_aspect_ratio(self):
        """When an image is rotated, its aspect ratio should be inverted."""
        tiler = Tiler()
        images = [_make_sticker(400, 100)]  # very wide image
        result = tiler.layout(images)
        p = result.placements[0]
        placed_ar = p.width / p.height
        if p.rotated:
            # Rotated: placed AR should be close to 1/4 = 0.25
            assert placed_ar < 1.0, "Rotated wide image should become taller than wide"
        else:
            # Not rotated: placed AR should be close to 4.0
            assert placed_ar > 1.0, "Non-rotated wide image should be wider than tall"

    def test_all_placements_within_bounds_with_rotation(self):
        """Rotated images must still fit within page bounds."""
        tiler = Tiler()
        images = [_make_sticker(w, h) for w, h in
                  [(100, 500), (500, 100), (200, 300), (300, 200)]]
        result = tiler.layout(images)
        for p in result.placements:
            assert p.x >= MARGIN
            assert p.y >= MARGIN
            assert p.x + p.width <= MARGIN + PRINTABLE_WIDTH
            assert p.y + p.height <= MARGIN + PRINTABLE_HEIGHT


class TestFillRatio:
    """The new tiler should achieve better fill ratios than basic packing."""

    def test_fill_ratio_reasonable(self):
        """With several square images, fill ratio should be at least 30%."""
        tiler = Tiler()
        images = [_make_sticker(300, 300) for _ in range(8)]
        result = tiler.layout(images)
        total_img_area = sum(p.width * p.height for p in result.placements)
        fill = total_img_area / (PRINTABLE_WIDTH * PRINTABLE_HEIGHT)
        assert fill > 0.3, f"Fill ratio {fill:.3f} is too low"

    def test_fill_ratio_varied_images(self):
        """With varied aspect ratios, fill ratio should still be decent."""
        tiler = Tiler()
        images = [_make_sticker(w, h) for w, h in
                  [(100, 100), (300, 100), (100, 300), (500, 500), (200, 150)]]
        result = tiler.layout(images)
        total_img_area = sum(p.width * p.height for p in result.placements)
        fill = total_img_area / (PRINTABLE_WIDTH * PRINTABLE_HEIGHT)
        assert fill > 0.2, f"Fill ratio {fill:.3f} is too low"


class TestMask:
    """Mask (crop) feature: effective dimensions affect layout."""

    def test_effective_dimensions_no_mask(self):
        img = _make_sticker(1000, 500)
        assert img.effective_width == 1000
        assert img.effective_height == 500

    def test_effective_dimensions_with_mask(self):
        img = _make_sticker(1000, 1000, mask=(100, 100, 800, 400))
        assert img.effective_width == 800
        assert img.effective_height == 400

    def test_masked_image_aspect_ratio_in_layout(self):
        """A square image masked to 2:1 should preserve that aspect ratio in placement.
        The tiler may rotate the image, so we check the aspect ratio matches either
        the normal (2:1) or rotated (1:2) orientation."""
        tiler = Tiler()
        img = _make_sticker(1000, 1000, mask=(0, 0, 1000, 500))
        result = tiler.layout([img])
        assert len(result.placements) == 1
        p = result.placements[0]
        placed_ar = p.width / p.height
        rotated = getattr(p, 'rotated', False)
        if rotated:
            # Rotated 2:1 becomes 1:2
            assert placed_ar < 1.0, "Rotated 2:1 image should be taller than wide"
            assert abs(placed_ar - 0.5) < 0.1, f"Rotated aspect should be ~0.5, got {placed_ar:.3f}"
        else:
            # Normal 2:1
            assert placed_ar > 1.0, "2:1 image should be wider than tall"
            assert abs(placed_ar - 2.0) < 0.2, f"Aspect should be ~2.0, got {placed_ar:.3f}"

    def test_mask_none_same_as_no_mask(self):
        """mask=None should produce same layout as no mask."""
        tiler = Tiler()
        img_no_mask = _make_sticker(500, 300)
        img_none_mask = _make_sticker(500, 300, mask=None)
        r1 = tiler.layout([img_no_mask])
        r2 = tiler.layout([img_none_mask])
        p1, p2 = r1.placements[0], r2.placements[0]
        assert p1.width == p2.width
        assert p1.height == p2.height

    def test_masked_images_fit_page(self):
        """Masked images should still fit within page bounds."""
        tiler = Tiler()
        images = [_make_sticker(1000, 1000, mask=(0, 0, 1000, 200)) for _ in range(15)]
        result = tiler.layout(images)
        assert len(result.placements) == 15
        for p in result.placements:
            assert p.y + p.height <= MARGIN + PRINTABLE_HEIGHT + 1


class TestUserScaling:
    """Per-image scale_step should affect placement size."""

    def test_positive_scale_step_enlarges(self):
        """When placed alongside a normal image, a scaled-up image should be larger."""
        tiler = Tiler()
        img_normal = _make_sticker(300, 300)
        img_scaled = StickerImage(png_data=b'', pixel_width=300, pixel_height=300, scale_step=2)
        result = tiler.layout([img_normal, img_scaled])
        p_normal = [p for p in result.placements if p.image_index == 0][0]
        p_scaled = [p for p in result.placements if p.image_index == 1][0]
        area_normal = p_normal.width * p_normal.height
        area_scaled = p_scaled.width * p_scaled.height
        assert area_scaled > area_normal, "Positive scale_step should enlarge the image"

    def test_negative_scale_step_shrinks(self):
        """When placed alongside a normal image, a scaled-down image should be smaller."""
        tiler = Tiler()
        img_normal = _make_sticker(300, 300)
        img_scaled = StickerImage(png_data=b'', pixel_width=300, pixel_height=300, scale_step=-2)
        result = tiler.layout([img_normal, img_scaled])
        p_normal = [p for p in result.placements if p.image_index == 0][0]
        p_scaled = [p for p in result.placements if p.image_index == 1][0]
        area_normal = p_normal.width * p_normal.height
        area_scaled = p_scaled.width * p_scaled.height
        assert area_scaled < area_normal, "Negative scale_step should shrink the image"


class TestPageSettings:
    """Tiler should respect custom PageSettings."""

    def test_custom_margin(self):
        settings = PageSettings(margin_inches=0.5)
        tiler = Tiler(settings)
        images = [_make_sticker(200, 200)]
        result = tiler.layout(images)
        p = result.placements[0]
        expected_margin = int(0.5 * 300)
        assert p.x >= expected_margin
        assert p.y >= expected_margin
