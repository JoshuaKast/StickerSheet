"""Unit tests for the Tiler layout engine."""
import math

from sticker_app import Tiler, StickerImage, MARGIN, PRINTABLE_WIDTH, PRINTABLE_HEIGHT, CUT_GAP


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


class TestBinCount:
    """Tiler._bin_count returns appropriate bin counts for different image quantities."""

    def test_small_counts(self):
        tiler = Tiler()
        assert tiler._bin_count(1) == 1
        assert tiler._bin_count(2) == 1
        assert tiler._bin_count(3) == 2
        assert tiler._bin_count(5) == 2
        assert tiler._bin_count(6) == 3
        assert tiler._bin_count(10) == 3
        assert tiler._bin_count(11) == 4
        assert tiler._bin_count(20) == 4
        assert tiler._bin_count(21) == 5


class TestHeightQuantization:
    """Step 2: heights are clustered into discrete bins."""

    def test_single_bin(self):
        tiler = Tiler()
        heights = [5.0, 5.5, 6.0]
        bins = tiler._compute_bins(heights, 1)
        assert len(bins) == 1

    def test_multiple_bins(self):
        tiler = Tiler()
        heights = [3.0, 3.5, 7.0, 7.5, 11.0, 11.5]
        bins = tiler._compute_bins(heights, 3)
        assert len(bins) >= 2  # at least 2 distinct bins (may merge if very close)
        assert len(bins) <= 3

    def test_bins_are_sorted(self):
        tiler = Tiler()
        heights = [10.0, 2.0, 6.0, 1.0, 8.0]
        bins = tiler._compute_bins(heights, 3)
        assert bins == sorted(bins)


class TestRowPacking:
    """Step 3: images are packed into rows without exceeding page width."""

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
        """A square image masked to 2:1 should produce a wider-than-tall placement."""
        tiler = Tiler()
        img = _make_sticker(1000, 1000, mask=(0, 0, 1000, 500))
        result = tiler.layout([img])
        assert len(result.placements) == 1
        p = result.placements[0]
        assert p.width > p.height, "Masked 2:1 image should be wider than tall"

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
