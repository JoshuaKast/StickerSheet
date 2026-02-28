"""Tiler: log-scale sizing, row quantization, row packing, scale-to-fit.

The layout engine that arranges sticker images into rows on the page.
"""

import math

from models import PageSettings, StickerImage, PlacedImage, LayoutRow, LayoutResult


class Tiler:
    """Layout engine implementing the row-based tiling algorithm."""

    def __init__(self, settings: PageSettings | None = None):
        s = settings or PageSettings()
        self.pw = s.printable_width
        self.ph = s.printable_height
        self.gap = s.cut_gap
        self.margin = s.margin

    def layout(self, images: list[StickerImage]) -> LayoutResult:
        """Run the full layout pipeline on the given images."""
        if not images:
            return LayoutResult()

        # Step 1: Log-scale sizing -- dampen resolution differences
        # Use log-scale on the geometric mean of dimensions to get an overall
        # "ideal size", then distribute width/height by the actual aspect ratio.
        # This prevents tall rows for wide images (the old approach applied log2
        # independently to width and height, which made all images roughly square
        # in ideal-space regardless of their actual aspect ratio).
        raw_ideal = []
        for img in images:
            pw, ph = img.effective_width, img.effective_height
            geo_mean = math.sqrt(pw * ph) if pw > 0 and ph > 0 else max(pw, ph, 1)
            ideal_size = math.log2(geo_mean + 1)
            aspect = pw / ph if ph > 0 else 1.0
            # Distribute: iw * ih == ideal_size^2, iw/ih == aspect
            ih = ideal_size / math.sqrt(aspect) if aspect > 0 else ideal_size
            iw = ih * aspect
            raw_ideal.append((iw, ih))

        # Step 2: Quantize heights into row bins (from UNSCALED ideals for stability)
        n_bins = self._bin_count(len(images))
        bins = self._compute_bins([ih for _, ih in raw_ideal], n_bins)

        # Step 3: Assign images to bins based on unscaled ideal heights.
        # Per-image scale_step is carried through as a multiplier applied
        # during packing, so it always produces a visible size change even
        # when there's only one bin.
        # groups: bin_h -> list of (ideal_w, image_index, per_image_factor)
        groups: dict[float, list[tuple[float, int, float]]] = {}
        for i, (iw, ih) in enumerate(raw_ideal):
            step = getattr(images[i], 'scale_step', 0)
            factor = 1.2 ** step if step != 0 else 1.0
            best_bin = min(bins, key=lambda b: abs(b - ih))
            scaled_w = iw * (best_bin / ih) if ih > 0 else iw
            groups.setdefault(best_bin, []).append((scaled_w, i, factor))

        # Step 4: Pack rows with scale-to-fit via binary search
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

            current: list[tuple[int, int, int, int]] = []  # (x, width, height, image_index)
            cx = 0

            for ideal_w, idx, img_factor in groups[bin_h]:
                w = max(1, int(ideal_w * scale * img_factor))
                h = max(1, int(bin_h * scale * img_factor))
                if w > self.pw:
                    w = self.pw

                needed = cx + (self.gap if current else 0) + w
                if needed > self.pw and current:
                    # Row height = tallest image in the row
                    actual_row_h = max(item_h for _, _, item_h, _ in current)
                    rows.append((actual_row_h, current))
                    current = []
                    cx = 0

                if current:
                    cx += self.gap
                current.append((cx, w, h, idx))
                cx += w

            if current:
                actual_row_h = max(item_h for _, _, item_h, _ in current)
                rows.append((actual_row_h, current))

        # Check vertical fit
        total = sum(h for h, _ in rows) + self.gap * max(0, len(rows) - 1)
        if total > self.ph:
            return None

        # Build final layout with absolute positions
        layout_rows = []
        y = self.margin
        for row_h, items in rows:
            placements = [
                PlacedImage(image_index=idx, x=self.margin + x, y=y, width=w, height=h)
                for x, w, h, idx in items
            ]
            layout_rows.append(LayoutRow(y=y, height=row_h, placements=placements))
            y += row_h + self.gap

        return LayoutResult(rows=layout_rows)
