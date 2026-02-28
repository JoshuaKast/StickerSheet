#!/usr/bin/env python3
"""Standalone tiler benchmark / quality evaluator.

Generates a set of randomly-sized images (plausible web dimensions), runs the
Tiler, and prints a suite of quality metrics.  No Qt, no GUI — pure Python.

Usage:
    python tiler_bench.py              # default 20 images, random seed 42
    python tiler_bench.py -n 30        # 30 images
    python tiler_bench.py --seed 7     # reproducible with a different seed
    python tiler_bench.py --verbose    # print per-image details
"""

import argparse
import math
import random
import sys

from models import PageSettings, StickerImage, PlacedImage
from tiler import Tiler

# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

# Typical web image dimensions (width, height) — sampled from real-world
# distribution: thumbnails, medium photos, banners, icons, high-res photos.
DIMENSION_POOLS = [
    # Icons / thumbnails
    (64, 64), (100, 100), (128, 128), (150, 150), (180, 180),
    # Small web images
    (200, 150), (250, 250), (300, 200), (320, 240),
    # Medium web images
    (400, 300), (480, 360), (500, 500), (600, 400), (640, 480),
    # Large web photos
    (800, 600), (1024, 768), (1200, 800), (1280, 960),
    # Wide banners / panoramic
    (728, 90), (960, 250), (1200, 300),
    # Tall / portrait
    (300, 600), (400, 800), (480, 720), (600, 900),
    # High-res photos
    (1600, 1200), (1920, 1080), (2048, 1536), (2560, 1440),
    # Very small favicons
    (16, 16), (32, 32), (48, 48),
]


def generate_images(n: int, rng: random.Random) -> list[StickerImage]:
    """Return *n* StickerImage objects with plausible web-image dimensions."""
    images: list[StickerImage] = []
    for _ in range(n):
        w, h = rng.choice(DIMENSION_POOLS)
        # Add ±20 % jitter so sizes aren't all identical
        w = max(8, int(w * rng.uniform(0.8, 1.2)))
        h = max(8, int(h * rng.uniform(0.8, 1.2)))
        # We don't need actual PNG data for the tiler — it only looks at
        # pixel_width / pixel_height.
        images.append(StickerImage(png_data=b"", pixel_width=w, pixel_height=h))
    return images


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TilerMetrics:
    """Compute and hold quality metrics for a tiler layout."""

    def __init__(
        self,
        images: list[StickerImage],
        settings: PageSettings,
        result,  # LayoutResult
    ):
        self.images = images
        self.settings = settings
        self.result = result
        self.placements: list[PlacedImage] = result.placements

        # Precompute
        self._compute_area_metrics()
        self._compute_street_lengths()
        self._compute_greedy_streets()
        self._compute_aspect_ratio_fidelity()
        self._compute_spacing()

    # -- (a) Area utilisation -----------------------------------------------

    def _compute_area_metrics(self):
        page_w = self.settings.page_width
        page_h = self.settings.page_height
        self.total_page_area = page_w * page_h

        margin = self.settings.margin
        self.printable_area = (
            self.settings.printable_width * self.settings.printable_height
        )

        self.filled_area = sum(p.width * p.height for p in self.placements)

        # Bounding-box of placed content (tightest enclosing rect)
        if self.placements:
            x_min = min(p.x for p in self.placements)
            y_min = min(p.y for p in self.placements)
            x_max = max(p.x + p.width for p in self.placements)
            y_max = max(p.y + p.height for p in self.placements)
            self.bbox_area = (x_max - x_min) * (y_max - y_min)
            self.bbox = (x_min, y_min, x_max, y_max)
        else:
            self.bbox_area = 0
            self.bbox = (0, 0, 0, 0)

        # Ratios
        self.fill_ratio_of_printable = (
            self.filled_area / self.printable_area if self.printable_area else 0
        )
        self.fill_ratio_of_bbox = (
            self.filled_area / self.bbox_area if self.bbox_area else 0
        )
        self.blank_ratio_of_printable = 1.0 - self.fill_ratio_of_printable

    # -- (b) Cut-street lengths ---------------------------------------------

    def _compute_street_lengths(self):
        """Measure the total length of cutting streets.

        Horizontal streets: full-width gaps between rows.
        Vertical streets: per-row gaps between adjacent images.
        """
        pw = self.settings.printable_width
        rows = self.result.rows

        self.horizontal_street_count = max(0, len(rows) - 1)
        # Each horizontal street spans the full printable width
        self.horizontal_street_total_length = self.horizontal_street_count * pw

        self.vertical_street_count = 0
        self.vertical_street_total_length = 0
        for row in rows:
            sorted_p = sorted(row.placements, key=lambda p: p.x)
            for i in range(len(sorted_p) - 1):
                gap_left = sorted_p[i].x + sorted_p[i].width
                gap_right = sorted_p[i + 1].x
                if gap_right > gap_left:
                    self.vertical_street_count += 1
                    # Vertical street length = height of the row
                    self.vertical_street_total_length += row.height

        self.total_street_length = (
            self.horizontal_street_total_length + self.vertical_street_total_length
        )

    # -- (b2) Greedy street cover (median street length) --------------------

    def _compute_greedy_streets(self):
        """Compute the greedy minimum-cut-set street metric.

        Algorithm:
        1. Enumerate all streets (horizontal between rows, vertical between
           adjacent images within a row).
        2. Sort by length descending.
        3. Greedily select streets — skip any that overlap with an already-
           selected street (intersections are OK) — until every tile adjoins
           a selected street on all four sides.  Sides that face the page
           margin are pre-covered (no cut needed along the page edge).
        4. Report mean and median of the selected street lengths.

        In a row-based layout no two candidate streets ever overlap (horizontal
        streets differ in y; vertical streets within a row differ in x; vertical
        streets across rows differ in y-range), so the overlap check is vacuous
        and every candidate is eligible.
        """
        rows = self.result.rows
        if not rows:
            self.greedy_street_lengths: list[int] = []
            self.greedy_street_mean = 0.0
            self.greedy_street_median = 0.0
            return

        pw = self.settings.printable_width
        sorted_rows = sorted(rows, key=lambda r: r.y)
        n_rows = len(sorted_rows)

        # Stable sorted placements per row
        row_placements = [
            sorted(row.placements, key=lambda p: p.x) for row in sorted_rows
        ]

        # --- Enumerate candidate streets ---
        # (length, kind, key)
        #   kind='h', key=k        → between sorted_rows[k] and [k+1]
        #   kind='v', key=(r, p)   → between placements p and p+1 in row r
        streets: list[tuple[int, str, object]] = []

        for k in range(n_rows - 1):
            streets.append((pw, 'h', k))

        for r_idx, sp in enumerate(row_placements):
            row_h = sorted_rows[r_idx].height
            for p_idx in range(len(sp) - 1):
                streets.append((row_h, 'v', (r_idx, p_idx)))

        # Longest first
        streets.sort(key=lambda s: s[0], reverse=True)

        # --- Coverage tracking ---
        # For each tile (r_idx, p_idx), track which internal sides still need
        # a street.  Sides adjacent to the page margin are pre-covered.
        needs: dict[tuple[int, int], set[str]] = {}

        for r_idx, sp in enumerate(row_placements):
            for p_idx in range(len(sp)):
                uncovered: set[str] = set()
                if r_idx > 0:
                    uncovered.add('top')
                if r_idx < n_rows - 1:
                    uncovered.add('bottom')
                if p_idx > 0:
                    uncovered.add('left')
                if p_idx < len(sp) - 1:
                    uncovered.add('right')
                if uncovered:
                    needs[(r_idx, p_idx)] = uncovered

        # --- Greedy selection ---
        selected: list[int] = []

        for length, kind, key in streets:
            if not needs:
                break  # all tiles fully covered

            # Determine which tile-sides this street covers
            covers: list[tuple[tuple[int, int], str]] = []
            if kind == 'h':
                k = key
                for p_idx in range(len(row_placements[k])):
                    covers.append(((k, p_idx), 'bottom'))
                for p_idx in range(len(row_placements[k + 1])):
                    covers.append(((k + 1, p_idx), 'top'))
            else:  # 'v'
                r_idx, p_idx = key
                covers.append(((r_idx, p_idx), 'right'))
                covers.append(((r_idx, p_idx + 1), 'left'))

            selected.append(length)

            for tile_id, side in covers:
                if tile_id in needs:
                    needs[tile_id].discard(side)
                    if not needs[tile_id]:
                        del needs[tile_id]

        self.greedy_street_lengths = selected

        if selected:
            self.greedy_street_mean = sum(selected) / len(selected)
            s = sorted(selected)
            n = len(s)
            if n % 2 == 1:
                self.greedy_street_median = float(s[n // 2])
            else:
                self.greedy_street_median = (s[n // 2 - 1] + s[n // 2]) / 2.0
        else:
            self.greedy_street_mean = 0.0
            self.greedy_street_median = 0.0

    # -- (c) Aspect-ratio fidelity ------------------------------------------

    def _compute_aspect_ratio_fidelity(self):
        """Check that each placed image preserves the source aspect ratio.

        Reports max and mean absolute error in aspect ratio.
        """
        self.ar_errors: list[tuple[int, float, float, float]] = []
        # (image_index, source_ar, placed_ar, relative_error)

        for p in self.placements:
            img = self.images[p.image_index]
            src_w = img.effective_width
            src_h = img.effective_height
            src_ar = src_w / src_h if src_h else 1.0
            placed_ar = p.width / p.height if p.height else 1.0
            rel_err = abs(placed_ar - src_ar) / src_ar if src_ar else 0.0
            self.ar_errors.append((p.image_index, src_ar, placed_ar, rel_err))

        errors = [e for _, _, _, e in self.ar_errors]
        self.ar_max_error = max(errors) if errors else 0.0
        self.ar_mean_error = sum(errors) / len(errors) if errors else 0.0

    # -- (d) Minimum spacing ------------------------------------------------

    def _compute_spacing(self):
        """Find the minimum gap between any two adjacent placed images.

        Checks both within-row (horizontal adjacency) and between-row
        (vertical adjacency).
        """
        min_h_gap = float("inf")
        min_v_gap = float("inf")

        rows = self.result.rows

        # Within-row horizontal gaps
        for row in rows:
            sorted_p = sorted(row.placements, key=lambda p: p.x)
            for i in range(len(sorted_p) - 1):
                gap = sorted_p[i + 1].x - (sorted_p[i].x + sorted_p[i].width)
                if gap < min_h_gap:
                    min_h_gap = gap

        # Between-row vertical gaps
        sorted_rows = sorted(rows, key=lambda r: r.y)
        for i in range(len(sorted_rows) - 1):
            row_bottom = sorted_rows[i].y + sorted_rows[i].height
            next_top = sorted_rows[i + 1].y
            gap = next_top - row_bottom
            if gap < min_v_gap:
                min_v_gap = gap

        self.min_horizontal_gap = min_h_gap if min_h_gap != float("inf") else 0
        self.min_vertical_gap = min_v_gap if min_v_gap != float("inf") else 0
        self.min_gap = min(
            self.min_horizontal_gap or float("inf"),
            self.min_vertical_gap or float("inf"),
        )
        if self.min_gap == float("inf"):
            self.min_gap = 0

    # -- Report -------------------------------------------------------------

    def print_report(self, verbose: bool = False):
        gap = self.settings.cut_gap
        dpi = 300

        print("=" * 68)
        print("TILER BENCHMARK REPORT")
        print("=" * 68)

        print(f"\nImages: {len(self.images)}   |   "
              f"Placed: {len(self.placements)}   |   "
              f"Rows: {len(self.result.rows)}")

        # (a) Area
        print(f"\n--- (a) Area utilisation ---")
        print(f"  Printable area:              {self.printable_area:>12,} px²")
        print(f"  Filled area:                 {self.filled_area:>12,} px²")
        print(f"  Bounding-box area:           {self.bbox_area:>12,} px²")
        print(f"  Fill ratio (of printable):   {self.fill_ratio_of_printable:>11.1%}")
        print(f"  Fill ratio (of bbox):        {self.fill_ratio_of_bbox:>11.1%}")
        print(f"  Blank ratio (of printable):  {self.blank_ratio_of_printable:>11.1%}")

        # (b) Streets
        print(f"\n--- (b) Cut-street lengths ---")
        print(f"  Horizontal streets:  {self.horizontal_street_count:>4}  "
              f"total length: {self.horizontal_street_total_length:>8,} px  "
              f"({self.horizontal_street_total_length / dpi:.1f} in)")
        print(f"  Vertical streets:    {self.vertical_street_count:>4}  "
              f"total length: {self.vertical_street_total_length:>8,} px  "
              f"({self.vertical_street_total_length / dpi:.1f} in)")
        print(f"  Total street length:        {self.total_street_length:>8,} px  "
              f"({self.total_street_length / dpi:.1f} in)")

        # (b2) Greedy street cover
        print(f"\n--- (b2) Median street length (greedy cover) ---")
        print(f"  Streets selected:        {len(self.greedy_street_lengths):>4}")
        print(f"  Mean length:           {self.greedy_street_mean:>8.0f} px  "
              f"({self.greedy_street_mean / dpi:.2f} in)")
        print(f"  Median length:         {self.greedy_street_median:>8.0f} px  "
              f"({self.greedy_street_median / dpi:.2f} in)")

        if verbose and self.greedy_street_lengths:
            print(f"\n  Selected streets (longest first):")
            for i, length in enumerate(self.greedy_street_lengths):
                print(f"    [{i + 1:>2}]  {length:>6} px  ({length / dpi:.2f} in)")

        # (c) Aspect ratio
        print(f"\n--- (c) Aspect-ratio fidelity ---")
        print(f"  Mean relative AR error:  {self.ar_mean_error:.4%}")
        print(f"  Max  relative AR error:  {self.ar_max_error:.4%}")
        ar_ok = self.ar_max_error < 0.02  # < 2 % tolerance
        print(f"  All within 2% tolerance: {'PASS' if ar_ok else 'FAIL'}")

        if verbose and self.ar_errors:
            print(f"\n  Per-image AR details:")
            for idx, src_ar, placed_ar, rel_err in self.ar_errors:
                img = self.images[idx]
                flag = " !!!" if rel_err >= 0.02 else ""
                print(f"    img[{idx:>2}] {img.pixel_width:>5}x{img.pixel_height:<5}  "
                      f"src_ar={src_ar:.3f}  placed_ar={placed_ar:.3f}  "
                      f"err={rel_err:.4%}{flag}")

        # (d) Spacing
        print(f"\n--- (d) Minimum spacing ---")
        print(f"  Configured cut gap:      {gap:>4} px  "
              f"({gap / dpi:.3f} in)")
        print(f"  Min horizontal gap:      {self.min_horizontal_gap:>4} px  "
              f"({self.min_horizontal_gap / dpi:.3f} in)")
        print(f"  Min vertical gap:        {self.min_vertical_gap:>4} px  "
              f"({self.min_vertical_gap / dpi:.3f} in)")
        spacing_ok = self.min_gap >= gap
        print(f"  All gaps >= cut gap:     {'PASS' if spacing_ok else 'FAIL'}"
              f"  (min = {self.min_gap} px)")

        if verbose:
            print(f"\n--- Placement dump ---")
            for row in self.result.rows:
                print(f"  Row y={row.y}  h={row.height}")
                for p in row.placements:
                    img = self.images[p.image_index]
                    print(f"    [{p.image_index:>2}] "
                          f"pos=({p.x},{p.y})  size={p.width}x{p.height}  "
                          f"src={img.pixel_width}x{img.pixel_height}")

        print("\n" + "=" * 68)

        return ar_ok and spacing_ok


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------

def check_overlaps(placements: list[PlacedImage]) -> list[tuple[int, int]]:
    """Return pairs of image indices whose bounding boxes overlap."""
    overlaps = []
    for i in range(len(placements)):
        a = placements[i]
        for j in range(i + 1, len(placements)):
            b = placements[j]
            # Two rects overlap iff they overlap on both axes
            if (a.x < b.x + b.width and a.x + a.width > b.x and
                    a.y < b.y + b.height and a.y + a.height > b.y):
                overlaps.append((a.image_index, b.image_index))
    return overlaps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bench(n_images: int, seed: int, verbose: bool) -> bool:
    rng = random.Random(seed)
    settings = PageSettings()  # default US Letter
    tiler = Tiler(settings)

    images = generate_images(n_images, rng)

    print(f"Generated {len(images)} images (seed={seed}):")
    for i, img in enumerate(images):
        print(f"  [{i:>2}] {img.pixel_width:>5} x {img.pixel_height:<5}  "
              f"AR={img.pixel_width / img.pixel_height:.2f}")

    result = tiler.layout(images)

    if not result.rows:
        print("\nERROR: Tiler returned empty layout.")
        return False

    # Check overlaps
    overlaps = check_overlaps(result.placements)
    if overlaps:
        print(f"\nOVERLAP DETECTED between images: {overlaps}")
    else:
        print(f"\nNo overlaps detected.  (checked {len(result.placements)} placements)")

    metrics = TilerMetrics(images, settings, result)
    passed = metrics.print_report(verbose=verbose)

    if overlaps:
        passed = False

    return passed


def main():
    parser = argparse.ArgumentParser(description="Tiler quality benchmark")
    parser.add_argument("-n", "--num-images", type=int, default=20,
                        help="Number of images to generate (default 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default 42)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-image details")
    args = parser.parse_args()

    passed = run_bench(args.num_images, args.seed, args.verbose)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
