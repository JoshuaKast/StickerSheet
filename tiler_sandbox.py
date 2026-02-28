#!/usr/bin/env python3
"""
Sandbox for developing and fuzz-testing tiling algorithms.

Generates random image sets, runs them through a tiler, and measures:
  - Fill ratio: image area / printable area
  - Street lengths: median uninterrupted cut line length
  - Aspect ratio preservation
  - Street gap compliance
"""
import math
import random
import statistics
from dataclasses import dataclass, field

# Page constants (300 DPI, US Letter)
DPI = 300
PAGE_WIDTH = int(8.5 * DPI)
PAGE_HEIGHT = int(11.0 * DPI)
DEFAULT_MARGIN = int(0.25 * DPI)
DEFAULT_GAP = max(1, int(0.125 * DPI))  # 37px at 300 DPI

PRINTABLE_WIDTH = PAGE_WIDTH - 2 * DEFAULT_MARGIN
PRINTABLE_HEIGHT = PAGE_HEIGHT - 2 * DEFAULT_MARGIN


@dataclass
class ImageSpec:
    """Input image specification."""
    width: int
    height: int
    scale_step: int = 0  # user scaling suggestion

    @property
    def aspect(self):
        return self.width / self.height if self.height > 0 else 1.0


@dataclass
class PlacedRect:
    """A placed image rectangle on the page."""
    index: int
    x: int
    y: int
    width: int
    height: int
    rotated: bool = False


@dataclass
class PackResult:
    """Result of a packing attempt."""
    placements: list[PlacedRect] = field(default_factory=list)
    page_width: int = PAGE_WIDTH
    page_height: int = PAGE_HEIGHT
    margin: int = DEFAULT_MARGIN
    gap: int = DEFAULT_GAP

    @property
    def printable_width(self):
        return self.page_width - 2 * self.margin

    @property
    def printable_height(self):
        return self.page_height - 2 * self.margin


# === Metrics ===

def fill_ratio(result: PackResult) -> float:
    """Ratio of total image area to printable area."""
    if not result.placements:
        return 0.0
    img_area = sum(p.width * p.height for p in result.placements)
    printable_area = result.printable_width * result.printable_height
    return img_area / printable_area if printable_area > 0 else 0.0


def aspect_ratio_errors(specs: list[ImageSpec], result: PackResult) -> list[float]:
    """For each placed image, compute aspect ratio error vs original."""
    errors = []
    for p in result.placements:
        orig = specs[p.index]
        orig_aspect = orig.aspect
        placed_aspect = p.width / p.height if p.height > 0 else 1.0
        if p.rotated:
            orig_aspect = 1.0 / orig_aspect if orig_aspect > 0 else 1.0
        error = abs(placed_aspect - orig_aspect) / orig_aspect if orig_aspect > 0 else 0.0
        errors.append(error)
    return errors


def street_lengths(result: PackResult) -> list[float]:
    """Compute all uninterrupted street (gap) segment lengths.

    A street is a horizontal or vertical line segment in the gap space
    between images. We scan for maximal continuous gap segments.
    """
    if len(result.placements) < 2:
        return [float(result.printable_width)]  # one full-width street above/below

    margin = result.margin
    pw = result.printable_width
    ph = result.printable_height

    rects = [(p.x, p.y, p.x + p.width, p.y + p.height) for p in result.placements]

    # Collect all unique y-coordinates and x-coordinates for scanning
    y_vals = sorted(set(
        [margin] + [r[1] for r in rects] + [r[3] for r in rects] + [margin + ph]
    ))
    x_vals = sorted(set(
        [margin] + [r[0] for r in rects] + [r[2] for r in rects] + [margin + pw]
    ))

    streets = []

    # Horizontal streets: scan at each y boundary
    for y in y_vals:
        # Find segments along this y-line not covered by any image
        # A horizontal line at y is blocked by image if image.y < y < image.y+height
        # and image.x <= segment_x < image.x+width
        blockers = []
        for x1, y1, x2, y2 in rects:
            # Image blocks this y-line if its vertical span strictly contains y
            # (at boundaries, the gap exists)
            if y1 < y < y2:
                blockers.append((x1, x2))

        # Merge blockers and find free segments
        if not blockers:
            streets.append(float(pw))
            continue

        blockers.sort()
        merged = [blockers[0]]
        for s, e in blockers[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Free segments
        prev_end = margin
        for s, e in merged:
            if s > prev_end:
                streets.append(float(s - prev_end))
            prev_end = max(prev_end, e)
        if prev_end < margin + pw:
            streets.append(float(margin + pw - prev_end))

    # Vertical streets: scan at each x boundary
    for x in x_vals:
        blockers = []
        for x1, y1, x2, y2 in rects:
            if x1 < x < x2:
                blockers.append((y1, y2))

        if not blockers:
            streets.append(float(ph))
            continue

        blockers.sort()
        merged = [blockers[0]]
        for s, e in blockers[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        prev_end = margin
        for s, e in merged:
            if s > prev_end:
                streets.append(float(s - prev_end))
            prev_end = max(prev_end, e)
        if prev_end < margin + ph:
            streets.append(float(margin + ph - prev_end))

    return streets if streets else [0.0]


def median_street_length(result: PackResult) -> float:
    lengths = street_lengths(result)
    return statistics.median(lengths) if lengths else 0.0


def evaluate(specs: list[ImageSpec], result: PackResult) -> dict:
    """Compute all metrics for a packing result."""
    ar_errors = aspect_ratio_errors(specs, result)
    sl = street_lengths(result)
    return {
        'fill_ratio': fill_ratio(result),
        'max_aspect_error': max(ar_errors) if ar_errors else 0.0,
        'mean_aspect_error': statistics.mean(ar_errors) if ar_errors else 0.0,
        'median_street_length': statistics.median(sl) if sl else 0.0,
        'mean_street_length': statistics.mean(sl) if sl else 0.0,
        'min_street_length': min(sl) if sl else 0.0,
        'num_placements': len(result.placements),
        'num_images': len(specs),
    }


# === Random image generators ===

def random_specs(n: int, seed: int | None = None) -> list[ImageSpec]:
    """Generate n random ImageSpecs with varied sizes and aspect ratios."""
    rng = random.Random(seed)
    specs = []
    for _ in range(n):
        # Random pixel dimensions: 50-4000px
        w = rng.randint(50, 4000)
        h = rng.randint(50, 4000)
        scale = rng.choice([-2, -1, 0, 0, 0, 0, 1, 2])
        specs.append(ImageSpec(w, h, scale))
    return specs


def uniform_specs(n: int, w: int = 300, h: int = 300) -> list[ImageSpec]:
    """Generate n identical ImageSpecs."""
    return [ImageSpec(w, h) for _ in range(n)]


def varied_aspect_specs(n: int, seed: int | None = None) -> list[ImageSpec]:
    """Generate n images with deliberately varied aspect ratios."""
    rng = random.Random(seed)
    specs = []
    aspects = [0.25, 0.5, 0.75, 1.0, 1.0, 1.33, 2.0, 4.0]
    for _ in range(n):
        aspect = rng.choice(aspects)
        base = rng.randint(100, 2000)
        w = int(base * math.sqrt(aspect))
        h = int(base / math.sqrt(aspect))
        specs.append(ImageSpec(max(1, w), max(1, h)))
    return specs


# === Adapter: run existing Tiler through sandbox ===

def run_old_tiler(specs: list[ImageSpec], margin=DEFAULT_MARGIN, gap=DEFAULT_GAP) -> PackResult:
    """Run the existing Tiler from sticker_app and convert results."""
    from sticker_app import Tiler, StickerImage, PageSettings

    settings = PageSettings()
    settings.margin_inches = margin / DPI
    settings.street_width_inches = gap / DPI

    images = [StickerImage(png_data=b'', pixel_width=s.width, pixel_height=s.height,
                           scale_step=s.scale_step) for s in specs]
    tiler = Tiler(settings)
    layout = tiler.layout(images)

    placements = []
    for p in layout.placements:
        placements.append(PlacedRect(
            index=p.image_index, x=p.x, y=p.y,
            width=p.width, height=p.height, rotated=False
        ))

    return PackResult(placements=placements, margin=margin, gap=gap)


# === Run benchmark ===

def benchmark(tiler_fn, label="tiler", num_trials=20, images_per_trial=10, seed=42):
    """Run multiple random trials and report aggregate metrics."""
    rng = random.Random(seed)
    results = []
    for trial in range(num_trials):
        trial_seed = rng.randint(0, 10**9)
        specs = random_specs(images_per_trial, seed=trial_seed)
        result = tiler_fn(specs)
        metrics = evaluate(specs, result)
        metrics['trial'] = trial
        results.append(metrics)

    # Aggregate
    fill_ratios = [r['fill_ratio'] for r in results]
    med_streets = [r['median_street_length'] for r in results]
    ar_errors = [r['max_aspect_error'] for r in results]

    print(f"\n{'='*60}")
    print(f" {label} — {num_trials} trials × {images_per_trial} images")
    print(f"{'='*60}")
    print(f" Fill ratio:    mean={statistics.mean(fill_ratios):.3f}  "
          f"min={min(fill_ratios):.3f}  max={max(fill_ratios):.3f}")
    print(f" Median street: mean={statistics.mean(med_streets):.1f}px  "
          f"min={min(med_streets):.1f}  max={max(med_streets):.1f}")
    print(f" AR error:      mean={statistics.mean(ar_errors):.4f}  "
          f"max={max(ar_errors):.4f}")
    print(f"{'='*60}")
    return results


def run_new_tiler(specs: list[ImageSpec], margin=DEFAULT_MARGIN, gap=DEFAULT_GAP) -> PackResult:
    """Run the new integrated optimal tiler from sticker_app."""
    from sticker_app import Tiler, StickerImage, PageSettings

    settings = PageSettings()
    settings.margin_inches = margin / DPI
    settings.street_width_inches = gap / DPI

    images = [StickerImage(png_data=b'', pixel_width=s.width, pixel_height=s.height,
                           scale_step=s.scale_step) for s in specs]
    tiler = Tiler(settings)
    layout = tiler.layout(images)

    placements = []
    for p in layout.placements:
        placements.append(PlacedRect(
            index=p.image_index, x=p.x, y=p.y,
            width=p.width, height=p.height,
            rotated=getattr(p, 'rotated', False)
        ))

    return PackResult(placements=placements, margin=margin, gap=gap)


if __name__ == '__main__':
    print("Benchmarking OLD tiler...")
    old_results = benchmark(run_old_tiler, label="Old Tiler (row-based)",
                            num_trials=30, images_per_trial=12, seed=42)

    print("\nBenchmarking NEW tiler...")
    new_results = benchmark(run_new_tiler, label="New Optimal Tiler",
                            num_trials=30, images_per_trial=12, seed=42)

    # Additional stress tests
    for n_imgs in [3, 6, 20, 30]:
        print(f"\n--- {n_imgs} images per trial ---")
        benchmark(run_old_tiler, label=f"Old ({n_imgs} imgs)",
                  num_trials=20, images_per_trial=n_imgs, seed=99)
        benchmark(run_new_tiler, label=f"New ({n_imgs} imgs)",
                  num_trials=20, images_per_trial=n_imgs, seed=99)
