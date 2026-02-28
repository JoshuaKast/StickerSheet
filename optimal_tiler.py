"""
Optimal tiling algorithm for StickerSheet.

Priorities (in order):
  a) Maintain aspect ratio (rotation by 90deg allowed if it helps packing)
  a1) Maintain minimum street spacing
  b) Maximize image area / empty area ratio
  c) Respect user scale_step suggestions
  d) Maximize median uninterrupted street length

Algorithm: Multi-strategy shelf packing with rotation.

Phase 1: Log-scale ideal sizing (proven, kept from old tiler)
Phase 2: For each image, pre-compute both orientations
Phase 3: Try multiple packing strategies, pick best by fill ratio:
  - Strategy A: Height-binned shelf packing (like old, but with rotation + cross-bin mixing)
  - Strategy B: Sorted-decreasing-height shelf packing (no bins, pure greedy)
  - Strategy C: Best-fit-decreasing shelf packing (fill row gaps greedily)
Phase 4: Scale-to-fit via binary search (shared)
"""
import math
from dataclasses import dataclass, field


DPI = 300
DEFAULT_PAGE_WIDTH = int(8.5 * DPI)
DEFAULT_PAGE_HEIGHT = int(11.0 * DPI)
DEFAULT_MARGIN = int(0.25 * DPI)
DEFAULT_GAP = max(1, int(0.125 * DPI))


@dataclass
class _Img:
    """Image during layout computation."""
    index: int
    iw: float          # ideal width (normal orientation)
    ih: float          # ideal height (normal orientation)
    factor: float      # user scale factor (1.2 ** scale_step)

    @property
    def iw_rot(self):
        return self.ih

    @property
    def ih_rot(self):
        return self.iw


@dataclass
class PlacedRect:
    index: int
    x: int
    y: int
    width: int
    height: int
    rotated: bool = False


@dataclass
class LayoutRow:
    y: int
    height: int
    placements: list[PlacedRect] = field(default_factory=list)


@dataclass
class PackResult:
    rows: list[LayoutRow] = field(default_factory=list)

    @property
    def placements(self):
        return [p for row in self.rows for p in row.placements]


def _log_scale(pw: int, ph: int) -> tuple[float, float]:
    """Convert pixel dimensions to ideal sizes via log-scale on geometric mean."""
    geo_mean = math.sqrt(pw * ph) if pw > 0 and ph > 0 else max(pw, ph, 1)
    ideal_size = math.log2(geo_mean + 1)
    aspect = pw / ph if ph > 0 else 1.0
    ih = ideal_size / math.sqrt(aspect) if aspect > 0 else ideal_size
    iw = ih * aspect
    return iw, ih


def _fill_ratio_quick(placements, pw, ph):
    if not placements or pw <= 0 or ph <= 0:
        return 0.0
    return sum(p.width * p.height for p in placements) / (pw * ph)


def _clamp_to_page(w: int, h: int, pw: int) -> tuple[int, int]:
    """If w exceeds page width, scale both dimensions down proportionally."""
    if w > pw and w > 0:
        ratio = pw / w
        h = max(1, int(h * ratio))
        w = pw
    return w, h


# === Strategy A: Binned shelf packing with rotation ===

def _kmeans_1d(values: list[float], k: int, iterations: int = 20) -> list[float]:
    """1D k-means to find bin centers."""
    if k >= len(values):
        result = sorted(set(values))
        return result if result else [1.0]

    sv = sorted(values)
    step = len(sv) / (k + 1)
    centers = [sv[min(int((i + 1) * step), len(sv) - 1)] for i in range(k)]

    for _ in range(iterations):
        clusters = [[] for _ in range(k)]
        for v in sv:
            best = min(range(k), key=lambda c: abs(v - centers[c]))
            clusters[best].append(v)
        centers = [
            (sum(cl) / len(cl)) if cl else centers[i]
            for i, cl in enumerate(clusters)
        ]

    return sorted(set(round(c, 6) for c in centers)) or [1.0]


def _pack_binned(images: list[_Img], bins: list[float], scale: float,
                 pw: int, ph: int, gap: int, margin: int) -> PackResult | None:
    """Binned shelf packing: assign images to height bins, then pack rows.
    Allows rotation if it puts image in a better bin.
    After initial row packing, attempts to fill row remainders with unplaced images."""

    # Assign to bins, choosing orientation that minimizes bin distance
    # (index, ideal_w_scaled, bin_h, factor, rotated)
    assigned = []
    for img in images:
        best_bin = None
        best_sw = None
        best_rot = False
        best_dist = float('inf')

        for b in bins:
            # Normal
            dist_n = abs(b - img.ih)
            sw_n = img.iw * (b / img.ih) if img.ih > 0 else img.iw
            # Rotated
            dist_r = abs(b - img.ih_rot)
            sw_r = img.iw_rot * (b / img.ih_rot) if img.ih_rot > 0 else img.iw_rot

            if dist_n <= dist_r:
                if dist_n < best_dist:
                    best_dist = dist_n
                    best_bin = b
                    best_sw = sw_n
                    best_rot = False
            else:
                if dist_r < best_dist:
                    best_dist = dist_r
                    best_bin = b
                    best_sw = sw_r
                    best_rot = True

        assigned.append((img.index, best_sw, best_bin, img.factor, best_rot))

    # Sort by (bin_h desc, scaled_width desc) for better packing
    assigned.sort(key=lambda t: (-t[2], -t[1] * t[3]))

    # Pack into rows greedily
    rows = []  # list of (row_h, [(x, w, h, idx, rot), ...])
    cx = 0
    current = []
    current_h = 0

    for idx, sw, bh, factor, rot in assigned:
        w = max(1, int(sw * scale * factor))
        h = max(1, int(bh * scale * factor))
        w, h = _clamp_to_page(w, h, pw)

        needed = cx + (gap if current else 0) + w
        if needed > pw and current:
            rows.append((current_h, current))
            current = []
            cx = 0
            current_h = 0

        if current:
            cx += gap
        current.append((cx, w, h, idx, rot))
        cx += w
        current_h = max(current_h, h)

    if current:
        rows.append((current_h, current))

    # Vertical fit check
    total = sum(rh for rh, _ in rows) + gap * max(0, len(rows) - 1)
    if total > ph:
        return None

    # Build result
    layout_rows = []
    y = margin
    for rh, items in rows:
        placements = [
            PlacedRect(index=idx, x=margin + x, y=y, width=w, height=h, rotated=rot)
            for x, w, h, idx, rot in items
        ]
        layout_rows.append(LayoutRow(y=y, height=rh, placements=placements))
        y += rh + gap

    return PackResult(rows=layout_rows)


# === Strategy B: Sorted decreasing height, pure shelf ===

def _pack_sorted_shelves(images: list[_Img], scale: float,
                         pw: int, ph: int, gap: int, margin: int,
                         allow_rotation: bool = True) -> PackResult | None:
    """Sort images by height (descending), pack into shelves left-to-right.
    Each shelf height = tallest image in that shelf."""

    # For each image, pick orientation
    items = []  # (index, iw, ih, factor, rotated)
    for img in images:
        if allow_rotation and img.iw > img.ih:
            # Landscape image: try rotating to portrait (taller) for better shelf fill
            # Only rotate if it makes the image taller (helps create tighter shelves)
            items.append((img.index, img.ih, img.iw, img.factor, True))
        else:
            items.append((img.index, img.iw, img.ih, img.factor, False))

    # Sort by effective height descending
    items.sort(key=lambda t: -t[2] * t[3])

    rows = []
    cx = 0
    current = []
    current_h = 0

    for idx, iw, ih, factor, rot in items:
        w = max(1, int(iw * scale * factor))
        h = max(1, int(ih * scale * factor))
        w, h = _clamp_to_page(w, h, pw)

        needed = cx + (gap if current else 0) + w
        if needed > pw and current:
            rows.append((current_h, current))
            current = []
            cx = 0
            current_h = 0

        if current:
            cx += gap
        current.append((cx, w, h, idx, rot))
        cx += w
        current_h = max(current_h, h)

    if current:
        rows.append((current_h, current))

    total = sum(rh for rh, _ in rows) + gap * max(0, len(rows) - 1)
    if total > ph:
        return None

    layout_rows = []
    y = margin
    for rh, items_in_row in rows:
        placements = [
            PlacedRect(index=idx, x=margin + x, y=y, width=w, height=h, rotated=rot)
            for x, w, h, idx, rot in items_in_row
        ]
        layout_rows.append(LayoutRow(y=y, height=rh, placements=placements))
        y += rh + gap

    return PackResult(rows=layout_rows)


# === Strategy C: Best-fit decreasing with rotation choice per-row ===

def _pack_bestfit(images: list[_Img], scale: float,
                  pw: int, ph: int, gap: int, margin: int) -> PackResult | None:
    """Best-fit decreasing: for each shelf, find the best image to fill remaining width.

    This tries to minimize wasted width in each row.
    For each image, both orientations are considered, choosing the one that
    best fits the current row's remaining width."""

    # Pre-compute pixel sizes for both orientations at this scale
    img_data = []
    for img in images:
        wn = max(1, int(img.iw * scale * img.factor))
        hn = max(1, int(img.ih * scale * img.factor))
        wr = max(1, int(img.iw_rot * scale * img.factor))
        hr = max(1, int(img.ih_rot * scale * img.factor))
        wn, hn = _clamp_to_page(wn, hn, pw)
        wr, hr = _clamp_to_page(wr, hr, pw)
        img_data.append({
            'index': img.index, 'factor': img.factor,
            'wn': wn, 'hn': hn, 'wr': wr, 'hr': hr,
        })

    # Sort by max(height) descending as starting order
    order = sorted(range(len(img_data)), key=lambda i: -max(img_data[i]['hn'], img_data[i]['hr']))
    placed = [False] * len(img_data)

    rows = []

    # Start rows with the tallest unplaced image
    for start_idx in order:
        if placed[start_idx]:
            continue

        d = img_data[start_idx]
        # Choose orientation: prefer the one that makes it wider (fills row better)
        if d['wn'] >= d['wr']:
            rw, rh, rot = d['wn'], d['hn'], False
        else:
            rw, rh, rot = d['wr'], d['hr'], True

        row_items = [(0, rw, rh, d['index'], rot)]
        cx = rw
        row_h = rh
        placed[start_idx] = True

        # Fill remaining row width with best-fitting unplaced images
        changed = True
        while changed:
            changed = False
            remaining = pw - cx - gap  # available width
            if remaining <= 0:
                break

            best_j = -1
            best_waste = remaining + 1  # minimize wasted space
            best_w = 0
            best_h = 0
            best_rot = False

            for j in range(len(img_data)):
                if placed[j]:
                    continue
                dj = img_data[j]
                # Try normal
                if dj['wn'] <= remaining and dj['hn'] <= row_h * 1.05:
                    waste = remaining - dj['wn']
                    if waste < best_waste:
                        best_waste = waste
                        best_j = j
                        best_w = dj['wn']
                        best_h = dj['hn']
                        best_rot = False
                # Try rotated
                if dj['wr'] <= remaining and dj['hr'] <= row_h * 1.05:
                    waste = remaining - dj['wr']
                    if waste < best_waste:
                        best_waste = waste
                        best_j = j
                        best_w = dj['wr']
                        best_h = dj['hr']
                        best_rot = True

            if best_j >= 0:
                placed[best_j] = True
                cx += gap
                row_items.append((cx, best_w, best_h, img_data[best_j]['index'], best_rot))
                cx += best_w
                row_h = max(row_h, best_h)
                changed = True

        rows.append((row_h, row_items))

    total = sum(rh for rh, _ in rows) + gap * max(0, len(rows) - 1)
    if total > ph:
        return None

    layout_rows = []
    y = margin
    for rh, items_in_row in rows:
        placements = [
            PlacedRect(index=idx, x=margin + x, y=y, width=w, height=h, rotated=rot)
            for x, w, h, idx, rot in items_in_row
        ]
        layout_rows.append(LayoutRow(y=y, height=rh, placements=placements))
        y += rh + gap

    return PackResult(rows=layout_rows)


# === Post-processing: stretch rows to fill width ===

def _stretch_rows(result: PackResult, pw: int, margin: int, gap: int) -> PackResult:
    """Proportionally stretch images in each row to fill the full row width.

    This improves fill ratio and creates full-page-width horizontal streets.
    Each image's width is increased proportionally; height is adjusted
    to maintain aspect ratio.
    """
    if not result.rows:
        return result

    new_rows = []
    for row in result.rows:
        if not row.placements:
            new_rows.append(row)
            continue

        # Current total image width and gap count
        n_imgs = len(row.placements)
        total_img_w = sum(p.width for p in row.placements)
        total_gaps = gap * max(0, n_imgs - 1)
        available = pw - total_gaps

        if total_img_w <= 0 or available <= 0:
            new_rows.append(row)
            continue

        stretch_factor = available / total_img_w

        # Only stretch if row is already reasonably full (>80% filled)
        # and limit stretch to 8% to maintain aspect ratio fidelity
        if stretch_factor > 1.08 or stretch_factor < 1.0:
            new_rows.append(row)
            continue

        new_placements = []
        cx = 0
        new_row_h = 0
        for p in row.placements:
            new_w = max(1, int(p.width * stretch_factor))
            # Scale height proportionally to maintain aspect ratio
            new_h = max(1, int(p.height * stretch_factor))
            new_placements.append(PlacedRect(
                index=p.index, x=margin + cx, y=row.y,
                width=new_w, height=new_h, rotated=p.rotated
            ))
            cx += new_w + gap
            new_row_h = max(new_row_h, new_h)

        new_rows.append(LayoutRow(y=row.y, height=new_row_h, placements=new_placements))

    return PackResult(rows=new_rows)


# === Scoring function for candidate comparison ===

def _compute_street_lengths(result: PackResult, pw: int, ph: int, margin: int) -> list[float]:
    """Compute all street segment lengths for a packing result."""
    if len(result.placements) < 2:
        return [float(pw)]

    rects = [(p.x, p.y, p.x + p.width, p.y + p.height) for p in result.placements]

    y_vals = sorted(set(
        [margin] + [r[1] for r in rects] + [r[3] for r in rects] + [margin + ph]
    ))
    x_vals = sorted(set(
        [margin] + [r[0] for r in rects] + [r[2] for r in rects] + [margin + pw]
    ))

    streets = []

    for y in y_vals:
        blockers = [(x1, x2) for x1, y1, x2, y2 in rects if y1 < y < y2]
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
        prev = margin
        for s, e in merged:
            if s > prev:
                streets.append(float(s - prev))
            prev = max(prev, e)
        if prev < margin + pw:
            streets.append(float(margin + pw - prev))

    for x in x_vals:
        blockers = [(y1, y2) for x1, y1, x2, y2 in rects if x1 < x < x2]
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
        prev = margin
        for s, e in merged:
            if s > prev:
                streets.append(float(s - prev))
            prev = max(prev, e)
        if prev < margin + ph:
            streets.append(float(margin + ph - prev))

    return streets or [0.0]


def _score_result(result: PackResult, pw: int, ph: int, margin: int, gap: int) -> float:
    """Score a packing result. Higher is better.

    Fill ratio is priority (b), street length is priority (d).
    Weight: 80% fill ratio, 20% normalized median street length.
    """
    if not result.placements:
        return 0.0

    fr = _fill_ratio_quick(result.placements, pw, ph)

    streets = _compute_street_lengths(result, pw, ph, margin)
    import statistics
    med_street = statistics.median(streets) if streets else 0.0
    # Normalize: max possible street = max(pw, ph)
    max_possible = max(pw, ph)
    street_norm = med_street / max_possible if max_possible > 0 else 0.0

    return fr * 0.8 + street_norm * 0.2


# === Scale-to-fit wrapper ===

def _scale_to_fit(pack_fn, pw, ph, max_ideal_h, **pack_kwargs) -> PackResult:
    """Binary search for largest scale that fits the page."""
    scale_hi = (ph * 0.85) / max_ideal_h if max_ideal_h > 0 else 100.0
    scale_lo = 0.1
    best = None

    for _ in range(50):
        mid = (scale_hi + scale_lo) / 2
        result = pack_fn(scale=mid, **pack_kwargs)
        if result is not None:
            best = result
            scale_lo = mid
        else:
            scale_hi = mid

    if best is None:
        best = pack_fn(scale=scale_lo, **pack_kwargs) or PackResult()
    return best


# === Main entry point ===

def optimal_layout(
    widths: list[int],
    heights: list[int],
    scale_steps: list[int] | None = None,
    page_width: int = None,
    page_height: int = None,
    margin: int = None,
    gap: int = None,
) -> PackResult:
    n = len(widths)
    if n == 0:
        return PackResult()

    pw_total = page_width or DEFAULT_PAGE_WIDTH
    ph_total = page_height or DEFAULT_PAGE_HEIGHT
    m = margin if margin is not None else DEFAULT_MARGIN
    g = gap if gap is not None else DEFAULT_GAP

    pw = pw_total - 2 * m
    ph = ph_total - 2 * m

    if scale_steps is None:
        scale_steps = [0] * n

    # Step 1: Log-scale sizing
    imgs = []
    for i in range(n):
        iw, ih = _log_scale(widths[i], heights[i])
        step = scale_steps[i]
        factor = 1.2 ** step if step != 0 else 1.0
        imgs.append(_Img(index=i, iw=iw, ih=ih, factor=factor))

    max_ih = max(max(img.ih, img.iw) for img in imgs)  # worst case height

    candidates = []

    # Strategy A: Binned packing with multiple bin counts
    ideal_heights = [img.ih for img in imgs]
    all_heights = ideal_heights + [img.iw for img in imgs]

    if n <= 2:
        bin_range = [1, 2]
    elif n <= 5:
        bin_range = [1, 2, 3]
    elif n <= 10:
        bin_range = [2, 3, 4]
    elif n <= 20:
        bin_range = [3, 4, 5]
    else:
        bin_range = [3, 4, 5, 6]

    for nb in bin_range:
        for heights_src in [ideal_heights, all_heights]:
            bins = _kmeans_1d(heights_src, nb)
            result = _scale_to_fit(
                lambda scale, _bins=bins: _pack_binned(imgs, _bins, scale, pw, ph, g, m),
                pw, ph, max_ih)
            candidates.append(result)

    # Strategy B: Sorted shelves (with and without rotation)
    for allow_rot in [True, False]:
        result = _scale_to_fit(
            lambda scale, _ar=allow_rot: _pack_sorted_shelves(
                imgs, scale, pw, ph, g, m, allow_rotation=_ar),
            pw, ph, max_ih)
        candidates.append(result)

    # Strategy C: Best-fit decreasing
    result = _scale_to_fit(
        lambda scale: _pack_bestfit(imgs, scale, pw, ph, g, m),
        pw, ph, max_ih)
    candidates.append(result)

    # Apply row stretching to all candidates
    stretched = []
    for c in candidates:
        if c and c.placements:
            stretched.append(_stretch_rows(c, pw, m, g))
        else:
            stretched.append(c)

    # Pick the candidate with best combined score
    best = None
    best_score = -1.0
    for c in stretched:
        if c and c.placements:
            sc = _score_result(c, pw, ph, m, g)
            if sc > best_score:
                best_score = sc
                best = c

    return best or PackResult()
