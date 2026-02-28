"""Tiler: optimized multi-algorithm layout engine.

Arranges sticker images into rows on the page using a best-of approach
across multiple packing strategies within a configurable time budget.

Strategies:
  1. Multi-bin row packing — vary bin counts and within-bin sort orders
  2. Adaptive-height shelf packing (FFDH variant)
  3. Width-first global packing — single-height bin with global sort
  4. Best-fit row packing — minimize row-end waste per row
  5. Random perturbation search — explore nearby configurations
"""

import math
import time
import random

from models import PageSettings, StickerImage, PlacedImage, LayoutRow, LayoutResult


class Tiler:
    """Optimized layout engine with multi-algorithm best-of approach.

    Runs multiple packing strategies within *time_limit* seconds and selects
    the result that maximises area utilisation, then optimises median street
    length among near-optimal solutions (>= 90 % of best fill ratio).
    """

    def __init__(self, settings: PageSettings | None = None, time_limit: float = 1.0):
        s = settings or PageSettings()
        self.pw = s.printable_width
        self.ph = s.printable_height
        self.gap = s.cut_gap
        self.margin = s.margin
        self.time_limit = time_limit

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def layout(self, images: list[StickerImage]) -> LayoutResult:
        """Run the optimised layout pipeline on *images*."""
        if not images:
            return LayoutResult()

        deadline = time.monotonic() + self.time_limit
        ideals = self._compute_ideals(images)
        candidates: list[LayoutResult] = []

        # Fast deterministic strategies
        self._strategy_multi_bin(ideals, candidates, deadline)
        if time.monotonic() < deadline:
            self._strategy_shelf(ideals, candidates, deadline)
        if time.monotonic() < deadline:
            self._strategy_width_first(ideals, candidates, deadline)
        if time.monotonic() < deadline:
            self._strategy_best_fit(ideals, candidates, deadline)

        # Stochastic search — uses remaining time budget
        if time.monotonic() < deadline and len(ideals) > 1:
            self._strategy_permutation(ideals, candidates, deadline)

        return self._select_best(candidates) if candidates else LayoutResult()

    # ------------------------------------------------------------------ #
    #  Ideal size computation                                             #
    # ------------------------------------------------------------------ #

    def _compute_ideals(self, images):
        """Log-scale sizing: dampen resolution differences, preserve AR.

        Returns ``[(ideal_w, ideal_h, aspect_ratio, per_image_factor), ...]``.
        """
        result = []
        for img in images:
            pw, ph = img.effective_width, img.effective_height
            geo_mean = math.sqrt(pw * ph) if pw > 0 and ph > 0 else max(pw, ph, 1)
            ideal_size = math.log2(geo_mean + 1)
            aspect = pw / ph if ph > 0 else 1.0
            ih = ideal_size / math.sqrt(aspect) if aspect > 0 else ideal_size
            iw = ih * aspect
            step = getattr(img, 'scale_step', 0)
            factor = 1.2 ** step if step != 0 else 1.0
            result.append((iw, ih, aspect, factor))
        return result

    # ------------------------------------------------------------------ #
    #  Strategy 1 — Multi-bin row packing                                 #
    # ------------------------------------------------------------------ #

    def _strategy_multi_bin(self, ideals, candidates, deadline):
        n = len(ideals)
        heights = [ih for _, ih, _, _ in ideals]

        for n_bins in range(1, min(n, 7) + 1):
            if time.monotonic() > deadline:
                break
            bins = self._compute_bins(heights, n_bins)
            base_groups = self._assign_to_bins(ideals, bins)

            for sort_key in ('width_desc', 'width_asc', 'original'):
                if time.monotonic() > deadline:
                    break
                groups = self._sort_groups(base_groups, sort_key)
                result = self._scale_to_fit(groups)
                if result and result.rows:
                    candidates.append(result)

    # ------------------------------------------------------------------ #
    #  Strategy 2 — Adaptive-height shelf packing (FFDH)                  #
    # ------------------------------------------------------------------ #

    def _strategy_shelf(self, ideals, candidates, deadline):
        n = len(ideals)
        sorted_by_h = sorted(range(n), key=lambda i: ideals[i][1], reverse=True)

        for threshold in (0.50, 0.65, 0.80, 0.90):
            if time.monotonic() > deadline:
                break

            shelves: list[tuple[float, list]] = []
            for i in sorted_by_h:
                _, ih, ar, factor = ideals[i]
                placed = False
                for _, (sh, items) in enumerate(shelves):
                    if ih >= sh * threshold:
                        items.append((sh * ar, i, factor))
                        placed = True
                        break
                if not placed:
                    shelves.append((ih, [(ih * ar, i, factor)]))

            groups = self._shelves_to_groups(shelves)

            for sort_key in ('width_desc', 'original'):
                if time.monotonic() > deadline:
                    break
                result = self._scale_to_fit(self._sort_groups(groups, sort_key))
                if result and result.rows:
                    candidates.append(result)

    # ------------------------------------------------------------------ #
    #  Strategy 3 — Width-first global packing                            #
    # ------------------------------------------------------------------ #

    def _strategy_width_first(self, ideals, candidates, deadline):
        n = len(ideals)
        mean_h = sum(ih for _, ih, _, _ in ideals) / n

        sort_fns = [
            lambda i: -ideals[i][0],                        # width desc
            lambda i: -(ideals[i][0] * ideals[i][1]),       # area desc
            lambda i: ideals[i][0],                          # width asc
        ]
        for sort_fn in sort_fns:
            if time.monotonic() > deadline:
                break
            ordered = sorted(range(n), key=sort_fn)
            items = [(mean_h * ideals[i][2], i, ideals[i][3]) for i in ordered]
            result = self._scale_to_fit({mean_h: items})
            if result and result.rows:
                candidates.append(result)

    # ------------------------------------------------------------------ #
    #  Strategy 4 — Best-fit row packing                                  #
    # ------------------------------------------------------------------ #

    def _strategy_best_fit(self, ideals, candidates, deadline):
        n = len(ideals)
        heights = [ih for _, ih, _, _ in ideals]

        for n_bins in range(1, min(n, 5) + 1):
            if time.monotonic() > deadline:
                break
            bins = self._compute_bins(heights, n_bins)
            groups = self._assign_to_bins(ideals, bins)
            result = self._scale_to_fit_bf(groups)
            if result and result.rows:
                candidates.append(result)

    # ------------------------------------------------------------------ #
    #  Strategy 5 — Random perturbation search                            #
    # ------------------------------------------------------------------ #

    def _strategy_permutation(self, ideals, candidates, deadline):
        n = len(ideals)
        heights = [ih for _, ih, _, _ in ideals]
        rng = random.Random(42)
        max_bins = min(n, 7)

        best_fill = max((self._fill_ratio(r) for r in candidates), default=0)
        stale = 0

        while time.monotonic() < deadline:
            if stale > 200:
                break

            n_bins = rng.randint(1, max_bins)
            bins = self._compute_bins(heights, n_bins)
            perturbed = sorted(b * rng.uniform(0.85, 1.15) for b in bins)
            groups = self._assign_to_bins(ideals, perturbed)

            order = rng.choice(('width_desc', 'width_asc', 'random'))
            if order == 'random':
                groups = {k: rng.sample(v, len(v)) for k, v in groups.items()}
            else:
                groups = self._sort_groups(groups, order)

            if rng.random() < 0.3:
                result = self._scale_to_fit_bf(groups)
            else:
                result = self._scale_to_fit(groups)

            if result and result.rows:
                fill = self._fill_ratio(result)
                if fill > best_fill:
                    best_fill = fill
                    stale = 0
                else:
                    stale += 1
                candidates.append(result)

    # ------------------------------------------------------------------ #
    #  Scoring & selection                                                #
    # ------------------------------------------------------------------ #

    def _fill_ratio(self, result):
        """filled_area / printable_area."""
        if not result.rows:
            return 0.0
        filled = sum(p.width * p.height for p in result.placements)
        return filled / (self.pw * self.ph)

    def _median_street(self, result):
        """Median of all cut-street lengths (horizontal + vertical)."""
        if not result.rows:
            return 0.0
        streets: list[int] = []
        # Horizontal streets — full printable width between rows
        for _ in range(max(0, len(result.rows) - 1)):
            streets.append(self.pw)
        # Vertical streets — row height between adjacent images
        for row in result.rows:
            sp = sorted(row.placements, key=lambda p: p.x)
            for i in range(len(sp) - 1):
                streets.append(row.height)
        if not streets:
            return 0.0
        streets.sort()
        mid = len(streets) // 2
        if len(streets) % 2 == 1:
            return float(streets[mid])
        return (streets[mid - 1] + streets[mid]) / 2.0

    def _select_best(self, candidates):
        """Maximise fill ratio, then median street among solutions >= 90 % of best."""
        if not candidates:
            return LayoutResult()
        if len(candidates) == 1:
            return candidates[0]

        scored = [(self._fill_ratio(r), r) for r in candidates]
        best_fill = max(f for f, _ in scored)
        threshold = best_fill * 0.9
        good = [(f, r) for f, r in scored if f >= threshold]

        if not good:
            return max(scored, key=lambda x: x[0])[1]
        # Among near-optimal, prefer higher median street, then higher fill
        return max(good, key=lambda x: (self._median_street(x[1]), x[0]))[1]

    # ------------------------------------------------------------------ #
    #  Grouping helpers                                                   #
    # ------------------------------------------------------------------ #

    def _assign_to_bins(self, ideals, bins):
        """Assign each image to the nearest bin height, returning groups dict."""
        groups: dict[float, list[tuple[float, int, float]]] = {}
        for i, (iw, ih, _ar, factor) in enumerate(ideals):
            best_bin = min(bins, key=lambda b: abs(b - ih))
            scaled_w = iw * (best_bin / ih) if ih > 0 else iw
            groups.setdefault(best_bin, []).append((scaled_w, i, factor))
        return groups

    def _sort_groups(self, groups, sort_key):
        """Return a shallow copy of *groups* with items sorted."""
        out: dict[float, list] = {}
        for bin_h, items in groups.items():
            if sort_key == 'width_desc':
                out[bin_h] = sorted(items, key=lambda x: x[0], reverse=True)
            elif sort_key == 'width_asc':
                out[bin_h] = sorted(items, key=lambda x: x[0])
            else:
                out[bin_h] = list(items)
        return out

    @staticmethod
    def _shelves_to_groups(shelves):
        """Convert shelf list to groups dict with unique keys."""
        groups: dict[float, list] = {}
        uid = 0
        for sh, items in shelves:
            key = sh + uid * 1e-10
            uid += 1
            groups[key] = items
        return groups

    def _compute_bins(self, heights: list[float], n_bins: int) -> list[float]:
        """Split sorted heights into *n_bins* groups, return median of each."""
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

    def _bin_count(self, n: int) -> int:
        """Choose number of height bins based on image count (legacy helper)."""
        if n <= 2:
            return 1
        if n <= 5:
            return 2
        if n <= 10:
            return 3
        if n <= 20:
            return 4
        return 5

    # ------------------------------------------------------------------ #
    #  Core packing — left-to-right (standard)                            #
    # ------------------------------------------------------------------ #

    def _scale_to_fit(self, groups: dict) -> LayoutResult:
        """Binary-search for the largest scale that fits the page (L-R pack)."""
        if not groups:
            return LayoutResult()
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
        """Left-to-right row packing at *scale*.  Returns ``None`` if overflow."""
        rows: list[tuple[int, list]] = []

        for bin_h in sorted(groups.keys(), reverse=True):
            current: list[tuple[int, int, int, int]] = []
            cx = 0

            for ideal_w, idx, img_factor in groups[bin_h]:
                w = max(1, int(ideal_w * scale * img_factor))
                h = max(1, int(bin_h * scale * img_factor))
                if w > self.pw:
                    h = max(1, int(h * self.pw / w))
                    w = self.pw

                needed = cx + (self.gap if current else 0) + w
                if needed > self.pw and current:
                    rows.append((max(ih for _, _, ih, _ in current), current))
                    current = []
                    cx = 0

                if current:
                    cx += self.gap
                current.append((cx, w, h, idx))
                cx += w

            if current:
                rows.append((max(ih for _, _, ih, _ in current), current))

        total = sum(rh for rh, _ in rows) + self.gap * max(0, len(rows) - 1)
        if total > self.ph:
            return None
        return self._build_layout(rows)

    # ------------------------------------------------------------------ #
    #  Core packing — best-fit (minimise row-end waste)                   #
    # ------------------------------------------------------------------ #

    def _scale_to_fit_bf(self, groups: dict) -> LayoutResult:
        """Binary-search for the largest scale that fits (best-fit pack)."""
        if not groups:
            return LayoutResult()
        max_bin = max(groups.keys())
        scale_hi = (self.ph * 0.6) / max_bin if max_bin > 0 else 100.0
        scale_lo = 0.1

        best = None
        for _ in range(40):
            mid = (scale_hi + scale_lo) / 2
            result = self._try_pack_bf(groups, mid)
            if result is not None:
                best = result
                scale_lo = mid
            else:
                scale_hi = mid

        if best is None:
            best = self._try_pack_bf(groups, scale_lo) or LayoutResult()
        return best

    def _try_pack_bf(self, groups: dict, scale: float) -> LayoutResult | None:
        """Best-fit row packing: greedily pick image that leaves least waste."""
        rows: list[tuple[int, list]] = []

        for bin_h in sorted(groups.keys(), reverse=True):
            remaining = list(range(len(groups[bin_h])))
            bin_items = groups[bin_h]

            while remaining:
                current: list[tuple[int, int, int, int]] = []
                cx = 0
                improved = True

                while improved:
                    improved = False
                    best_j_pos = -1
                    min_leftover = self.pw + 1

                    for j_pos, j in enumerate(remaining):
                        ideal_w, idx, img_factor = bin_items[j]
                        w = max(1, int(ideal_w * scale * img_factor))
                        if w > self.pw:
                            w = self.pw
                        needed = cx + (self.gap if current else 0) + w
                        if needed <= self.pw:
                            leftover = self.pw - needed
                            if leftover < min_leftover:
                                min_leftover = leftover
                                best_j_pos = j_pos

                    if best_j_pos >= 0:
                        j = remaining.pop(best_j_pos)
                        ideal_w, idx, img_factor = bin_items[j]
                        w = max(1, int(ideal_w * scale * img_factor))
                        h = max(1, int(bin_h * scale * img_factor))
                        if w > self.pw:
                            h = max(1, int(h * self.pw / w))
                            w = self.pw
                        if current:
                            cx += self.gap
                        current.append((cx, w, h, idx))
                        cx += w
                        improved = True

                if current:
                    rows.append((max(ih for _, _, ih, _ in current), current))

        total = sum(rh for rh, _ in rows) + self.gap * max(0, len(rows) - 1)
        if total > self.ph:
            return None
        return self._build_layout(rows)

    # ------------------------------------------------------------------ #
    #  Layout builder                                                     #
    # ------------------------------------------------------------------ #

    def _build_layout(self, rows: list[tuple[int, list]]) -> LayoutResult:
        """Convert raw ``(row_h, [(x, w, h, idx), ...])`` into a LayoutResult."""
        layout_rows: list[LayoutRow] = []
        y = self.margin
        for row_h, items in rows:
            placements = [
                PlacedImage(image_index=idx, x=self.margin + x, y=y, width=w, height=h)
                for x, w, h, idx in items
            ]
            layout_rows.append(LayoutRow(y=y, height=row_h, placements=placements))
            y += row_h + self.gap
        return LayoutResult(rows=layout_rows)
