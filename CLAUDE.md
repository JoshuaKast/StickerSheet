# Sticker Sheet Maker

A PySide6 desktop app for tiling pasted images onto a US Letter page for printing as sticker sheets.

## Concept

The user browses image sources (DuckDuckGo Images, etc.) in a browser, copies images to the clipboard, and pastes them into this app. The app automatically tiles all pasted images onto a printable US Letter page, optimizing layout with optional rotation. Empty gaps ("streets") between images provide straight cutting paths. The user prints the result onto sticker paper and cuts them out.

## Tech Stack

- **GUI**: PySide6 (Qt6 for Python, LGPL, native macOS look)
- **Layout**: Custom row-based tiler — log-scale sizing, row-height quantization, row packing (rectpack evaluated and dropped; row-based approach gives straight cut lines)
- **Image Handling**: Pillow (for image normalization before packing) + Qt's QImage/QPixmap (for display and print)
- **Project Files**: Python `pickle` — serialize the full project (image data + layout) into a single `.sticker` file
- **Print**: QPrintDialog / QPrinter (native macOS print dialog)

## Architecture

Single-file to start (`sticker_app.py`), split only when complexity demands it.

### Core Classes

- **StickerProject** — data model: list of images (as raw bytes + metadata), current layout result, page settings (letter size, margins, cut-line width). Pickle-serializable.
- **PageWidget(QWidget)** — the WYSIWYG page view. Paints the tiled layout at screen resolution. Handles paste events. This is the central (and only) widget in the main window.
- **MainWindow(QMainWindow)** — menu bar (File: New/Open/Save/Save As/Print, Edit: Paste/Delete Selected/Clear All), hosts PageWidget.
- **Tiler** — orchestrates the full layout pipeline: sizing, row assignment, and final placement. See "Layout Algorithm" below.

### Key Dimensions

- US Letter: 8.5 x 11 inches
- Print DPI: 300 (for high-quality output)
- Page pixels at 300 DPI: 2550 x 3300
- Margins: 0.25 inch (75px at 300 DPI) on all sides — keeps images away from unprintable edges
- Cut-line gap: ~4px at 300 DPI between images (empty space — "streets" for cutting, not visibly drawn)

### Layout Algorithm

The core insight: **image pixel resolution should NOT determine print size.** A 150x150 image and a 1500x1500 image should print at roughly the same physical size. The audience is a 5-year-old who wants consistent sticker sizes, and the operator wants straight cut lines.

#### Step 1: Assign "ideal" sizes via log scaling

Raw pixel dimensions are compressed with a logarithmic function to produce unitless target sizes that reflect the image's aspect ratio but dampen resolution differences:

```
ideal_size = log2(pixels + 1)
```

So a 150px-wide image gets ideal width ~7.2, and a 1500px-wide image gets ~10.5 — a 10x resolution difference becomes a ~1.5x size difference. Aspect ratios are preserved (the log is applied to both dimensions independently).

#### Step 2: Quantize heights into row bins

To enable straight horizontal cut lines along the long (11") dimension:
- Collect all ideal heights, cluster them into a small number of discrete row heights (e.g., 3-5 bins)
- Each image snaps to the nearest bin height; its width scales proportionally to maintain aspect ratio
- This means a row of images all share the same height — one straight horizontal cut per row

Bin selection can be simple: sort ideal heights, split into N roughly equal groups, use each group's median as the bin height.

#### Step 3: Pack rows onto the page

With quantized heights, the problem simplifies to a 1D strip-packing per row:
- For each row height bin, line up images left-to-right with cut-line gaps between them
- When a row is full (hits page width), start a new row of the same or different bin height
- Stack rows top-to-bottom with cut-line gaps between rows
- If the page overflows vertically, uniformly scale ALL bin heights down and repack

This approach may or may not use `rectpack` — it might be simpler to implement directly since rows + quantized heights reduce it to a much easier problem than general 2D bin packing. We'll evaluate during implementation: if the row-based approach works well, skip rectpack entirely; if we want to try mixed-height packing for better space utilization, rectpack with pre-quantized sizes is still an option.

#### Cut line goals

Cut lines are not visibly drawn — they're just empty gaps ("streets") between images where you cut. The layout creates straight cutting paths:

- **Horizontal streets span the full page width** (between rows) — fewest cuts, longest straight lines
- **Vertical streets are per-row** — within a row, images are separated by vertical gaps
- Result: a grid-like structure where each row has uniform height, easy to cut with a paper cutter

## Roadmap

### Phase 1: Skeleton App
- [x] Create `requirements.txt` (PySide6, Pillow)
- [x] Main window with menu bar (File, Edit menus — stubs)
- [x] PageWidget that draws a white US Letter rectangle centered in the window, scaled to fit
- [x] Status bar showing image count and zoom level

### Phase 2: Paste & Display
- [x] Handle Cmd+V / clipboard paste: extract image from clipboard (QClipboard.image())
- [x] Also handle drag-and-drop of image files onto the window
- [ ] Handle drag-and-drop of image URLs (some browsers supply a URL instead of image data) — detect URL mime type and download the image
- [x] Store pasted images in StickerProject as PNG byte blobs (normalize all formats to PNG via Pillow)
- [x] Display pasted images in a simple grid on the page (temporary, before real packing)

### Phase 3: Layout / Tiling
- [x] Implement log-scale sizing: convert pixel dimensions to ideal print sizes via `log2(px + 1)`, preserving aspect ratio
- [x] Implement row-height quantization: cluster ideal heights into N bins, snap each image to nearest bin
- [x] Implement row packing: fill rows left-to-right, stack rows top-to-bottom
- [x] Implement scale-to-fit: if rows overflow the page, uniformly shrink all bin heights and repack (binary search)
- [x] On every paste/delete, re-run layout and update PageWidget
- [x] Layout includes cut-line gaps (empty streets) between rows and between images within rows
- [x] Evaluate whether `rectpack` adds value over the row-based approach; use it or drop the dependency

### Phase 4: Print
- [x] File > Print: open QPrintDialog, render the page at 300 DPI via QPainter onto QPrinter (system print dialog provides Print-to-PDF for free)
- [x] Ensure WYSIWYG: same layout logic for screen and print, just different DPI
- [ ] Test with actual sticker paper

### Phase 5: Save / Load
- [x] StickerProject pickle serialization (images as PNG bytes, layout metadata)
- [x] File > Save / Save As: write `.sticker` file (pickled StickerProject)
- [x] File > Open: load `.sticker` file, re-tile, display
- [x] File > New: clear project
- [x] Track dirty state, prompt "unsaved changes" on close/new/open

### Phase 6: Selection & Polish
- [x] Click an image on the page to select it (highlight border)
- [x] Delete key removes selected image, re-tiles
- [x] Cmd+C copies selected image back to clipboard (nice-to-have)
- [x] Right-click context menu on images (delete, copy)
- [x] Undo/redo stack (QUndoStack) for paste and delete operations

### Phase 7: Stretch Goals
- [ ] Multiple pages (if images overflow one page, add a second)
- [ ] Manual image reordering / pinning (lock an image's position)
- [ ] Adjustable margins and cut-line style in preferences
- [ ] Global hotkey to capture image under mouse cursor from browser — ideas:
  - **AppleScript + screencapture**: Register a global hotkey (e.g. via PyObjC or a small Swift helper), use AppleScript to get the frontmost browser's current URL or run JavaScript to find the image element under the cursor, then download it. Fragile but zero browser extension needed.
  - **Accessibility API approach**: Use macOS Accessibility APIs (via PyObjC) to inspect the browser's DOM/AX tree for the image element under the cursor and extract its URL.
  - **PySide6 embedded QWebEngineView**: Ship a built-in mini-browser inside the app. User browses in-app; on click or hotkey, the app intercepts the image directly from the web engine. Full control, no AppleScript, but adds QtWebEngine as a heavy dependency.
  - **System-wide drag target**: Keep the app as a floating always-on-top thumbnail/dock. User drags images from any browser onto it. Already partially works via drag-and-drop; this just makes the drop target more accessible.

## Dev Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
python sticker_app.py

# Run with debug logging
python sticker_app.py --debug
```

## File Structure (target)

```
Stickers/
  CLAUDE.md          # this file
  requirements.txt
  sticker_app.py     # main application (single file to start)
  .venv/             # virtual environment (gitignored)
```

## Design Decisions

- **Single file**: Keep it in one file as long as it stays under ~800 lines. Split into modules only when navigating becomes painful.
- **Pickle for save files**: Simplest possible persistence. `.sticker` files are just pickled `StickerProject` objects. Not meant to be cross-version stable — this is a personal tool, not a product.
- **Log-scale sizing**: A 10x resolution difference becomes ~1.5x size difference. Stickers end up roughly consistent in size, which is what a kid wants. The exact log base can be tuned.
- **Row-based layout over general bin packing**: Quantizing heights into row bins gives us straight horizontal cut lines for free. This sacrifices some packing density vs. arbitrary 2D packing, but the tradeoff is worth it — easier cutting matters more than squeezing in one extra sticker.
- **rectpack as optional**: We keep it in requirements as a fallback, but the row-based approach may be all we need. Will evaluate in Phase 3.
- **Scale-to-fit strategy**: Rather than rejecting images that don't fit, uniformly scale ALL bin heights down until the layout fits the page. Paste and it just works.
- **300 DPI internal**: All layout math happens in 300 DPI pixel space. The screen view scales down for display. Print renders at native resolution. One coordinate system, no conversion bugs.
- **Cut lines are invisible streets**: Gaps between images are empty space where you cut — no visible lines are drawn. Horizontal gaps span full page width between rows; vertical gaps separate images within a row. The layout is designed around cuttability, not just space efficiency.
