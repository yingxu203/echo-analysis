# -*- coding: utf-8 -*-
"""
annotate_mmode.py

Interactive tool for creating ground-truth training masks for the M-Mode
Echo Segmenter model. For each DICOM file, this tool:
  1. Runs the same preprocessing + model prediction as the main tool to find
     the same "best" frame that was used for measurements
  2. Lets you click points along each of the 4 anatomical boundaries:
       - Top of Anterior Wall (LVAW top)
       - Bottom of Anterior Wall / Top of Cavity (LVAW/LVID boundary)
       - Bottom of Cavity / Top of Posterior Wall (LVID/LVPW boundary)
       - Bottom of Posterior Wall (LVPW bottom)
  3. Interpolates between your clicked points to build a full pixel mask
  4. Saves the image + mask pair for use in fine-tuning

USAGE:
    python annotate_mmode.py "/path/to/dicom/folder"

CONTROLS (while annotating):
    - Left-click on the image to add a point to the current boundary
    - Press 'n' to move to the next boundary (need at least 2 points first)
    - Press 'u' to undo the last point on the current boundary
    - Press 'q' to finish this frame and save (only works after all 4 boundaries have points)
    - Close the window / press 'q' with incomplete boundaries to SKIP this frame (not saved)

TIP: Click points roughly every 15-25 pixels along the x-axis for a smooth
curve. You don't need to click every single column - the tool interpolates
between your points.

Already-annotated files are automatically skipped on repeat runs, so you can
stop and resume this process across multiple sessions.

NOTE ON FILE ORDER: files are NOT processed in plain alphabetical order.
The existing M-mode training set ended up ~30/31 examples from the
alphabetically-first "ISO O*" group with almost no "W*" (WT/WF/WM)
examples, because annotation sessions kept stopping before the sort order
reached the W files - alphabetical sorting silently produces a skewed
training set if a session is ever cut short. Instead, files here are
interleaved round-robin across group codes (OF/OM/OX/WT/WF/WM) so that
ANY prefix of the file list - i.e. wherever you stop - stays balanced
across groups.
"""

import os
import re
import sys
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.exposure import rescale_intensity

from util_mmode import crop_frame
from util_nn_mmode import get_unet

OUTPUT_DIR = 'training_data_mmode'
WEIGHTS_PATH = './model_weights/weights_MMode_clean_v4.h5'

GROUP_PATTERN = re.compile(r'(O[FMX]|W[FMT])\s?\d')


def get_group(filename):
    """Extract the strain/sex group code (OF/OM/OX/WT/WF/WM) from a filename."""
    m = GROUP_PATTERN.search(filename.upper())
    return m.group(1) if m else 'UNKNOWN'


def stratified_order(files):
    """Round-robin across group codes so that any prefix of the returned
    list - not just the full list - has balanced representation across
    groups. This prevents an interrupted annotation session from silently
    skewing the training set toward whichever group sorts first."""
    buckets = {}
    for f in files:
        buckets.setdefault(get_group(f), []).append(f)
    for g in buckets:
        buckets[g].sort()

    ordered = []
    iterators = [iter(v) for v in buckets.values()]
    while iterators:
        remaining = []
        for it in iterators:
            try:
                ordered.append(next(it))
                remaining.append(it)
            except StopIteration:
                pass
        iterators = remaining
    return ordered

BOUNDARY_NAMES = [
    'Top of Anterior Wall (LVAW top)',
    'Bottom of Anterior Wall / Top of Cavity (LVAW/LVID)',
    'Bottom of Cavity / Top of Posterior Wall (LVID/LVPW)',
    'Bottom of Posterior Wall (LVPW bottom)'
]
BOUNDARY_COLORS = ['red', 'blue', 'green', 'orange']


def get_best_frame(filepath, model):
    """Replicates the main tool's preprocessing + frame selection logic,
    so the annotated frame matches what was actually measured before."""
    data = pydicom.dcmread(filepath)
    img_raw = data.pixel_array

    if len(np.shape(img_raw)) == 4:
        imslice = img_raw[range(np.shape(img_raw)[0])]
    elif len(np.shape(img_raw)) == 3:
        imslice = img_raw
    else:
        return None

    images = []
    for z in range(np.shape(img_raw)[0]):
        image = crop_frame(np.squeeze(imslice[z, :, :]))
        image = np.float32(image)
        image = rescale_intensity(image, out_range=(0, 255))
        image = np.uint8(np.round(resize(image, (256, 256))))
        images.append(image)

    imgs = np.stack(images)
    imgs = imgs[..., np.newaxis]

    mean, std = 127, 51
    imgs_norm = imgs.astype(np.float32)
    imgs_norm -= mean
    imgs_norm /= std

    pred = model.predict(imgs_norm, verbose=0, batch_size=1)
    conf = list(np.squeeze(np.mean(np.mean(np.mean(np.abs(pred - 0.5), 3), 2), 1)))
    idx = conf.index(max(conf))

    return imgs[idx, :, :, 0]


class BoundaryAnnotator:
    """Interactive matplotlib-based clicker for tracing the 4 boundaries."""

    def __init__(self, image, filename):
        self.image = image
        self.filename = filename
        self.current_boundary = 0
        self.points = [[] for _ in range(4)]
        self.finished = False

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.redraw()
        plt.show()

    def on_click(self, event):
        if event.inaxes != self.ax or self.current_boundary >= 4:
            return
        x, y = event.xdata, event.ydata
        self.points[self.current_boundary].append((x, y))
        self.redraw()

    def on_key(self, event):
        if event.key == 'n':
            if len(self.points[self.current_boundary]) < 2:
                print("  Need at least 2 points before moving to next boundary.")
                return
            self.current_boundary += 1
            self.redraw()
            if self.current_boundary >= 4:
                print("  All 4 boundaries marked. Press 'q' to save and move to next file.")
        elif event.key == 'u':
            if self.points[self.current_boundary]:
                self.points[self.current_boundary].pop()
                self.redraw()
        elif event.key == 'q':
            if self.current_boundary >= 4 and all(len(p) >= 2 for p in self.points):
                self.finished = True
            plt.close(self.fig)

    def redraw(self):
        self.ax.clear()
        self.ax.imshow(self.image, cmap='gray')
        for i, pts in enumerate(self.points):
            for x, y in pts:
                self.ax.plot(x, y, 'o', color=BOUNDARY_COLORS[i], markersize=6)
            if len(pts) >= 2:
                pts_sorted = sorted(pts, key=lambda p: p[0])
                xs = [p[0] for p in pts_sorted]
                ys = [p[1] for p in pts_sorted]
                self.ax.plot(xs, ys, '-', color=BOUNDARY_COLORS[i], linewidth=1, alpha=0.6)

        if self.current_boundary < 4:
            title = (f"{self.filename}\n"
                     f"Boundary {self.current_boundary + 1}/4 ({BOUNDARY_COLORS[self.current_boundary]}): "
                     f"{BOUNDARY_NAMES[self.current_boundary]}\n"
                     f"Click points | 'n' = next boundary | 'u' = undo | 'q' = save+finish")
        else:
            title = f"{self.filename}\nAll boundaries marked. Press 'q' to save and continue."
        self.ax.set_title(title, fontsize=10)
        self.fig.canvas.draw()

    def get_mask(self):
        if not self.finished:
            return None
        h, w = self.image.shape
        curves = []
        for pts in self.points:
            pts_sorted = sorted(pts, key=lambda p: p[0])
            xs = [p[0] for p in pts_sorted]
            ys = [p[1] for p in pts_sorted]
            xs_full = np.arange(w)
            ys_full = np.interp(xs_full, xs, ys)
            curves.append(ys_full)

        mask = np.zeros((h, w), dtype=np.uint8)
        for col in range(w):
            y1, y2, y3, y4 = [c[col] for c in curves]
            y1, y2, y3, y4 = sorted([y1, y2, y3, y4])  # safety: enforce top-to-bottom order
            mask[int(round(y1)):int(round(y2)), col] = 1
            mask[int(round(y2)):int(round(y3)), col] = 2
            mask[int(round(y3)):int(round(y4)), col] = 3
        return mask


def main():
    if len(sys.argv) < 2:
        print('Usage: python annotate_mmode.py "<path_to_dicom_folder>" [optional_specific_filename.dcm]')
        return

    input_dir = sys.argv[1]
    target_file = sys.argv[2] if len(sys.argv) > 2 else None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.dcm')]
    files = stratified_order(all_files)

    if target_file:
        if target_file not in files:
            print(f"ERROR: '{target_file}' not found in {input_dir}")
            return
        files = [target_file]
        print(f"Annotating single file: {target_file}")
    else:
        from collections import Counter
        counts = Counter(get_group(f) for f in files)
        print(f"Group balance in annotation order: {dict(counts)}")

    print("Loading model (used only to pick the same 'best' frame as the main tool)...")
    model = get_unet()
    model.load_weights(WEIGHTS_PATH)

    annotated_count = 0
    for file in files:
        base = file[:-4]
        img_path = os.path.join(OUTPUT_DIR, f"{base}_image.npy")
        mask_path = os.path.join(OUTPUT_DIR, f"{base}_mask.npy")

        if os.path.exists(mask_path) and not target_file:
            print(f"Skipping {file} (already annotated)")
            continue

        filepath = os.path.join(input_dir, file)
        print(f"\n=== {file} ===")
        try:
            image = get_best_frame(filepath, model)
        except Exception as e:
            print(f"  Skipping (error reading/processing): {e}")
            continue

        if image is None:
            print("  Skipping (unsupported image shape)")
            continue

        annotator = BoundaryAnnotator(image, file)
        mask = annotator.get_mask()

        if mask is None:
            print(f"  Skipped {file} (annotation incomplete or window closed early)")
            continue

        np.save(img_path, image)
        np.save(mask_path, mask)
        annotated_count += 1
        print(f"  Saved. ({annotated_count} annotated this session)")

    print(f"\nDone this session. Total annotated files in {OUTPUT_DIR}:",
          len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('_mask.npy')]))


if __name__ == '__main__':
    main()
