# -*- coding: utf-8 -*-
"""
annotate_bmode.py

Interactive tool for creating ground-truth training masks for the B-Mode
Echo Segmenter model. Unlike M-mode (which traces 4 wall boundaries), B-mode
only needs a single closed contour around the LV cavity, which gets filled
into a binary mask.

For each DICOM file, this tool:
  1. Runs the same preprocessing + model prediction as the main tool
  2. Picks two representative frames per cine loop: an estimated diastolic
     frame (largest predicted LV area) and an estimated systolic frame
     (smallest predicted LV area) - matching the two phases the main tool
     actually measures (see findcardiacpeaks in util_bmode.py)
  3. Lets you click points around the LV cavity boundary on each frame
  4. Fills the closed contour to build a binary mask
  5. Saves the image + mask pair for use in fine-tuning

USAGE:
    python annotate_bmode.py "/path/to/dicom/folder"
    python annotate_bmode.py "/path/to/dicom/folder" specific_file.dcm

CONTROLS (while annotating):
    - Left-click on the image to add a point around the LV cavity boundary
      (click in order around the perimeter, e.g. clockwise)
    - Press 'u' to undo the last point
    - Press 'q' to close the contour and save (needs at least 3 points)
    - Close the window / press 'q' with fewer than 3 points to SKIP (not saved)

TIP: You don't need to click every pixel - click points roughly every
10-20 pixels around the boundary; the polygon is filled automatically
between your points in click order.

Already-annotated frames are automatically skipped on repeat runs, so you
can stop and resume this process across multiple sessions.

NOTE ON FILE ORDER: files are NOT processed in plain alphabetical order.
The M-mode annotation set for this project ended up ~30/31 examples from
the alphabetically-first "ISO O*" group with almost no "W*" (WT/WF/WM)
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
import cv2
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
from skimage.transform import resize

from util_bmode import dicom_preprocess
from util_nn_bmode import get_unet

OUTPUT_DIR = 'training_data_bmode'
WEIGHTS_PATH = './model_weights/weights_ECHO_clean.h5'

MEAN, STD = 92, 57

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


def get_candidate_frames(filepath, model):
    """Replicates the main tool's preprocessing + cardiac-phase detection,
    so the annotated frames match what the main tool actually measures.

    Returns a list of (phase_label, raw_uint8_256x256_image) tuples -
    one estimated diastolic frame and one estimated systolic frame.
    """
    data = pydicom.dcmread(filepath)
    img_raw = data.pixel_array

    if len(np.shape(img_raw)) != 4:
        return []

    images = []
    for i in range(img_raw.shape[0]):
        image = dicom_preprocess(img_raw[i])
        image = np.float32(image)
        image = np.uint8(np.round(resize(image, (256, 256))))
        images.append(image)

    imgs = np.stack(images)
    imgs = imgs[..., np.newaxis]

    imgs_norm = imgs.astype(np.float32)
    imgs_norm -= MEAN
    imgs_norm /= STD

    pred = model.predict(imgs_norm, verbose=0, batch_size=1)
    pred_bin = np.round(pred).astype(np.uint8)

    areas = np.sum(np.reshape(pred_bin, (pred_bin.shape[0], -1)), axis=1)
    areas_smoothed = scipy.ndimage.gaussian_filter(areas.astype(np.float64), 5)

    diastoles, _ = scipy.signal.find_peaks(areas_smoothed, distance=10)
    systoles, _ = scipy.signal.find_peaks(areas_smoothed * -1, distance=10)

    dia_idx = diastoles[0] if len(diastoles) else int(np.argmax(areas_smoothed))
    sys_idx = systoles[0] if len(systoles) else int(np.argmin(areas_smoothed))

    candidates = [('dia', imgs[dia_idx, :, :, 0])]
    if sys_idx != dia_idx:
        candidates.append(('sys', imgs[sys_idx, :, :, 0]))

    return candidates


class ContourAnnotator:
    """Interactive matplotlib-based clicker for tracing the LV cavity contour."""

    def __init__(self, image, title):
        self.image = image
        self.title = title
        self.points = []
        self.finished = False

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.redraw()
        plt.show()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        self.points.append((event.xdata, event.ydata))
        self.redraw()

    def on_key(self, event):
        if event.key == 'u':
            if self.points:
                self.points.pop()
                self.redraw()
        elif event.key == 'q':
            if len(self.points) >= 3:
                self.finished = True
            plt.close(self.fig)

    def redraw(self):
        self.ax.clear()
        self.ax.imshow(self.image, cmap='gray')
        if self.points:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self.ax.plot(xs, ys, 'o-', color='red', markersize=5, linewidth=1)
            if len(self.points) >= 3:
                # preview the closing edge back to the first point
                self.ax.plot([xs[-1], xs[0]], [ys[-1], ys[0]], '--', color='red', alpha=0.5)

        title = (f"{self.title}\n"
                 f"Click around LV cavity boundary | 'u' = undo | "
                 f"'q' = close+save (need >=3 pts)")
        self.ax.set_title(title, fontsize=10)
        self.fig.canvas.draw()

    def get_mask(self):
        if not self.finished:
            return None
        h, w = self.image.shape
        pts = np.array([[int(round(x)), int(round(y))] for x, y in self.points], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        return mask


def main():
    if len(sys.argv) < 2:
        print('Usage: python annotate_bmode.py "<path_to_dicom_folder>" [optional_specific_filename.dcm]')
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

    print("Loading model (used only to pick representative sys/dia frames)...")
    model = get_unet()
    model.load_weights(WEIGHTS_PATH)

    annotated_count = 0
    for file in files:
        base = file[:-4]

        try:
            candidates = get_candidate_frames(os.path.join(input_dir, file), model)
        except Exception as e:
            print(f"  Skipping {file} (error reading/processing): {e}")
            continue

        if not candidates:
            print(f"  Skipping {file} (unsupported image shape)")
            continue

        for phase, image in candidates:
            img_path = os.path.join(OUTPUT_DIR, f"{base}_{phase}_image.npy")
            mask_path = os.path.join(OUTPUT_DIR, f"{base}_{phase}_mask.npy")

            if os.path.exists(mask_path) and not target_file:
                print(f"Skipping {file} [{phase}] (already annotated)")
                continue

            print(f"\n=== {file} [{phase}] ===")
            annotator = ContourAnnotator(image, f"{file} ({phase})")
            mask = annotator.get_mask()

            if mask is None:
                print(f"  Skipped {file} [{phase}] (annotation incomplete or window closed early)")
                continue

            np.save(img_path, image)
            np.save(mask_path, mask)
            annotated_count += 1
            print(f"  Saved. ({annotated_count} annotated this session)")

    print(f"\nDone this session. Total annotated frames in {OUTPUT_DIR}:",
          len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('_mask.npy')]))


if __name__ == '__main__':
    main()
