# -*- coding: utf-8 -*-
"""
compare_bmode_weights.py

Before/after comparison of two sets of B-Mode model weights on a sample of
DICOM files - built to check whether fine-tuning actually improved
segmentation, especially on groups (e.g. WT/WF/WM) that were underrepresented
in earlier annotation rounds.

USAGE:
    python compare_bmode_weights.py --input_dir "<path>" [--n_per_group 4]
        [--weights_old ./model_weights/weights_ECHO_clean.h5]
        [--weights_new ./model_weights/weights_BMode_finetuned.h5]

By default, samples files NOT already in training_data_bmode (held-out,
never seen by either model during training/fine-tuning), split evenly
across the 6 strain/sex groups (OF/OM/OX/WT/WF/WM) so the comparison isn't
skewed toward whichever group happens to be easiest.

Outputs:
    <output_dir>/compare_output.pdf  - side-by-side diastole overlay, old vs new
    <output_dir>/compare_metrics.csv - EF/areas per file per weight set
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import pydicom
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from util_bmode import dicom_preprocess, computeMetrics, findcardiacpeaks, getRes_rawDICOM, getRes, postprocess_masks
from util_nn_bmode import get_unet

MEAN, STD = 92, 57
GROUP_PATTERN = re.compile(r'(O[FMX]|W[FMT])\s?\d')


def get_group(filename):
    m = GROUP_PATTERN.search(filename.upper())
    return m.group(1) if m else 'UNKNOWN'


def select_sample(input_dir, n_per_group, data_dir='training_data_bmode'):
    annotated = set(f.rsplit('_', 2)[0] for f in os.listdir(data_dir) if f.endswith('_mask.npy')) \
        if os.path.isdir(data_dir) else set()
    files = [f for f in os.listdir(input_dir) if f.endswith('.dcm')]
    buckets = {}
    for f in files:
        if f[:-4] in annotated:
            continue
        buckets.setdefault(get_group(f), []).append(f)

    sample = []
    for g in sorted(buckets):
        sample.extend(sorted(buckets[g])[:n_per_group])
    return sample


def run_model(filepath, model):
    data = pydicom.dcmread(filepath)
    img_raw = data.pixel_array

    try:
        res_x, res_y = getRes_rawDICOM(data)
    except Exception:
        res_x, res_y = getRes(data)
        res_x, res_y = res_x * 10, res_y * 10

    images = []
    for i in range(img_raw.shape[0]):
        image = dicom_preprocess(img_raw[i])
        org_y, org_x = image.shape
        image = np.float32(image)
        image = np.uint8(np.round(resize(image, (256, 256))))
        images.append(image)

    imgs = np.stack(images)[..., np.newaxis]
    res_x, res_y = res_x * org_x / 256, res_y * org_y / 256

    imgs_norm = imgs.astype(np.float32)
    imgs_norm -= MEAN
    imgs_norm /= STD

    pred = model.predict(imgs_norm, verbose=0, batch_size=1)
    pred = pred.astype(np.uint8)
    pred = postprocess_masks(pred, contappr=False)

    systoles, diastoles = findcardiacpeaks(pred)
    sys_area, dia_area, sys_vol, dia_vol, ef = computeMetrics(pred, systoles, diastoles, res_x, res_y)

    return imgs, pred, systoles, diastoles, sys_area, dia_area, sys_vol, dia_vol, ef


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--n_per_group', type=int, default=4)
    parser.add_argument('--weights_old', default='./model_weights/weights_ECHO_clean.h5')
    parser.add_argument('--weights_new', default='./model_weights/weights_BMode_finetuned.h5')
    parser.add_argument('--output_dir', default='./compare_output_bmode')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sample = select_sample(args.input_dir, args.n_per_group)
    print(f"Sample: {len(sample)} held-out files across groups: "
          f"{sorted(set(get_group(f) for f in sample))}")
    for f in sample:
        print(f"  [{get_group(f)}] {f}")

    print("Loading OLD weights...")
    model_old = get_unet()
    model_old.load_weights(args.weights_old)

    print("Loading NEW weights...")
    model_new = get_unet()
    model_new.load_weights(args.weights_new)

    rows = []
    pdf_path = os.path.join(args.output_dir, 'compare_output.pdf')
    with PdfPages(pdf_path) as pdf:
        for file in sample:
            filepath = os.path.join(args.input_dir, file)
            group = get_group(file)
            try:
                imgs_o, pred_o, sys_o, dia_o, sa_o, da_o, sv_o, dv_o, ef_o = run_model(filepath, model_old)
                imgs_n, pred_n, sys_n, dia_n, sa_n, da_n, sv_n, dv_n, ef_n = run_model(filepath, model_new)
            except Exception as e:
                print(f"  Skipping {file}: {e}")
                continue

            rows.append({
                'file': file[:-4], 'group': group,
                'EF_old (%)': ef_o, 'EF_new (%)': ef_n,
                'dia_area_old (mm2)': da_o, 'dia_area_new (mm2)': da_n,
                'sys_area_old (mm2)': sa_o, 'sys_area_new (mm2)': sa_n,
            })

            # Visual: diastole overlay, old vs new
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            for ax, imgs, pred, dia, ef, label in [
                (axes[0], imgs_o, pred_o, dia_o, ef_o, 'OLD'),
                (axes[1], imgs_n, pred_n, dia_n, ef_n, 'NEW'),
            ]:
                idx = dia[0]
                a = rescale_intensity(imgs[idx, :, :, 0], out_range=(-1, 1))
                b = pred[idx][:, :, 0].astype('uint8')
                ab = mark_boundaries(a, b, color=(1, 44/255, 44/255))
                ax.imshow(rescale_intensity(ab, out_range=(0, 255)).astype('uint8'))
                ax.set_title(f"{label}  EF={ef:.1f}%")
                ax.axis('off')
            fig.suptitle(f"{file}  [{group}]", fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  {file} [{group}]  EF old={ef_o:.1f}%  EF new={ef_n:.1f}%")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, 'compare_metrics.csv')
    df.to_csv(csv_path, index=False)

    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {csv_path}")

    if len(df):
        print("\nMean EF by group (old vs new):")
        print(df.groupby('group')[['EF_old (%)', 'EF_new (%)']].mean())


if __name__ == '__main__':
    main()
