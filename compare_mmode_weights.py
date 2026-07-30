# -*- coding: utf-8 -*-
"""
compare_mmode_weights.py

Before/after comparison of two sets of M-Mode model weights on a sample of
DICOM files - built to check whether fine-tuning actually improved
segmentation, especially on groups (e.g. WT/WF/WM) that were underrepresented
in earlier annotation rounds.

USAGE:
    python compare_mmode_weights.py --input_dir "<path>" [--n_per_group 4]
        [--weights_old ./model_weights/weights_MMode_finetuned_v2.h5]
        [--weights_new ./model_weights/weights_MMode_finetuned_v3.h5]

By default, samples files NOT already in training_data_mmode (held-out,
never seen by either model during training/fine-tuning), split evenly
across the 6 strain/sex groups (OF/OM/OX/WT/WF/WM).

Outputs:
    <output_dir>/compare_output.pdf  - side-by-side segmentation, old vs new
    <output_dir>/compare_metrics.csv - LVID/LVAW/LVPW/FS/LV_Mass per file per weight set
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import pydicom
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from util_mmode import getRes_rawDICOM, compute_MMode_metrics, postprocess, getRes, crop_frame
from util_nn_mmode import get_unet

MEAN, STD = 127, 51
GROUP_PATTERN = re.compile(r'(O[FMX]|W[FMT])\s?\d')


def get_group(filename):
    m = GROUP_PATTERN.search(filename.upper())
    return m.group(1) if m else 'UNKNOWN'


def select_sample(input_dir, n_per_group, data_dir='training_data_mmode'):
    annotated = set(f[:-4] for f in os.listdir(data_dir) if f.endswith('_mask.npy')) \
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
        res_x, res_y = res_x * 1000, res_y * 10

    if len(np.shape(img_raw)) == 4:
        imslice = img_raw[range(np.shape(img_raw)[0])]
    elif len(np.shape(img_raw)) == 3:
        imslice = img_raw
    else:
        raise ValueError('unsupported image shape')

    images = []
    for z in range(np.shape(img_raw)[0]):
        image = crop_frame(np.squeeze(imslice[z, :, :]))
        org_y, org_x = image.shape
        image = np.float32(image)
        image = rescale_intensity(image, out_range=(0, 255))
        image = np.uint8(np.round(resize(image, (256, 256))))
        images.append(image)

    imgs = np.stack(images)[..., np.newaxis]
    res_x, res_y = res_x * org_x / 256, res_y * org_y / 256

    imgs_norm = imgs.astype(np.float32)
    imgs_norm -= MEAN
    imgs_norm /= STD

    pred = model.predict(imgs_norm, verbose=0, batch_size=1)
    conf = list(np.squeeze(np.mean(np.mean(np.mean(np.abs(pred - 0.5), 3), 2), 1)))
    pred = np.round(pred).astype(np.uint8)
    label = np.squeeze(postprocess(pred))

    cutoff = int(np.floor(256 * .1))
    label = label[:, :, cutoff - 1:-cutoff]
    imgs_c = imgs[:, :, cutoff - 1:-cutoff, :]

    coverage_scores = []
    for z in range(label.shape[0]):
        frame_label = label[z, :, :]
        n_cols = frame_label.shape[1]
        valid_cols = sum(1 for col in range(n_cols) if len(np.unique(frame_label[:, col])) == 4)
        coverage_scores.append(valid_cols / n_cols)

    max_coverage = max(coverage_scores)
    candidate_idxs = [i for i, c in enumerate(coverage_scores) if c >= max_coverage - 0.02]
    idx = max(candidate_idxs, key=lambda i: conf[i])

    metrics = compute_MMode_metrics(np.squeeze(label[idx, :, :]), res_y, res_x, agg_fn=np.median)
    LVAW_s, LVAW_d, LVID_s, LVID_d, LVPW_s, LVPW_d, FS, LV_Mass, LV_Mass_Cor, HR = metrics

    return imgs_c, label, idx, LVAW_s, LVAW_d, LVID_s, LVID_d, LVPW_s, LVPW_d, FS, LV_Mass, HR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--n_per_group', type=int, default=4)
    parser.add_argument('--weights_old', default='./model_weights/weights_MMode_finetuned_v2.h5')
    parser.add_argument('--weights_new', default='./model_weights/weights_MMode_finetuned_v3.h5')
    parser.add_argument('--output_dir', default='./compare_output_mmode')
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
                res_o = run_model(filepath, model_old)
                res_n = run_model(filepath, model_new)
            except Exception as e:
                print(f"  Skipping {file}: {e}")
                continue

            imgs_o, label_o, idx_o, LVAWs_o, LVAWd_o, LVIDs_o, LVIDd_o, LVPWs_o, LVPWd_o, FS_o, LVM_o, HR_o = res_o
            imgs_n, label_n, idx_n, LVAWs_n, LVAWd_n, LVIDs_n, LVIDd_n, LVPWs_n, LVPWd_n, FS_n, LVM_n, HR_n = res_n

            rows.append({
                'file': file[:-4], 'group': group,
                'FS_old (%)': FS_o, 'FS_new (%)': FS_n,
                'LVID_d_old (mm)': LVIDd_o, 'LVID_d_new (mm)': LVIDd_n,
                'LVID_s_old (mm)': LVIDs_o, 'LVID_s_new (mm)': LVIDs_n,
                'LV_Mass_old (mg)': LVM_o, 'LV_Mass_new (mg)': LVM_n,
                'HR_old': HR_o, 'HR_new': HR_n,
            })

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            for ax, imgs, label, idx, FS, label_txt in [
                (axes[0], imgs_o, label_o, idx_o, FS_o, 'OLD'),
                (axes[1], imgs_n, label_n, idx_n, FS_n, 'NEW'),
            ]:
                a = rescale_intensity(imgs[idx, :, :, 0], out_range=(-1, 1))
                b = label[idx, :, :]
                ab = mark_boundaries(a, b, color=(1, 44/255, 44/255))
                ax.imshow(rescale_intensity(ab, out_range=(0, 255)).astype('uint8'))
                ax.set_title(f"{label_txt}  FS={FS:.1f}%")
                ax.axis('off')
            fig.suptitle(f"{file}  [{group}]", fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  {file} [{group}]  FS old={FS_o:.1f}%  FS new={FS_n:.1f}%")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, 'compare_metrics.csv')
    df.to_csv(csv_path, index=False)

    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {csv_path}")

    if len(df):
        print("\nMean FS by group (old vs new):")
        print(df.groupby('group')[['FS_old (%)', 'FS_new (%)']].mean())


if __name__ == '__main__':
    main()
