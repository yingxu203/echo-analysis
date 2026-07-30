# -*- coding: utf-8 -*-
"""
echoanalysis_subset_bmode.py

Runs B-mode inference (metrics + QC overlay PDF) on an arbitrary subset of
DICOM files with an arbitrary weights file. Built for targeted corrections:
after fine-tuning on new hand-annotations for one treatment/strain group,
re-run just that group's files with the new weights, without touching the
rest of the dataset - while still getting the same output.csv +
output_images.pdf QC record that the full echoanalysis_main.py run
produces, in its own clearly-labeled run folder.

USAGE:
    python echoanalysis_subset_bmode.py \
        --input_dir "./annotate_PF_OX_bmode" \
        --weights ./model_weights/weights_BMode_finetuned_v2.h5 \
        --label PFOX_corrected \
        --out_base "/path/to/B-Mode" \
        [--merge_into "/path/to/B-Mode/output/.../output.csv"]

Outputs, all in one timestamped folder under <out_base>/output/:
    output.csv           - metrics for just this subset
    output_images.pdf    - segmentation overlay per file (sys | dia)
    output_merged.csv    - only if --merge_into given: the merge-target CSV
                            with this subset's rows replaced/added, everything
                            else left untouched
"""

import os
import argparse
import datetime
import numpy as np
import pandas as pd
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from compare_bmode_weights import run_model
from util_nn_bmode import get_unet


def line_color_for(filename):
    fname_upper = filename.upper()
    if any(tag in fname_upper for tag in ['WT', 'WF', 'WM']):
        return (0/255, 0/255, 255/255)   # blue
    elif any(tag in fname_upper for tag in ['OX', 'OF', 'OM']):
        return (255/255, 44/255, 44/255)  # red
    return (1, 1, 0)  # yellow fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help="folder of .dcm files to run")
    parser.add_argument('--weights', required=True, help="path to the .h5 weights to use")
    parser.add_argument('--label', default='subset', help="short label for the run folder name")
    parser.add_argument('--out_base', default=None,
                         help="parent dir to create output/ under (defaults to --input_dir)")
    parser.add_argument('--merge_into', default=None,
                         help="existing output.csv to merge this subset's rows into (other rows kept as-is)")
    args = parser.parse_args()

    out_base = args.out_base or args.input_dir
    weights_label = os.path.splitext(os.path.basename(args.weights))[0]
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(out_base, 'output', f'run_{timestamp}_{weights_label}_{args.label}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"This run's output folder: {run_dir}")

    print(f"Loading weights: {args.weights}")
    model = get_unet()
    model.load_weights(args.weights)

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith('.dcm'))
    print(f"Running inference on {len(files)} files...")

    rows = []
    sys_segs, dia_segs, seg_files = [], [], []
    for f in files:
        filepath = os.path.join(args.input_dir, f)
        try:
            imgs, pred, systoles, diastoles, sys_area, dia_area, sys_vol, dia_vol, ef = run_model(filepath, model)
        except Exception as e:
            print(f"  WARNING: failed to process {f}: {e}")
            rows.append({'animal_id': f[:-4], 'sys_area (mm2)': np.nan, 'dia_area (mm2)': np.nan,
                         'sys_volume (mm3)': np.nan, 'dia_volume (mm3)': np.nan, 'ejection_fraction': np.nan})
            continue

        rows.append({'animal_id': f[:-4], 'sys_area (mm2)': sys_area, 'dia_area (mm2)': dia_area,
                     'sys_volume (mm3)': sys_vol, 'dia_volume (mm3)': dia_vol, 'ejection_fraction': ef})

        color = line_color_for(f)
        sys_idx, dia_idx = systoles[0], diastoles[0]

        a = rescale_intensity(imgs[sys_idx, :, :, 0], out_range=(-1, 1))
        b = pred[sys_idx][:, :, 0].astype('uint8')
        ab = mark_boundaries(a, b, color=color)
        sys_segs.append(rescale_intensity(ab, out_range=(0, 255)).astype('uint8'))

        a = rescale_intensity(imgs[dia_idx, :, :, 0], out_range=(-1, 1))
        b = pred[dia_idx][:, :, 0].astype('uint8')
        ab = mark_boundaries(a, b, color=color)
        dia_segs.append(rescale_intensity(ab, out_range=(0, 255)).astype('uint8'))
        seg_files.append(f[:-4])

    df = pd.DataFrame(rows)
    csv_path = os.path.join(run_dir, 'output.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    pdf_path = os.path.join(run_dir, 'output_images.pdf')
    with PdfPages(pdf_path) as pdf:
        for i, name in enumerate(seg_files):
            plt.figure(figsize=(8, 6))
            plt.suptitle(name, fontsize=14)
            plt.subplot(121)
            plt.imshow(sys_segs[i])
            plt.title('systole')
            plt.subplot(122)
            plt.imshow(dia_segs[i])
            plt.title('diastole')
            pdf.savefig()
            plt.close()

        d = pdf.infodict()
        d['Title'] = f'Segmentation QC - {args.label}'
        d['Subject'] = f'B-mode subset run, weights={weights_label}'
        d['Keywords'] = 'Automated Echocardiography Analysis'
        d['ModDate'] = datetime.datetime.today()
    print(f"Saved: {pdf_path}")

    if args.merge_into:
        old_df = pd.read_csv(args.merge_into)
        merged = old_df[~old_df['animal_id'].isin(df['animal_id'])].copy()
        merged = pd.concat([merged, df], ignore_index=True)
        merged_path = os.path.join(run_dir, 'output_merged.csv')
        merged.to_csv(merged_path, index=False)
        print(f"Merged into {args.merge_into} ({len(df)} rows replaced/added, "
              f"{len(merged) - len(df)} untouched) -> {merged_path}")


if __name__ == '__main__':
    main()
