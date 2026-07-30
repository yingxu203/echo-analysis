# -*- coding: utf-8 -*-
"""
echoanalysis_subset_mmode.py

Runs M-mode inference (metrics + QC overlay PDF) on an arbitrary subset of
DICOM files with an arbitrary weights file. Built for targeted corrections:
after fine-tuning on new hand-annotations for one treatment/strain group,
re-run just that group's files with the new weights, without touching the
rest of the dataset - while still getting the same output.csv +
output_images.pdf QC record that the full echoanalysis_mmode_main.py run
produces, in its own clearly-labeled run folder.

USAGE:
    python echoanalysis_subset_mmode.py \
        --input_dir "./annotate_PF_OX_mmode" \
        --weights ./model_weights/weights_MMode_finetuned_v4.h5 \
        --label PFOX_corrected \
        --out_base "/path/to/M-Mode" \
        [--merge_into "/path/to/M-Mode/output/.../output.csv"]

Outputs, all in one timestamped folder under <out_base>/output/:
    output.csv           - metrics for just this subset
    output_images.pdf    - segmentation overlay per file (best frame)
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

from compare_mmode_weights import run_model
from util_nn_mmode import get_unet


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
    segs, seg_files = [], []
    for f in files:
        filepath = os.path.join(args.input_dir, f)
        try:
            res = run_model(filepath, model)
            imgs, label, idx, LVAW_s, LVAW_d, LVID_s, LVID_d, LVPW_s, LVPW_d, FS, LV_Mass, HR = res
        except Exception as e:
            print(f"  WARNING: failed to process {f}: {e}")
            rows.append({'animal_id': f[:-4], 'LVAW_sys (mm)': np.nan, 'LVAW_dia (mm)': np.nan,
                         'LVPW_sys (mm)': np.nan, 'LVPW_dia (mm)': np.nan, 'LVID_sys (mm)': np.nan,
                         'LVID_dia (mm)': np.nan, 'FS (%)': np.nan, 'LV_Mass (mg)': np.nan,
                         'LV_Mass_Cor (mg)': np.nan, 'Heart_Rate': np.nan})
            continue

        LV_Mass_Cor = 0.8 * LV_Mass
        rows.append({'animal_id': f[:-4], 'LVAW_sys (mm)': LVAW_s, 'LVAW_dia (mm)': LVAW_d,
                     'LVPW_sys (mm)': LVPW_s, 'LVPW_dia (mm)': LVPW_d, 'LVID_sys (mm)': LVID_s,
                     'LVID_dia (mm)': LVID_d, 'FS (%)': FS, 'LV_Mass (mg)': LV_Mass,
                     'LV_Mass_Cor (mg)': LV_Mass_Cor, 'Heart_Rate': HR})

        color = line_color_for(f)
        a = rescale_intensity(imgs[idx, :, :, 0], out_range=(-1, 1))
        b = label[idx, :, :]
        ab = mark_boundaries(a, b, color=color)
        segs.append(rescale_intensity(ab, out_range=(0, 255)).astype('uint8'))
        seg_files.append(f[:-4])

    df = pd.DataFrame(rows)
    csv_path = os.path.join(run_dir, 'output.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    pdf_path = os.path.join(run_dir, 'output_images.pdf')
    with PdfPages(pdf_path) as pdf:
        for i, name in enumerate(seg_files):
            plt.figure(figsize=(8, 6))
            plt.title(name, fontsize=14)
            plt.imshow(segs[i])
            pdf.savefig()
            plt.close()

        d = pdf.infodict()
        d['Title'] = f'Segmentation QC - {args.label}'
        d['Subject'] = f'M-mode subset run, weights={weights_label}'
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
