# -*- coding: utf-8 -*-
"""
build_qc_comparison.py

Builds a side-by-side QC comparison PDF: original (pre-fine-tuning) script
output on the left, latest (post-fine-tuning, all correction rounds
applied) output on the right - one page per DICOM file, matched by the
filename text embedded on each page.

Run standalone (paths are hardcoded below for this specific comparison).
"""

import os
import io
import fitz
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

ORIG_BMODE = '/Users/juze/Library/Mobile Documents/com~apple~CloudDocs/Work/Work_Ying/PhD Experiment Result/Result/ECHO/Osmotic pump/1st Round Result from Original Python Script/B mode_output_images_original script.pdf'
ORIG_MMODE = '/Users/juze/Library/Mobile Documents/com~apple~CloudDocs/Work/Work_Ying/PhD Experiment Result/Result/ECHO/Osmotic pump/1st Round Result from Original Python Script/M mode_output_images_original script.pdf'

BMODE_BASE = '/Users/juze/Downloads/EXPORTED ECHO IMAGES_DCIOM/B-Mode/output'
MMODE_BASE = '/Users/juze/Downloads/EXPORTED ECHO IMAGES_DCIOM/M-Mode/output'

# Priority order: later entries win when a filename appears in more than one
# (i.e. a file that went through multiple correction rounds uses the latest)
BMODE_LATEST_PRIORITY = [
    os.path.join(BMODE_BASE, 'output_images.pdf'),
    os.path.join(BMODE_BASE, 'run_20260729_122450_weights_BMode_finetuned_v2_PFOX_corrected', 'output_images.pdf'),
    os.path.join(BMODE_BASE, 'run_20260729_143156_weights_BMode_finetuned_v3_highEF_corrected', 'output_images.pdf'),
    os.path.join(BMODE_BASE, 'run_20260729_164222_weights_BMode_finetuned_v4_highEF_round2_corrected', 'output_images.pdf'),
]
MMODE_LATEST_PRIORITY = [
    os.path.join(MMODE_BASE, 'output_images.pdf'),
    os.path.join(MMODE_BASE, 'run_20260729_122540_weights_MMode_finetuned_v4_PFOX_corrected', 'output_images.pdf'),
]

OUT_DIR = '/Users/juze/Desktop/ECHO Analysis/qc_comparisons'
ZOOM = 2.0


def build_page_index(pdf_path):
    """Returns {filename: (pdf_path, page_index)} using the last non-empty
    text line on each page as the filename."""
    doc = fitz.open(pdf_path)
    index = {}
    for i in range(len(doc)):
        text = doc[i].get_text()
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            continue
        index[lines[-1]] = (pdf_path, i)
    doc.close()
    return index


def build_latest_index(pdf_paths_in_priority_order):
    index = {}
    source_count = {}
    for path in pdf_paths_in_priority_order:
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}, skipping")
            continue
        idx = build_page_index(path)
        index.update(idx)  # later paths override earlier ones
        source_count[path] = len(idx)
    return index, source_count


def render_page_image(pdf_path, page_idx, zoom=ZOOM):
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB')
    doc.close()
    return img


def build_comparison(mode_label, orig_pdf, latest_priority, out_path):
    print(f"=== {mode_label} ===")
    print("Indexing original PDF...")
    orig_index = build_page_index(orig_pdf)
    print(f"  {len(orig_index)} pages indexed")

    print("Indexing latest PDFs (priority order, later wins)...")
    latest_index, source_count = build_latest_index(latest_priority)
    for path, n in source_count.items():
        print(f"  {os.path.relpath(path, os.path.dirname(BMODE_BASE) if 'B-Mode' in path else os.path.dirname(MMODE_BASE))}: {n} pages")
    print(f"  {len(latest_index)} unique filenames in latest (after priority merge)")

    common = sorted(set(orig_index) & set(latest_index))
    only_orig = sorted(set(orig_index) - set(latest_index))
    only_latest = sorted(set(latest_index) - set(orig_index))
    print(f"Matched (in both): {len(common)}")
    print(f"Only in original (no longer in dataset?): {len(only_orig)}")
    print(f"Only in latest (new files added this session, no original to compare): {len(only_latest)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with PdfPages(out_path) as pdf:
        for name in common:
            orig_path, orig_pg = orig_index[name]
            latest_path, latest_pg = latest_index[name]

            left_img = render_page_image(orig_path, orig_pg)
            right_img = render_page_image(latest_path, latest_pg)

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            axes[0].imshow(left_img)
            axes[0].set_title('ORIGINAL (pre fine-tune)', fontsize=12)
            axes[0].axis('off')
            axes[1].imshow(right_img)
            axes[1].set_title('LATEST (post fine-tune)', fontsize=12)
            axes[1].axis('off')
            fig.suptitle(name, fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, dpi=110)
            plt.close(fig)

        d = pdf.infodict()
        d['Title'] = f'{mode_label} QC Comparison - Original vs Latest'
        d['Subject'] = 'Left: pre-fine-tuning baseline. Right: current best (all correction rounds applied).'

    print(f"Saved: {out_path}\n")
    return len(common), only_orig, only_latest


if __name__ == '__main__':
    n_b, only_orig_b, only_latest_b = build_comparison(
        'B-mode', ORIG_BMODE, BMODE_LATEST_PRIORITY,
        os.path.join(OUT_DIR, 'QC_Comparison_Bmode.pdf'))

    n_m, only_orig_m, only_latest_m = build_comparison(
        'M-mode', ORIG_MMODE, MMODE_LATEST_PRIORITY,
        os.path.join(OUT_DIR, 'QC_Comparison_Mmode.pdf'))

    print("=== Summary ===")
    print(f"B-mode: {n_b} pages compared")
    print(f"M-mode: {n_m} pages compared")
    if only_latest_b:
        print(f"\nB-mode files with no original to compare (new this session): {len(only_latest_b)}")
        for f in only_latest_b:
            print("  ", f)
    if only_latest_m:
        print(f"\nM-mode files with no original to compare (new this session): {len(only_latest_m)}")
        for f in only_latest_m:
            print("  ", f)
