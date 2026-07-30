# ECHO Analysis — User Manual for MacOS

This guide assumes you have never used this computer or this tool before.
Follow it top to bottom the first time; after that, skip straight to
**Part 2**.

Everything in this guide is run from a program called **Terminal**
(search for it with `Cmd + Space`, type "Terminal", press Enter). You
type or paste a command, press Enter, and wait for it to finish before
typing the next one.

---

## Part 1 — One-Time Computer Setup

Skip this whole part if `conda activate EchoDataAnalysis` already works
on this machine (see Part 2, Step 1).

### 1. Install Anaconda
Download from https://www.anaconda.com/download (choose the Apple
Silicon/arm64 installer for M-series Macs) and install it.

### 2. Create the environment
```bash
conda create -n EchoDataAnalysis python=3.9
```
Type `y` and press Enter when asked to proceed.

### 3. Activate it and install the required packages
```bash
conda activate EchoDataAnalysis
python -m pip install numpy pandas pydicom scikit-image scipy matplotlib opencv-python openpyxl tensorflow-macos tensorflow-metal keras==2.15.0
```
This takes a few minutes. If you see numpy conflict errors afterward, run:
```bash
python -m pip install "numpy==1.26.4"
```

### 4. Confirm the model weights are present
```bash
cd ~/Desktop/"ECHO Analysis"
ls model_weights
```
You should see a list of `.h5` files. If that folder is missing entirely,
something didn't copy over correctly — ask whoever gave you this folder
for a fresh copy rather than trying to recreate the weights.

That's it for one-time setup. Everything below is what you'll actually
run day to day.

---

## Part 2 — Every Time You Use This Tool

### Step 1: Open Terminal and activate the environment
```bash
conda activate EchoDataAnalysis
```
No output means success. Your prompt should now start with
`(EchoDataAnalysis)`.

### Step 2: Go to the tool's folder
```bash
cd ~/Desktop/"ECHO Analysis"
```
(If your copy lives somewhere else, use that path instead.)

Every command in this manual assumes you're sitting in this folder and
have the environment activated. If a command fails with something like
`No such file or directory` or `ModuleNotFoundError`, the most common
cause is forgetting one of these two steps.

---

## Part 3 — Running an Analysis

There are two ways to do this: the point-and-click app, or three
individual commands that give you more control (and are what actually
produces the full comparison report). **If you want the Excel workbook
and comparison figures across treatment groups, you need the command-line
path (Option B) — the app only produces per-file measurements.**

### Option A: The point-and-click app (Echo Segmenter)

```bash
python Echo_Segmenter_UI_Full.py
```
A window will open after a few seconds.

1. Click **Select** and choose the folder containing your exported DICOM
   images (it automatically sorts B-mode and M-mode files if they're
   mixed together in one folder).
2. Leave **Save QC Results** checked if you want a PDF of the
   segmentation overlays, not just the numbers.
3. Click **Run**, then wait — a pop-up tells you when it's done.
4. Your results are inside an `output` subfolder of whatever folder you
   selected (see "Where results land" below).

This runs the same underlying scripts as Option B below, just for one
folder at a time and without the cross-treatment comparison step.

### Option B: Command line, step by step

**Step 1 — Analyze your B-mode images:**
```bash
python echoanalysis_main.py --input_dir "/path/to/your/B-Mode/folder" --verbose
```

**Step 2 — Analyze your M-mode images:**
```bash
python echoanalysis_mmode_main.py --input_dir "/path/to/your/M-Mode/folder" --verbose
```
(Steps 1 and 2 don't depend on each other — run them in either order, or
in two separate Terminal tabs at the same time to save time. Each takes
roughly 4-5 seconds per DICOM file, so a 250-file folder takes about
15-20 minutes.)

Drop the `--verbose` flag if you don't want the QC PDF (just the numbers,
faster).

**Step 3 — Generate the full comparison report** (figures + Excel,
comparing treatment groups across Day 0/7/14):
```bash
python analyze_study.py --bmode_csv "/path/to/B-Mode/output/<run folder>/output.csv" --mmode_csv "/path/to/M-Mode/output/<run folder>/output.csv" --out_dir "./study_results"
```
Steps 1 and 2 each print their own output folder path when they finish
(look for the line `This run's output folder: ...`) — copy that exact
path in for `--bmode_csv`/`--mmode_csv` above, pointing at the
`output.csv` file inside it.

Optional flag:
- `--min_heart_rate 400` — drop any M-mode capture below this heart
  rate before averaging (bad anesthesia depth or a bad capture tends to
  show up as an implausibly low heart rate). Default is 400 if you don't
  pass this.

---

## Part 4 — Where Results Land

**Nothing is ever overwritten.** Every run creates its own new,
timestamped folder, so old results are always still there.

**From Steps 1/2** (`echoanalysis_main.py` / `echoanalysis_mmode_main.py`):
```
<your input folder>/output/run_<timestamp>_<weights file used>/
    output.csv           <- one row per DICOM file, all the measurements
    output_images.pdf    <- segmentation overlay snapshots (if --verbose)
```

**From Step 3** (`analyze_study.py`):
```
study_results/run_<timestamp>_HRge<N>_completeDaysOnly/
    study_results.xlsx
    figures/
        ALL_POINTS_INCLUDING_OUTLIERS/   <- clean data, nothing statistically excluded
        OUTLIERS_REMOVED/                <- same data, extreme values removed (check bad points are gone HERE)
```

What "clean" means for Step 3: only animals with data at **all three**
of Day 0, 7, and 14 are included, and (for M-mode) any capture with heart
rate below the `--min_heart_rate` threshold is dropped before averaging.
Within `study_results.xlsx`, sheets ending in `_noOut` are the
outliers-removed statistics; sheets without that suffix include
everything. `Bmode_excluded_incomplete` / `Mmode_excluded_incomplete` and
`Mmode_excluded_lowHR` list exactly which animals/captures got dropped
and why.

---

## Part 5 — Advanced: Fixing Bad Segmentations by Hand

Use this only if you've looked at the results and found the model is
clearly getting specific images wrong (e.g. an impossible negative
ejection fraction). This is a multi-step process — ask for help walking
through it rather than running it blind:

1. **`annotate_bmode.py`** / **`annotate_mmode.py`** — click-trace the
   correct boundary on a handful of problem images by hand. Saves
   training examples to `training_data_bmode` / `training_data_mmode`.
   ```bash
   python annotate_bmode.py "/path/to/dicom/folder"
   ```
2. **`finetune_bmode.py`** / **`finetune_mmode.py`** — retrains the
   model on your corrections (a few minutes). Produces a new `.h5` file
   in `model_weights/`; does **not** overwrite the old one.
   ```bash
   python finetune_bmode.py
   ```
3. Update the `WEIGHTS_PATH` line near the top of `echoanalysis_main.py`
   (or `echoanalysis_mmode_main.py`) to point at the new weights file,
   **only after** confirming on a few test images that it's actually
   better — a small hand-annotated batch can also make things worse if
   you're not careful.

---

## Troubleshooting

| Message contains... | What it means | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'X'` | A package isn't installed | `python -m pip install X` |
| `sh: activate: command not found` | Harmless internal message | Ignore, doesn't stop anything |
| `Illegal instruction: 4` | Wrong TensorFlow build for this Mac's chip | Reinstall `tensorflow-macos` + `tensorflow-metal`, not plain `tensorflow` |
| `FileNotFoundError` mentioning a `.h5` file | Model weights missing or wrong path | Check `model_weights/` folder and the `WEIGHTS_PATH` line in the script |
| `No such file or directory` on `--input_dir` | Path typo, or forgot the quotes around a path with spaces | Always wrap paths in double quotes |
| Terminal looks stuck | Might be mid-run (large folders take 15-20 min) or waiting on a previous command | Give it a few minutes; if truly stuck, `Ctrl + C` then retry |
| Numbers look wrong for one specific treatment group only | Model may be mis-segmenting that group's images specifically | See Part 5 |

---

## Quick Reference — All Commands

```bash
# One-time setup
conda create -n EchoDataAnalysis python=3.9
conda activate EchoDataAnalysis
python -m pip install numpy pandas pydicom scikit-image scipy matplotlib opencv-python openpyxl tensorflow-macos tensorflow-metal keras==2.15.0

# Every session
conda activate EchoDataAnalysis
cd ~/Desktop/"ECHO Analysis"

# Point-and-click
python Echo_Segmenter_UI_Full.py

# Command line: full pipeline
python echoanalysis_main.py --input_dir "/path/to/B-Mode" --verbose
python echoanalysis_mmode_main.py --input_dir "/path/to/M-Mode" --verbose
python analyze_study.py --bmode_csv "<B-Mode output.csv path>" --mmode_csv "<M-Mode output.csv path>" --out_dir "./study_results"

# Advanced: hand-correct + retrain
python annotate_bmode.py "/path/to/dicom/folder"
python finetune_bmode.py
```

---

## Status as of this writing (2026-07-28)

- B-mode is currently using `weights_BMode_finetuned.h5` officially.
- M-mode is using `weights_MMode_finetuned_v3.h5` officially.
- For personal preference, any file name contains WT/WF/WM the QC output will generate blue#0000FF lines, OX/OF/OM will generate red#FF2C2C QC output lines.
- The analysis is set to compare the data in order from day0 to day7 till day14 to form trends for final figures.
