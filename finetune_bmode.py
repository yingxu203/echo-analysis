# -*- coding: utf-8 -*-
"""
finetune_bmode.py

Fine-tunes the existing B-Mode U-Net model on your manually-corrected
training examples (created with annotate_bmode.py).

USAGE:
    python finetune_bmode.py

This will:
  1. Load all image/mask pairs from ./training_data_bmode/
  2. Split them into training and validation sets
  3. Load the EXISTING model weights (not training from scratch)
  4. Continue training ("fine-tuning") using a low learning rate, so it
     nudges the model toward your corrections without forgetting everything
     it already learned
  5. Save the new weights to a NEW file (does not overwrite the original),
     so you can compare before/after and roll back if needed
"""

import os
import glob
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from util_nn_bmode import get_unet, dice_coef_loss, dice_coef

DATA_DIR = 'training_data_bmode'
# Round 1: start from the ORIGINAL base weights
# Round 2+: point this at your previous best result to keep building on progress
# weights_BMode_finetuned_v3.h5 is the most recent round (1st high-EF-outlier
# correction), so this round (2nd high-EF-outlier correction batch) builds on that.
WEIGHTS_IN = './model_weights/weights_BMode_finetuned_v3.h5'
WEIGHTS_OUT = './model_weights/weights_BMode_finetuned_v4.h5'

# Training hyperparameters - conservative defaults for fine-tuning on a small dataset
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
EPOCHS = 100          # increased ceiling - early stopping will halt sooner if needed
PATIENCE = 15         # stop if val_loss hasn't improved for this many epochs
VALIDATION_FRACTION = 0.15

# Must match echoanalysis_main.py exactly
MEAN, STD = 92, 57


def load_data():
    image_files = sorted(glob.glob(os.path.join(DATA_DIR, '*_image.npy')))
    images = []
    masks = []
    for img_f in image_files:
        mask_f = img_f.replace('_image.npy', '_mask.npy')
        if not os.path.exists(mask_f):
            continue
        images.append(np.load(img_f))
        masks.append(np.load(mask_f))

    if len(images) == 0:
        raise RuntimeError(
            f"No training pairs found in {DATA_DIR}. "
            f"Run annotate_bmode.py first to create some."
        )

    images = np.stack(images)[..., np.newaxis].astype(np.float32)
    masks = np.stack(masks)[..., np.newaxis].astype(np.float32)
    return images, masks


def main():
    images, masks = load_data()
    print(f"Loaded {images.shape[0]} training examples from {DATA_DIR}")

    if images.shape[0] < 10:
        print("WARNING: fewer than 10 examples found. Fine-tuning may not "
              "generalize well with this few examples - consider annotating more "
              "before relying heavily on the result.")

    # Standardize using the SAME mean/std as original training
    images = (images - MEAN) / STD

    # Train/validation split
    n = images.shape[0]
    n_val = max(1, int(round(n * VALIDATION_FRACTION)))
    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, y_train = images[train_idx], masks[train_idx]
    X_val, y_val = images[val_idx], masks[val_idx]

    print(f"Training on {X_train.shape[0]} examples, validating on {X_val.shape[0]}")

    print("Loading existing model weights...")
    model = get_unet()
    model.load_weights(WEIGHTS_IN)

    # Recompile with a LOW learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=dice_coef_loss,
        metrics=[dice_coef]
    )

    os.makedirs(os.path.dirname(WEIGHTS_OUT), exist_ok=True)
    checkpoint_path = WEIGHTS_OUT.replace('.h5', '_best_checkpoint.h5')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True,
                         save_weights_only=True, verbose=0)
    ]

    print(f"Starting fine-tuning for up to {EPOCHS} epochs "
          f"(will stop early if val_loss hasn't improved for {PATIENCE} epochs)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # restore_best_weights=True already puts the best-epoch weights back into `model`,
    # so saving now saves the BEST version, not necessarily the last epoch
    model.save_weights(WEIGHTS_OUT)
    print(f"\nSaved BEST fine-tuned weights (by val_loss) to: {WEIGHTS_OUT}")
    print(f"(A checkpoint copy was also saved during training to: {checkpoint_path})")
    print("\nFinal training/validation metrics:")
    print(f"  train_loss: {history.history['loss'][-1]:.4f}  "
          f"train_dice: {history.history['dice_coef'][-1]:.4f}")
    print(f"  val_loss:   {history.history['val_loss'][-1]:.4f}  "
          f"val_dice:   {history.history['val_dice_coef'][-1]:.4f}")
    print("\nNOTE: the original weights file was NOT overwritten. To use the "
          "new weights, update echoanalysis_main.py to load "
          f"'{WEIGHTS_OUT}' instead of the original, after you've verified "
          "the new results look better on a few test scans.")


if __name__ == '__main__':
    main()
