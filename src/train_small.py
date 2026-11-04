"""Entrena la version reducida (vocabulario chico) para que converja rapido.

Uso: py -3.10 src/train_small.py rnn|lstm
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from model import build_training_model

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "processed_small"
RUNS_DIR = ROOT / "runs_small"
RUNS_DIR.mkdir(exist_ok=True)

EPOCHS = 60
BATCH_SIZE = 32


def main():
    cell_type = sys.argv[1]
    X_train = np.load(PROC_DIR / "X_train.npy")
    Y_train = np.load(PROC_DIR / "Y_train.npy")
    X_dev = np.load(PROC_DIR / "X_dev.npy")
    Y_dev = np.load(PROC_DIR / "Y_dev.npy")
    with open(PROC_DIR / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    model, encoder_model, decoder_model = build_training_model(
        meta["vocab_size_es"], meta["vocab_size_qu"], cell_type, embedding_dim=64, units=128)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    out_dir = RUNS_DIR / cell_type
    out_dir.mkdir(exist_ok=True)
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ]

    t0 = time.time()
    hist = model.fit([X_train, Y_train[:, :-1]], Y_train[:, 1:],
                      validation_data=([X_dev, Y_dev[:, :-1]], Y_dev[:, 1:]),
                      batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=cb, verbose=2)
    dt = time.time() - t0

    encoder_model.save_weights(str(out_dir / "encoder.weights.h5"))
    decoder_model.save_weights(str(out_dir / "decoder.weights.h5"))
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({"cell_type": cell_type, "train_seconds": dt, "epochs_ran": len(hist.history["loss"]),
                   "final_val_acc": hist.history["val_accuracy"][-1], "final_val_loss": hist.history["val_loss"][-1]}, f)
    print(f"[{cell_type}] listo en {dt:.1f}s, {len(hist.history['loss'])} epocas, val_acc={hist.history['val_accuracy'][-1]:.3f}")


if __name__ == "__main__":
    main()
