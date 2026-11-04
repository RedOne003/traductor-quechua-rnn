"""Entrena el traductor es->quechua con RNN simple o LSTM.

Uso:
    py -3.10 src/train.py rnn
    py -3.10 src/train.py lstm
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from model import build_training_model

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "processed"
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)

EPOCHS = 15
BATCH_SIZE = 64
SEED = 42


def load_data():
    X_train = np.load(PROC_DIR / "X_train.npy")
    Y_train = np.load(PROC_DIR / "Y_train.npy")
    X_dev = np.load(PROC_DIR / "X_dev.npy")
    Y_dev = np.load(PROC_DIR / "Y_dev.npy")
    with open(PROC_DIR / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return X_train, Y_train, X_dev, Y_dev, meta


def main():
    cell_type = sys.argv[1] if len(sys.argv) > 1 else "lstm"
    assert cell_type in ("rnn", "lstm")

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    X_train, Y_train, X_dev, Y_dev, meta = load_data()
    vocab_es = meta["vocab_size_es"]
    vocab_qu = meta["vocab_size_qu"]

    # teacher forcing: entrada del decoder = Y sin el ultimo token, objetivo = Y sin el primero
    dec_in_train = Y_train[:, :-1]
    dec_tgt_train = Y_train[:, 1:]
    dec_in_dev = Y_dev[:, :-1]
    dec_tgt_dev = Y_dev[:, 1:]

    model, encoder_model, decoder_model = build_training_model(vocab_es, vocab_qu, cell_type)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    out_dir = RUNS_DIR / cell_type
    out_dir.mkdir(exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(out_dir / "best.weights.h5"), save_best_only=True, save_weights_only=True, monitor="val_loss"),
    ]

    t0 = time.time()
    history = model.fit(
        [X_train, dec_in_train],
        dec_tgt_train,
        validation_data=([X_dev, dec_in_dev], dec_tgt_dev),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )
    train_seconds = time.time() - t0

    model.save_weights(str(out_dir / "final.weights.h5"))
    encoder_model.save_weights(str(out_dir / "encoder.weights.h5"))
    decoder_model.save_weights(str(out_dir / "decoder.weights.h5"))

    n_params = model.count_params()
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({
            "cell_type": cell_type,
            "history": {k: [float(x) for x in v] for k, v in history.history.items()},
            "train_seconds": train_seconds,
            "n_params": int(n_params),
            "epochs_ran": len(history.history["loss"]),
        }, f, ensure_ascii=False, indent=2)

    print(f"[{cell_type}] entrenamiento terminado en {train_seconds:.1f}s, {n_params:,} parametros")


if __name__ == "__main__":
    main()
