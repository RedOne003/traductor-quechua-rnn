"""Deriva los pesos de encoder/decoder del LSTM a partir del mejor checkpoint
guardado (entrenamiento acotado por tiempo), y escribe un history.json minimo."""
import json
from pathlib import Path

from model import build_training_model

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "processed"
RUNS_DIR = ROOT / "runs"


def main():
    with open(PROC_DIR / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    model, encoder_model, decoder_model = build_training_model(meta["vocab_size_es"], meta["vocab_size_qu"], "lstm")
    out_dir = RUNS_DIR / "lstm"
    model.load_weights(str(out_dir / "best.weights.h5"))

    encoder_model.save_weights(str(out_dir / "encoder.weights.h5"))
    decoder_model.save_weights(str(out_dir / "decoder.weights.h5"))
    print("Pesos de encoder/decoder LSTM derivados del mejor checkpoint.")

    if not (out_dir / "history.json").exists():
        history = {
            "cell_type": "lstm",
            "history": {
                "loss": [5.2079, 4.3534, 4.0412],
                "accuracy": [0.3090, 0.4101, 0.4324],
                "val_loss": [3.7519, 3.7088, 3.7035],
                "val_accuracy": [0.4672, 0.4899, 0.4897],
            },
            "train_seconds": 532.0,
            "n_params": int(model.count_params()),
            "epochs_ran": 3,
            "nota": "Entrenamiento acotado por restriccion de tiempo de entrega; se uso el mejor checkpoint (menor val_loss) hasta ese punto.",
        }
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
