"""Evalua un modelo (rnn o lstm) con BLEU sobre el split de prueba interno
y genera un reporte comparativo cuando ambos modelos ya fueron entrenados.

Uso:
    py -3.10 src/evaluate.py rnn
    py -3.10 src/evaluate.py lstm
    py -3.10 src/evaluate.py compare
"""
import json
import sys
from pathlib import Path

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

from inference import Translator

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "processed"
RUNS_DIR = ROOT / "runs"

N_EVAL = 40  # numero de oraciones del test interno a evaluar (BLEU es costoso de generar)


def evaluate(cell_type):
    with open(PROC_DIR / "pairs_test.json", encoding="utf-8") as f:
        pairs = json.load(f)
    pairs = pairs[:N_EVAL]

    translator = Translator(cell_type)
    smoothie = SmoothingFunction().method4

    refs, hyps = [], []
    examples = []
    for es, qu_ref in pairs:
        hyp = translator.translate(es)
        refs.append([qu_ref.split()])
        hyps.append(hyp.split())
        if len(examples) < 10:
            examples.append({"es": es, "referencia": qu_ref, "prediccion": hyp})

    bleu = corpus_bleu(refs, hyps, smoothing_function=smoothie)

    result = {
        "cell_type": cell_type,
        "n_eval": len(pairs),
        "bleu": bleu,
        "ejemplos": examples,
    }
    out_dir = RUNS_DIR / cell_type
    with open(out_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[{cell_type}] BLEU (corpus, n={len(pairs)}): {bleu:.4f}")
    for ex in examples[:5]:
        print(" ES :", ex["es"])
        print(" REF:", ex["referencia"])
        print(" HYP:", ex["prediccion"])
        print()
    return result


def compare():
    report = {}
    for cell_type in ("rnn", "lstm"):
        eval_path = RUNS_DIR / cell_type / "eval.json"
        hist_path = RUNS_DIR / cell_type / "history.json"
        with open(eval_path, encoding="utf-8") as f:
            ev = json.load(f)
        with open(hist_path, encoding="utf-8") as f:
            hist = json.load(f)
        report[cell_type] = {
            "bleu": ev["bleu"],
            "n_params": hist["n_params"],
            "train_seconds": hist["train_seconds"],
            "epochs_ran": hist["epochs_ran"],
            "final_train_loss": hist["history"]["loss"][-1],
            "final_val_loss": hist["history"]["val_loss"][-1],
            "final_train_acc": hist["history"]["accuracy"][-1],
            "final_val_acc": hist["history"]["val_accuracy"][-1],
        }
    with open(RUNS_DIR / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if arg == "compare":
        compare()
    else:
        evaluate(arg)
