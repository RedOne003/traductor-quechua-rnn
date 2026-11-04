# -*- coding: utf-8 -*-
"""Exporta el modelo pequeno (LSTM) a JSON plano para correrlo 100% en el
navegador (sin servidor), listo para Netlify estatico."""
import json
import pickle
from pathlib import Path

import numpy as np

from model import build_training_model

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "processed_small"
RUNS_DIR = ROOT / "runs_small" / "lstm"
OUT = ROOT / "webapp"
OUT.mkdir(exist_ok=True)

with open(PROC_DIR / "tokenizer_es.pkl", "rb") as f:
    tok_es = pickle.load(f)
with open(PROC_DIR / "tokenizer_qu.pkl", "rb") as f:
    tok_qu = pickle.load(f)

vocab_es = len(tok_es.word_index) + 1
vocab_qu = len(tok_qu.word_index) + 1

model, encoder_model, decoder_model = build_training_model(vocab_es, vocab_qu, "lstm", embedding_dim=64, units=128)
encoder_model.load_weights(str(RUNS_DIR / "encoder.weights.h5"))
decoder_model.load_weights(str(RUNS_DIR / "decoder.weights.h5"))


def W(layer_name, model_obj):
    layer = model_obj.get_layer(layer_name)
    return [w.tolist() for w in layer.get_weights()]


data = {
    "word_index_es": tok_es.word_index,
    "word_index_qu": tok_qu.word_index,
    "index_word_qu": {v: k for k, v in tok_qu.word_index.items()},
    "units": 128,
    "sos_id": tok_qu.word_index.get("<sos>"),
    "eos_id": tok_qu.word_index.get("<eos>"),
    "max_len": 10,
    "encoder_embedding": W("encoder_embedding", encoder_model),
    "encoder_lstm": W("encoder_lstm", encoder_model),
    "decoder_embedding": W("decoder_embedding", decoder_model),
    "decoder_lstm": W("decoder_lstm", decoder_model),
    "decoder_output": W("decoder_output", decoder_model),
}

with open(OUT / "model.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

print("Exportado a", OUT / "model.json")
