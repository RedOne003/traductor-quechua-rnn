"""Carga un modelo entrenado (rnn o lstm) y traduce oraciones es -> quechua
usando decodificacion greedy paso a paso sobre los modelos de inferencia."""
import pickle
from pathlib import Path

import numpy as np

from model import build_training_model, UNITS
from preprocess import normalize

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "processed"
RUNS_DIR = ROOT / "runs"


class Translator:
    def __init__(self, cell_type):
        assert cell_type in ("rnn", "lstm")
        self.cell_type = cell_type

        with open(PROC_DIR / "tokenizer_es.pkl", "rb") as f:
            self.tok_es = pickle.load(f)
        with open(PROC_DIR / "tokenizer_qu.pkl", "rb") as f:
            self.tok_qu = pickle.load(f)

        vocab_es = min(self.tok_es.num_words or len(self.tok_es.word_index) + 1, len(self.tok_es.word_index) + 1)
        vocab_qu = min(self.tok_qu.num_words or len(self.tok_qu.word_index) + 1, len(self.tok_qu.word_index) + 1)

        _, self.encoder_model, self.decoder_model = build_training_model(vocab_es, vocab_qu, cell_type)

        out_dir = RUNS_DIR / cell_type
        self.encoder_model.load_weights(str(out_dir / "encoder.weights.h5"))
        self.decoder_model.load_weights(str(out_dir / "decoder.weights.h5"))

        self.idx2word_qu = {i: w for w, i in self.tok_qu.word_index.items()}
        self.sos_id = self.tok_qu.word_index.get("<sos>")
        self.eos_id = self.tok_qu.word_index.get("<eos>")
        self.max_len = 19

    def translate(self, sentence_es: str, max_len=None) -> str:
        max_len = max_len or self.max_len
        text = f"<sos> {normalize(sentence_es, is_quechua=False)} <eos>"
        seq = self.tok_es.texts_to_sequences([text])
        enc_input = seq  # variable length, batch of 1

        enc_input = np.array(enc_input, dtype="int32")
        states = self.encoder_model.predict(enc_input, verbose=0)
        if self.cell_type == "rnn":
            states = [states]

        target_seq = np.array([[self.sos_id]], dtype="int32")
        decoded_words = []
        for _ in range(max_len):
            outputs = self.decoder_model.predict([target_seq] + states, verbose=0)
            output_tokens = outputs[0]
            new_states = outputs[1:]
            next_id = int(np.argmax(output_tokens[0, -1, :]))
            if next_id == self.eos_id or next_id == 0:
                break
            word = self.idx2word_qu.get(next_id, "<unk>")
            decoded_words.append(word)
            target_seq = np.array([[next_id]], dtype="int32")
            states = new_states
        return " ".join(decoded_words)
