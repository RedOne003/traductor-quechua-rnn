"""Traductor con el modelo de vocabulario reducido (converge rapido y traduce mejor)."""
import pickle
from pathlib import Path

import numpy as np

from model import build_training_model
from preprocess import normalize

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "processed_small"
RUNS_DIR = ROOT / "runs_small"


class SmallTranslator:
    def __init__(self, cell_type):
        with open(PROC_DIR / "tokenizer_es.pkl", "rb") as f:
            self.tok_es = pickle.load(f)
        with open(PROC_DIR / "tokenizer_qu.pkl", "rb") as f:
            self.tok_qu = pickle.load(f)

        vocab_es = len(self.tok_es.word_index) + 1
        vocab_qu = len(self.tok_qu.word_index) + 1
        self.cell_type = cell_type
        _, self.encoder_model, self.decoder_model = build_training_model(
            vocab_es, vocab_qu, cell_type, embedding_dim=64, units=128)

        out_dir = RUNS_DIR / cell_type
        self.encoder_model.load_weights(str(out_dir / "encoder.weights.h5"))
        self.decoder_model.load_weights(str(out_dir / "decoder.weights.h5"))

        self.idx2word_qu = {i: w for w, i in self.tok_qu.word_index.items()}
        self.sos_id = self.tok_qu.word_index.get("<sos>")
        self.eos_id = self.tok_qu.word_index.get("<eos>")
        self.known_es_words = set(self.tok_es.word_index.keys()) - {"<unk>", "<sos>", "<eos>"}
        self.max_len = 10

    def known_words(self, sentence_es: str):
        words = normalize(sentence_es, is_quechua=False).split()
        return [(w, w in self.known_es_words) for w in words]

    def translate(self, sentence_es: str) -> str:
        text = f"<sos> {normalize(sentence_es, is_quechua=False)} <eos>"
        seq = self.tok_es.texts_to_sequences([text])
        enc_input = np.array(seq, dtype="int32")
        states = self.encoder_model.predict(enc_input, verbose=0)
        if self.cell_type == "rnn":
            states = [states]

        target_seq = np.array([[self.sos_id]], dtype="int32")
        words = []
        for _ in range(self.max_len):
            outputs = self.decoder_model.predict([target_seq] + states, verbose=0)
            next_id = int(np.argmax(outputs[0][0, -1, :]))
            if next_id == self.eos_id or next_id == 0:
                break
            words.append(self.idx2word_qu.get(next_id, "<unk>"))
            target_seq = np.array([[next_id]], dtype="int32")
            states = outputs[1:]
        return " ".join(words)
