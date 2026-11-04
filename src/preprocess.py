"""Preprocesamiento del corpus paralelo espanol-quechua (AmericasNLP/JW300).

- Limpia y normaliza texto.
- Filtra pares por longitud (oraciones cortas, mas rapidas de entrenar en CPU).
- Tokeniza con keras Tokenizer (nivel palabra) y agrega tokens <sos>/<eos>.
- Aplica padding y guarda tensores + tokenizers listos para entrenar.
"""
import json
import pickle
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROC_DIR = ROOT / "processed"
PROC_DIR.mkdir(exist_ok=True)

MAX_LEN = 15          # longitud maxima en palabras por oracion (antes de agregar sos/eos)
MAX_PAIRS = 30000      # submuestra del corpus de 125k pares para entrenar en tiempos razonables en CPU
VOCAB_SIZE = 12000     # tamano maximo de vocabulario por idioma
SEED = 42


def normalize(text: str, is_quechua: bool) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize("NFC", text)
    # separa signos de puntuacion del resto de la palabra
    text = re.sub(r"([.,!?¿¡;:\"'()—–])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Záéíóúñüʼ'.,!?¿¡;:0-9 ]+" if not is_quechua else r"[^a-záéíóúñüʼ'.,!?¿¡;:0-9 ]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_pairs(es_path: Path, quy_path: Path):
    with open(es_path, encoding="utf-8") as f_es, open(quy_path, encoding="utf-8") as f_qu:
        es_lines = [l.rstrip("\n") for l in f_es]
        qu_lines = [l.rstrip("\n") for l in f_qu]
    assert len(es_lines) == len(qu_lines), "Los archivos paralelos no tienen el mismo numero de lineas"
    return list(zip(es_lines, qu_lines))


def filter_and_clean(pairs):
    cleaned = []
    for es, qu in pairs:
        es_n = normalize(es, is_quechua=False)
        qu_n = normalize(qu, is_quechua=True)
        if not es_n or not qu_n:
            continue
        n_es = len(es_n.split())
        n_qu = len(qu_n.split())
        if 1 <= n_es <= MAX_LEN and 1 <= n_qu <= MAX_LEN:
            cleaned.append((es_n, qu_n))
    return cleaned


def dedupe(pairs):
    seen = set()
    out = []
    for es, qu in pairs:
        key = (es, qu)
        if key not in seen:
            seen.add(key)
            out.append((es, qu))
    return out


def build_tokenizer(sentences, num_words):
    tok = Tokenizer(num_words=num_words, oov_token="<unk>", filters="")
    tok.fit_on_texts(sentences)
    return tok


def main():
    random.seed(SEED)

    print("Cargando train/dev/test...")
    train_pairs = load_pairs(DATA_DIR / "train.es", DATA_DIR / "train.quy")
    dev_pairs = load_pairs(DATA_DIR / "dev.es", DATA_DIR / "dev.quy")

    print(f"Pares crudos: train={len(train_pairs)} dev={len(dev_pairs)}")

    train_pairs = dedupe(filter_and_clean(train_pairs))
    dev_pairs = dedupe(filter_and_clean(dev_pairs))

    print(f"Pares tras limpieza/filtro (<= {MAX_LEN} palabras): train={len(train_pairs)} dev={len(dev_pairs)}")

    random.shuffle(train_pairs)
    train_pairs = train_pairs[:MAX_PAIRS]

    # separa un split de prueba interno a partir del propio train (test.quy no es publico)
    n_test = min(1000, len(train_pairs) // 20)
    test_pairs = train_pairs[:n_test]
    train_pairs = train_pairs[n_test:]

    print(f"Split final: train={len(train_pairs)} dev={len(dev_pairs)} test(interno)={len(test_pairs)}")

    es_train = [f"<sos> {es} <eos>" for es, _ in train_pairs]
    qu_train = [f"<sos> {qu} <eos>" for _, qu in train_pairs]

    tok_es = build_tokenizer(es_train, VOCAB_SIZE)
    tok_qu = build_tokenizer(qu_train, VOCAB_SIZE)

    def encode(tokenizer, texts, maxlen):
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post")

    maxlen = MAX_LEN + 2  # + <sos> + <eos>

    def make_split(pairs, name):
        es_txt = [f"<sos> {es} <eos>" for es, _ in pairs]
        qu_txt = [f"<sos> {qu} <eos>" for _, qu in pairs]
        X = encode(tok_es, es_txt, maxlen)
        Y = encode(tok_qu, qu_txt, maxlen)
        np.save(PROC_DIR / f"X_{name}.npy", X)
        np.save(PROC_DIR / f"Y_{name}.npy", Y)
        with open(PROC_DIR / f"pairs_{name}.json", "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False)
        return X, Y

    make_split(train_pairs, "train")
    make_split(dev_pairs, "dev")
    make_split(test_pairs, "test")

    with open(PROC_DIR / "tokenizer_es.pkl", "wb") as f:
        pickle.dump(tok_es, f)
    with open(PROC_DIR / "tokenizer_qu.pkl", "wb") as f:
        pickle.dump(tok_qu, f)

    meta = {
        "max_len": maxlen,
        "vocab_size_es": min(VOCAB_SIZE, len(tok_es.word_index) + 1),
        "vocab_size_qu": min(VOCAB_SIZE, len(tok_qu.word_index) + 1),
        "n_train": len(train_pairs),
        "n_dev": len(dev_pairs),
        "n_test": len(test_pairs),
        "source": "AmericasNLP 2021 quechua-spanish (JW300, Agic & Vulic 2019)",
    }
    with open(PROC_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Listo. Metadatos:", meta)


if __name__ == "__main__":
    main()
