"""Version reducida del corpus: solo oraciones cortas formadas por las
palabras mas frecuentes, para que el modelo aprenda bien con poco tiempo
de entrenamiento (vocabulario pequeno converge mucho mas rapido)."""
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import load_pairs, filter_and_clean, dedupe, DATA_DIR

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "processed_small"
PROC_DIR.mkdir(exist_ok=True)

MAX_LEN = 8
TOP_WORDS = 400
SEED = 42


def main():
    random.seed(SEED)
    train_pairs = dedupe(filter_and_clean(load_pairs(DATA_DIR / "train.es", DATA_DIR / "train.quy")))
    dev_pairs = dedupe(filter_and_clean(load_pairs(DATA_DIR / "dev.es", DATA_DIR / "dev.quy")))

    train_pairs = [(es, qu) for es, qu in train_pairs if len(es.split()) <= MAX_LEN and len(qu.split()) <= MAX_LEN]

    es_counter = Counter(w for es, _ in train_pairs for w in es.split())
    qu_counter = Counter(w for _, qu in train_pairs for w in qu.split())
    top_es = {w for w, _ in es_counter.most_common(TOP_WORDS)}
    top_qu = {w for w, _ in qu_counter.most_common(TOP_WORDS)}

    small_pairs = [(es, qu) for es, qu in train_pairs
                   if set(es.split()) <= top_es and set(qu.split()) <= top_qu]
    random.shuffle(small_pairs)

    n_test = min(200, len(small_pairs) // 10)
    test_pairs = small_pairs[:n_test]
    train_small = small_pairs[n_test:]

    dev_small = [(es, qu) for es, qu in dev_pairs
                 if len(es.split()) <= MAX_LEN and len(qu.split()) <= MAX_LEN
                 and set(es.split()) <= top_es and set(qu.split()) <= top_qu]
    if not dev_small:
        dev_small = train_small[:100]
        train_small = train_small[100:]

    print(f"small: train={len(train_small)} dev={len(dev_small)} test={len(test_pairs)}")

    es_texts = [f"<sos> {es} <eos>" for es, _ in train_small]
    qu_texts = [f"<sos> {qu} <eos>" for _, qu in train_small]
    tok_es = Tokenizer(num_words=TOP_WORDS + 10, oov_token="<unk>", filters="")
    tok_es.fit_on_texts(es_texts)
    tok_qu = Tokenizer(num_words=TOP_WORDS + 10, oov_token="<unk>", filters="")
    tok_qu.fit_on_texts(qu_texts)

    maxlen = MAX_LEN + 2

    def make_split(pairs, name):
        es_txt = [f"<sos> {es} <eos>" for es, _ in pairs]
        qu_txt = [f"<sos> {qu} <eos>" for _, qu in pairs]
        X = pad_sequences(tok_es.texts_to_sequences(es_txt), maxlen=maxlen, padding="post", truncating="post")
        Y = pad_sequences(tok_qu.texts_to_sequences(qu_txt), maxlen=maxlen, padding="post", truncating="post")
        np.save(PROC_DIR / f"X_{name}.npy", X)
        np.save(PROC_DIR / f"Y_{name}.npy", Y)
        with open(PROC_DIR / f"pairs_{name}.json", "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False)

    make_split(train_small, "train")
    make_split(dev_small, "dev")
    make_split(test_pairs, "test")

    import pickle
    with open(PROC_DIR / "tokenizer_es.pkl", "wb") as f:
        pickle.dump(tok_es, f)
    with open(PROC_DIR / "tokenizer_qu.pkl", "wb") as f:
        pickle.dump(tok_qu, f)

    meta = {
        "max_len": maxlen,
        "vocab_size_es": min(TOP_WORDS + 10, len(tok_es.word_index) + 1),
        "vocab_size_qu": min(TOP_WORDS + 10, len(tok_qu.word_index) + 1),
        "n_train": len(train_small),
        "n_dev": len(dev_small),
        "n_test": len(test_pairs),
        "top_words": TOP_WORDS,
    }
    with open(PROC_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(meta)


if __name__ == "__main__":
    main()
