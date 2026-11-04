# Traductor Español → Quechua con Redes Neuronales Recurrentes

Proyecto del curso **Inteligencia Artificial** (UAC, Ingeniería de Sistemas) — comparación
de un traductor neuronal secuencia-a-secuencia construido con **RNN simple** y con **LSTM**.

## Estructura

```
traductor-quechua-rnn/
├── data/                  # corpus crudo (se descarga, no se versiona)
├── processed/             # tensores, tokenizers y metadatos (generado)
├── runs/
│   ├── rnn/               # pesos, historial y evaluación del modelo RNN simple
│   ├── lstm/              # pesos, historial y evaluación del modelo LSTM
│   └── comparison.json    # comparación final RNN vs LSTM
├── src/
│   ├── download_data.py   # descarga el corpus paralelo es-quechua
│   ├── preprocess.py       # limpieza, tokenización, vocabulario y padding
│   ├── model.py             # arquitectura seq2seq parametrizable (rnn | lstm)
│   ├── train.py              # entrenamiento con teacher forcing
│   ├── inference.py          # decodificación greedy paso a paso
│   ├── evaluate.py           # evaluación BLEU y comparación
│   └── translate.py          # demo de traducción por consola
└── informe/                # informe en PDF
```

## Dataset

Corpus paralelo español–quechua (variante Ayacucho, `quy`) del shared task
[AmericasNLP 2021](https://github.com/AmericasNLP/americasnlp2021), basado en
JW300 (Agić & Vulić, 2019; Tiedemann, 2012). 125,008 pares originales; tras
limpieza y filtrado por longitud (≤ 15 palabras) se usó una submuestra de
30,000 pares (29,000 train / 758 dev / 1,000 test interno) para que el
entrenamiento sea viable en CPU.

```bash
py -3.10 src/download_data.py
```

## Cómo reproducir

```bash
pip install -r requirements.txt
py -3.10 src/download_data.py
py -3.10 src/preprocess.py
py -3.10 src/train.py rnn
py -3.10 src/train.py lstm
py -3.10 src/evaluate.py rnn
py -3.10 src/evaluate.py lstm
py -3.10 src/evaluate.py compare
py -3.10 src/translate.py lstm "Buenos días, cómo estás"
```

## Arquitectura

Encoder–decoder clásico:

- **Encoder**: `Embedding(128)` → capa recurrente (`SimpleRNN` o `LSTM`, 256 unidades) que
  produce el/los estado(s) final(es).
- **Decoder**: `Embedding(128)` → capa recurrente del mismo tipo, inicializada con el
  estado del encoder, con *teacher forcing* durante el entrenamiento → `Dense(vocab, softmax)`.
- Ambos modelos comparten exactamente la misma dimensión de embedding, unidades,
  optimizador (Adam), función de pérdida (`sparse_categorical_crossentropy`) y datos,
  para que la única variable sea el tipo de celda recurrente.

## Resultados

Ver [`runs/comparison.json`](runs/comparison.json) y el informe en `informe/`
para la comparación completa (BLEU, pérdida, tiempo de entrenamiento, ejemplos
de traducción) y el análisis RNN simple vs LSTM.

## Licencia de los datos

El corpus se reutiliza bajo los términos del shared task AmericasNLP 2021. Cítese:

```
Agić & Vulić (2019). JW300: A Wide-Coverage Parallel Corpus for Low-Resource Languages. ACL.
Tiedemann (2012). Parallel Data, Tools and Interfaces in OPUS. LREC.
```
