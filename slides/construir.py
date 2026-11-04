import sys, os
SKILL = os.path.expanduser('~/.claude/skills/exposicion-uac')
sys.path.insert(0, os.path.join(SKILL, 'scripts'))
from uac_deck import *

D = Deck('Traductor Español-Quechua con RNN', pie='IA · TRADUCTOR ESPAÑOL-QUECHUA CON RNN')

D.cover(kicker='INTELIGENCIA ARTIFICIAL · 2025-I',
        titulo='TRADUCTOR ESPAÑOL-QUECHUA',
        subtitulo='Comparación de arquitecturas RNN simple y LSTM para traducción automática',
        chips='RNN · LSTM · SEQ2SEQ · BLEU',
        bloque='INTEGRANTES',
        lineas=['Completar con los integrantes del equipo'],
        pieizq=['Docente: Completar'])

s = D.content('OBJETIVO Y FASES DEL PROYECTO',
              intro='Diseñar, entrenar y comparar dos traductores neuronales es→quechua: '
                    'uno con RNN simple y otro con LSTM, sobre el mismo corpus.',
              nota='Explicar que el proyecto sigue 5 fases: datos, RNN simple, LSTM, analisis e informe, exposicion.')
D.numbered(s, [
    ('1. Datos', 'Corpus paralelo público es-quechua, limpieza y tokenización.'),
    ('2. RNN simple', 'Arquitectura encoder-decoder con SimpleRNN.'),
    ('3. LSTM', 'Misma arquitectura, celda LSTM con memoria de largo plazo.'),
    ('4. Comparación', 'Métricas BLEU, pérdida y análisis cualitativo.'),
], ncols=2)

s = D.content('EL CORPUS: ESPAÑOL-QUECHUA (AYACUCHO)',
              intro='AmericasNLP 2021 (JW300): 125,008 pares originales. '
                    'Se limpiaron, filtraron por longitud y submuestrearon para entrenar en CPU.',
              nota='Mencionar que el quechua es aglutinante: una palabra combina varios morfemas.')
D.kpis(s, [('125,008', 'pares originales'), ('29,000', 'pares de entrenamiento'),
           ('758', 'de validación'), ('12,000', 'palabras de vocabulario (c/idioma)')])

s = D.content('PREPROCESAMIENTO',
              intro='Pipeline reproducible antes de entrenar cualquier modelo.',
              nota='Insistir en los tokens especiales <sos>/<eos> y el padding.')
D.flow(s, [
    ('1', 'NORMALIZAR', 'Minúsculas, separar puntuación, limpiar caracteres atípicos.'),
    ('2', 'FILTRAR', 'Oraciones de 1 a 15 palabras por idioma.'),
    ('3', 'TOKENIZAR', 'Vocabulario de 12,000 palabras + token <unk>.'),
    ('4', 'PADDING', 'Secuencias a longitud fija (17) con <sos>/<eos>.'),
])

s = D.content('ARQUITECTURA SEQ2SEQ (ENCODER-DECODER)',
              intro='Mismo esqueleto para ambos modelos: solo cambia el tipo de celda recurrente.',
              nota='Explicar que esto hace la comparacion justa: mismos datos, misma dimension, mismo optimizador.')
D.cards(s, [
    ('ENCODER', 'Embedding (128) → capa recurrente (SimpleRNN o LSTM, 256 unidades) → estado final.'),
    ('DECODER', 'Embedding (128) → recurrente inicializada con el estado del encoder, teacher forcing → Dense softmax (vocab).'),
], pie_items=['Adam · sparse_categorical_crossentropy', 'Inferencia: decodificación greedy paso a paso'])

s = D.content('RNN SIMPLE vs LSTM', intro='Ventajas y desventajas de cada celda recurrente.',
              nota='LSTM mitiga el desvanecimiento del gradiente gracias a sus puertas.')
D.two_col(s,
    izq=[('Parámetros', 'Menos (6.35M)'), ('Velocidad/época', 'Más rápida'),
         ('Memoria larga', 'Débil: sufre desvanecimiento del gradiente'), ('Convergencia', 'Se estancó temprano (early stopping, época 5)')],
    der=[('Parámetros', 'Más (6.94M)'), ('Velocidad/época', 'Más lenta por puertas adicionales'),
         ('Memoria larga', 'Fuerte: puertas de entrada/olvido/salida'), ('Convergencia', 'Menor loss de validación con menos épocas (3)')],
    titulos=('RNN SIMPLE', 'LSTM'))

s = D.content('RESULTADOS: RNN SIMPLE vs LSTM',
              intro='Mismo test interno (40 oraciones), nunca usado en entrenamiento.',
              nota='Aclarar que el entrenamiento se acoto por tiempo de entrega; con mas epocas ambos mejorarian.')
D.tabla(s, [4.2, 3.9, 3.9], [
    ['Métrica', 'RNN simple', 'LSTM'],
    ['Loss validación', '3.933', '3.704'],
    ['Accuracy/token (val)', '0.488', '0.490'],
    ['BLEU (test, n=40)', '0.0111', '0.0128'],
    ['Épocas completadas', '5', '3'],
])

s = D.content('EJEMPLO DE TRADUCCIÓN',
              intro='"para ilustrarlo: imagine que usted quiere edificar una casa"',
              nota='Mostrar que el LSTM genera estructura mas coherente aunque ambos modelos aun fallan con palabras poco frecuentes.')
D.tabla(s, [5.2, 11.6], [
    ['Fuente', 'Texto'],
    ['Referencia (quechua)', 'tupanachisun : wasi ruranaykipaq tanteawaq .'],
    ['RNN simple', '<unk> <unk> <unk> .'],
    ['LSTM', 'bibliam nin : <unk> <unk> <unk> .'],
])

s = D.content('CONCLUSIONES',
              intro='Hallazgos principales del proyecto.',
              nota='Cerrar insistiendo en que el pipeline completo es reproducible desde el repositorio.')
D.numbered(s, [
    ('Pipeline completo', 'Descarga, preprocesamiento, entrenamiento, evaluación y demo web, todo reproducible.'),
    ('LSTM superior', 'Menor loss y mejor BLEU que la RNN simple, con menos épocas de entrenamiento.'),
    ('Limitación principal', 'Entrenamiento acotado por tiempo y corpus reducido explican el BLEU bajo.'),
    ('Trabajo futuro', 'Más épocas, corpus completo (125k), atención, tokenización subpalabra (BPE).'),
], ncols=2)

D.closing(ref_titulo='CORPUS Y REFERENCIAS',
          ref_lineas=['AmericasNLP 2021 — github.com/AmericasNLP/americasnlp2021',
                      'Agić & Vulić (2019), JW300. Tiedemann (2012), OPUS.',
                      'Repo del proyecto: github.com/RedOne003/traductor-quechua-rnn'],
          bloque='INTEGRANTES', lineas=['Completar con los integrantes del equipo'])

ruta, _ = D.build(os.path.join(os.path.dirname(__file__), 'traductor_quechua_rnn.pptx'))
print('listo:', ruta)
