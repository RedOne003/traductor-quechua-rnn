# -*- coding: utf-8 -*-
"""Genera informe/informe.pdf con los resultados reales del proyecto."""
import json
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "informe"
OUT.mkdir(exist_ok=True)

REPO_URL = "https://github.com/RedOne003/traductor-quechua-rnn"
DEMO_URL = "https://traductor-quechua-rnn.netlify.app"


def load(name):
    with open(RUNS / name, encoding="utf-8") as f:
        return json.load(f)


comp = load("comparison.json")
rnn_eval = load("rnn/eval.json")
lstm_eval = load("lstm/eval.json")
meta = json.load(open(ROOT / "processed" / "meta.json", encoding="utf-8"))


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, "Traductor Espanol-Quechua con RNN - Informe", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Pagina {self.page_no()}", align="C")


def h1(pdf, text):
    pdf.set_font("Helvetica", "B", 14)
    pdf.ln(4)
    pdf.multi_cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def h2(pdf, text):
    pdf.set_font("Helvetica", "B", 12)
    pdf.ln(2)
    pdf.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")


def p(pdf, text):
    pdf.set_font("Helvetica", "", 10.5)
    pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def mono(pdf, text):
    pdf.set_font("Courier", "", 9)
    pdf.multi_cell(0, 5, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

pdf.set_font("Helvetica", "B", 18)
pdf.multi_cell(0, 10, "Desarrollo de un Traductor Espanol-Quechua con RNN", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 11)
pdf.multi_cell(0, 7, "Curso: Inteligencia Artificial - Facultad de Ingenieria y Arquitectura\nPrograma: Ingenieria de Sistemas - Universidad Andina del Cusco\nRepositorio: " + REPO_URL + "\nDemo en linea: " + DEMO_URL, new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)

h1(pdf, "1. Objetivo")
p(pdf, "Disenar, implementar y comparar dos traductores automaticos de espanol a quechua "
       "basados en arquitecturas de Redes Neuronales Recurrentes (RNN): una RNN simple y "
       "una RNN con celdas LSTM, evaluando su desempeno sobre el mismo corpus y las mismas "
       "condiciones de entrenamiento.")

h1(pdf, "2. Recopilacion y preprocesamiento de datos")
p(pdf, f"Se utilizo el corpus paralelo publico espanol-quechua (variante Ayacucho, 'quy') del "
       f"shared task AmericasNLP 2021, derivado de JW300 (Agic & Vulic, 2019; Tiedemann, 2012), "
       f"disponible en github.com/AmericasNLP/americasnlp2021. El corpus original contiene "
       f"125,008 pares de oraciones alineadas.")
p(pdf, "Preprocesamiento aplicado:")
p(pdf, "- Normalizacion: minusculas, separacion de signos de puntuacion, limpieza de caracteres atipicos.\n"
       "- Filtrado por longitud: se conservaron oraciones de 1 a 15 palabras por idioma, para "
       "mantener secuencias manejables en una RNN basica.\n"
       "- Submuestreo a 30,000 pares (de los ~51,600 que superaron el filtro) para que el "
       "entrenamiento fuera viable en CPU dentro del tiempo disponible.\n"
       "- Tokenizacion a nivel de palabra (Keras Tokenizer), vocabulario limitado a 12,000 "
       "palabras por idioma, token <unk> para fuera de vocabulario.\n"
       "- Se agregaron tokens especiales <sos> y <eos> a cada oracion.\n"
       "- Padding post a longitud fija (17 tokens) con pad_sequences.")
p(pdf, f"Particion final: {meta['n_train']} pares de entrenamiento, {meta['n_dev']} de "
       f"validacion (dev oficial del shared task) y {meta['n_test']} de prueba interna "
       f"(no se usaron para entrenar ni ajustar hiperparametros).")

h1(pdf, "3. Arquitecturas implementadas")
p(pdf, "Se implemento un esquema encoder-decoder (sequence-to-sequence) identico para ambos "
       "modelos, variando unicamente el tipo de celda recurrente, para que la comparacion sea "
       "controlada:")
p(pdf, "- Encoder: capa Embedding (128 dimensiones, con mascara para el padding) seguida de una "
       "capa recurrente (SimpleRNN o LSTM, 256 unidades) que entrega el o los estados finales.\n"
       "- Decoder: Embedding (128) + capa recurrente del mismo tipo, inicializada con el estado "
       "del encoder, entrenada con teacher forcing, seguida de una capa densa softmax sobre el "
       "vocabulario quechua (12,000 clases).\n"
       "- Optimizador: Adam. Perdida: sparse_categorical_crossentropy. Metrica: accuracy por token.\n"
       "- Para la inferencia se construyeron modelos auxiliares de encoder y decoder que permiten "
       "decodificacion greedy (paso a paso), reutilizando las mismas capas entrenadas.")

h1(pdf, "4. Entrenamiento")
p(pdf, "Ambos modelos se entrenaron con el mismo conjunto de datos, mismo tamano de batch (64) "
       "y early stopping sobre la perdida de validacion (paciencia de 3 epocas), guardando el "
       "mejor checkpoint segun val_loss.")
p(pdf, "Nota sobre el alcance del entrenamiento: por la restriccion de tiempo de entrega de esta "
       "actividad, el entrenamiento se acoto antes de alcanzar convergencia completa (la RNN "
       "simple se detuvo por early stopping en la epoca 5; el LSTM se detuvo manualmente en la "
       "epoca 3, usando el mejor checkpoint disponible hasta ese punto). Esto se refleja en el "
       "desempeno (seccion 6) y se discute como limitacion en la seccion 7.")

r = comp["rnn"]
l = comp["lstm"]
h1(pdf, "5. Resultados cuantitativos")
pdf.set_font("Helvetica", "B", 10.5)
pdf.cell(55, 7, "Metrica", border=1)
pdf.cell(60, 7, "RNN simple", border=1)
pdf.cell(60, 7, "LSTM", border=1, new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 10.5)
rows = [
    ("Parametros entrenables", f"{r['n_params']:,}", f"{l['n_params']:,}"),
    ("Epocas completadas", str(r["epochs_ran"]), str(l["epochs_ran"])),
    ("Tiempo de entrenamiento", f"{r['train_seconds']:.0f} s", f"{l['train_seconds']:.0f} s"),
    ("Loss final (train)", f"{r['final_train_loss']:.3f}", f"{l['final_train_loss']:.3f}"),
    ("Loss final (val)", f"{r['final_val_loss']:.3f}", f"{l['final_val_loss']:.3f}"),
    ("Accuracy por token (val)", f"{r['final_val_acc']:.3f}", f"{l['final_val_acc']:.3f}"),
    (f"BLEU (test interno, n={rnn_eval['n_eval']})", f"{r['bleu']:.4f}", f"{l['bleu']:.4f}"),
]
for name, a, b in rows:
    pdf.cell(55, 7, name, border=1)
    pdf.cell(60, 7, a, border=1)
    pdf.cell(60, 7, b, border=1, new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)

h2(pdf, "Ejemplos de traduccion")
for i in range(3):
    ex_r = rnn_eval["ejemplos"][i]
    ex_l = lstm_eval["ejemplos"][i]
    p(pdf, f"ES:  {ex_r['es']}")
    mono(pdf, f"REF (quechua):  {ex_r['referencia']}\nRNN simple:     {ex_r['prediccion']}\nLSTM:           {ex_l['prediccion']}")

h1(pdf, "6. Analisis de resultados")
p(pdf, "El LSTM obtuvo una perdida de validacion menor (3.70 vs 3.93) y un BLEU ligeramente "
       "superior (0.0128 vs 0.0111) que la RNN simple, pese a haber completado menos epocas "
       "de entrenamiento (3 vs 5). Esto es consistente con la literatura: las celdas LSTM, al "
       "contar con una puerta de memoria (cell state) que mitiga el problema del desvanecimiento "
       "del gradiente, logran capturar dependencias de mas largo alcance con menos pasos de "
       "entrenamiento que una RNN simple.")
p(pdf, "Los valores de BLEU obtenidos son bajos en terminos absolutos. Esto se debe principalmente "
       "a: (1) el entrenamiento se detuvo muy temprano por la restriccion de tiempo disponible "
       "para esta entrega, antes de que los modelos convergieran; (2) el quechua es un idioma "
       "altamente aglutinante (una sola palabra puede combinar varios morfemas que en espanol "
       "se expresan como varias palabras), lo que dificulta la tokenizacion a nivel de palabra "
       "y aumenta la dispersion del vocabulario; y (3) el tamano del corpus usado (30,000 pares) "
       "es pequeno para un problema de traduccion automatica neuronal, que tipicamente requiere "
       "ordenes de magnitud mas de datos para generalizar bien.")

h1(pdf, "7. RNN simple vs LSTM: ventajas y desventajas")
p(pdf, "RNN simple:")
p(pdf, "+ Menos parametros y calculo mas liviano por paso de tiempo.\n"
       "+ Mas rapida de entrenar por epoca en secuencias cortas.\n"
       "- Sufre el problema del desvanecimiento/explosion del gradiente en secuencias largas, "
       "lo que limita su capacidad de recordar contexto lejano (relevante en traduccion, donde "
       "el orden de palabras entre espanol y quechua difiere).\n"
       "- Tiende a converger a una solucion pobre o estancarse en pocas epocas (se observo "
       "early stopping temprano).")
p(pdf, "LSTM:")
p(pdf, "+ Las puertas de entrada/olvido/salida permiten retener o descartar informacion de forma "
       "selectiva, mitigando el desvanecimiento del gradiente y mejorando el modelado de "
       "dependencias largas.\n"
       "+ En este experimento logro menor loss de validacion y mejor BLEU con menos epocas.\n"
       "- Mayor costo computacional por paso de tiempo (mas parametros y operaciones) y mayor "
       "tiempo de entrenamiento por epoca.\n"
       "- Requiere mas datos y epocas para explotar completamente su capacidad, por lo que con "
       "un entrenamiento muy breve la ventaja relativa frente a la RNN simple, aunque presente, "
       "es todavia modesta.")

h1(pdf, "8. Conclusiones")
p(pdf, "1. Se implemento exitosamente un pipeline completo (descarga de datos, preprocesamiento, "
       "entrenamiento, evaluacion e interfaz de demo) para traduccion automatica neuronal "
       "espanol-quechua, reutilizable y documentado en el repositorio de GitHub.\n"
       "2. El LSTM supero a la RNN simple en todas las metricas evaluadas (loss de validacion y "
       "BLEU), confirmando la ventaja teorica de las celdas con memoria de largo plazo para "
       "tareas de traduccion automatica.\n"
       "3. El desempeno absoluto de ambos modelos es limitado debido al entrenamiento acotado "
       "por tiempo y al tamano reducido del corpus; se recomienda como trabajo futuro entrenar "
       "por mas epocas, usar el corpus completo (125k pares), incorporar mecanismos de atencion "
       "y tokenizacion subpalabra (BPE) para lidiar con la morfologia aglutinante del quechua.\n"
       "4. Se desarrollo una interfaz web (Gradio) que permite comparar ambas traducciones en "
       "tiempo real, facilitando la demostracion del funcionamiento de ambos traductores.")

h1(pdf, "9. Repositorio y reproducibilidad")
p(pdf, f"Codigo fuente completo, datos de procesamiento, pesos de los modelos y resultados: {REPO_URL}")
p(pdf, f"Demo interactiva desplegada (traductor con vocabulario reducido, 100% en el navegador): {DEMO_URL}")
p(pdf, "El README del repositorio detalla los pasos exactos para reproducir el preprocesamiento, "
       "el entrenamiento de ambos modelos, la evaluacion BLEU y la demo web.")

h1(pdf, "10. Consideraciones eticas")
p(pdf, "El corpus utilizado es de acceso publico y se cita a sus autores originales (Agic & "
       "Vulic, 2019; Tiedemann, 2012) conforme a los terminos del shared task AmericasNLP. El "
       "quechua es una lengua originaria con comunidades de hablantes activas; este proyecto "
       "tiene fines estrictamente academicos y no pretende sustituir el trabajo de traductores "
       "ni linguistas nativos, sino explorar tecnicas de IA aplicadas a lenguas de bajos "
       "recursos, reconociendo las limitaciones del modelo resultante.")

pdf.output(str(OUT / "informe.pdf"))
print("Informe generado en", OUT / "informe.pdf")
