"""Interfaz web (Gradio) del traductor espanol -> quechua.

Permite elegir el modelo (RNN simple o LSTM) y comparar ambas traducciones
lado a lado para la misma oracion. Pensada para la demo de la exposicion.

Uso:
    py -3.10 src/app.py
"""
import gradio as gr

from inference import Translator

print("Cargando modelos...")
translators = {}
for cell_type in ("rnn", "lstm"):
    try:
        translators[cell_type] = Translator(cell_type)
        print(f"  {cell_type} cargado.")
    except Exception as e:
        print(f"  no se pudo cargar {cell_type}: {e}")


def traducir(oracion):
    if not oracion or not oracion.strip():
        return "", ""
    out_rnn = translators["rnn"].translate(oracion) if "rnn" in translators else "(modelo no disponible)"
    out_lstm = translators["lstm"].translate(oracion) if "lstm" in translators else "(modelo no disponible)"
    return out_rnn, out_lstm


EJEMPLOS = [
    "Buenos días, cómo estás",
    "Mi casa es grande",
    "Dios te ama",
    "Vamos a la escuela",
    "Gracias por tu ayuda",
]

with gr.Blocks(title="Traductor Español-Quechua (RNN vs LSTM)") as demo:
    gr.Markdown(
        "# Traductor Español → Quechua\n"
        "Comparación de un traductor neuronal con **RNN simple** y con **LSTM** "
        "(mismo corpus, misma arquitectura base, solo cambia la celda recurrente).\n\n"
        "Proyecto del curso de Inteligencia Artificial — UAC, Ingeniería de Sistemas."
    )
    with gr.Row():
        entrada = gr.Textbox(label="Oración en español", placeholder="Escribe una oración corta en español...")
    boton = gr.Button("Traducir", variant="primary")
    with gr.Row():
        salida_rnn = gr.Textbox(label="Traducción — RNN simple")
        salida_lstm = gr.Textbox(label="Traducción — LSTM")

    gr.Examples(examples=EJEMPLOS, inputs=entrada)

    boton.click(fn=traducir, inputs=entrada, outputs=[salida_rnn, salida_lstm])
    entrada.submit(fn=traducir, inputs=entrada, outputs=[salida_rnn, salida_lstm])

if __name__ == "__main__":
    demo.launch()
