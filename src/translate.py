"""Demo de traduccion por linea de comandos.

Uso:
    py -3.10 src/translate.py lstm "Buenos dias, como estas"
    py -3.10 src/translate.py rnn  "Buenos dias, como estas"
"""
import sys

from inference import Translator


def main():
    if len(sys.argv) < 3:
        print("Uso: py -3.10 src/translate.py [rnn|lstm] \"oracion en espanol\"")
        sys.exit(1)
    cell_type = sys.argv[1]
    sentence = " ".join(sys.argv[2:])
    translator = Translator(cell_type)
    out = translator.translate(sentence)
    print(f"ES  : {sentence}")
    print(f"QUY : {out}")


if __name__ == "__main__":
    main()
