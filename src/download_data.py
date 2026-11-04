"""Descarga el corpus paralelo espanol-quechua (AmericasNLP 2021, basado en JW300)."""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/AmericasNLP/americasnlp2021/main/data/quechua-spanish"
FILES = ["train.es", "train.quy", "dev.es", "dev.quy", "test.es", "dict.es", "dict.quy", "README.md"]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def main():
    for name in FILES:
        dest = DATA_DIR / name
        url = f"{BASE}/{name}"
        print(f"Descargando {name} ...")
        urllib.request.urlretrieve(url, dest)
    print("Listo. Archivos en", DATA_DIR)


if __name__ == "__main__":
    main()
