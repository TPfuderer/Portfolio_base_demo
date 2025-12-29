from pathlib import Path
from portfolio_base.tegut_ocr.ocr_easy import extract_text_easyocr

# -------------------------------------------------
# Basis-Pfad (robust, egal wo gestartet)
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # -> src/

CROPS_DIR = (
    BASE_DIR
    / "portfolio_base"
    / "data"
    / "output"
    / "KW52"
    / "crops"
    / "ocr"
)

assert CROPS_DIR.exists(), f"Ordner nicht gefunden: {CROPS_DIR}"

# -------------------------------------------------
# 10 Bilder auswählen
# -------------------------------------------------
images = sorted(CROPS_DIR.glob("*.jpg"))[:10]

print(f"🔍 Teste {len(images)} Bilder aus:\n{CROPS_DIR}\n")

# -------------------------------------------------
# OCR-Durchlauf
# -------------------------------------------------
for i, img_path in enumerate(images, start=1):
    print("=" * 80)
    print(f"[{i}] {img_path.name}")

    res = extract_text_easyocr(
        img_path,
        min_confidence=0.3
    )

    print("Text:")
    print(res["text"])
    print()
    print(f"Mean confidence: {res['mean_confidence']:.3f}")


