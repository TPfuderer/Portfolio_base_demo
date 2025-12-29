from pathlib import Path
from portfolio_base.tegut_ocr.yolo_detect import detect_products

# -------------------------------------------------
# Projekt-Root sauber bestimmen
# quick_test.py liegt in: src/portfolio_base/
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # -> src/
DATA_DIR = PROJECT_ROOT / "portfolio_base" / "data"

# -------------------------------------------------
# Test-PDF
# -------------------------------------------------
pdf = DATA_DIR / "input" / "pdf_new" / "tegut_test.pdf"

if not pdf.exists():
    raise FileNotFoundError(f"PDF nicht gefunden: {pdf}")

# -------------------------------------------------
# YOLO-Pipeline ausführen
# -------------------------------------------------
crops = detect_products(pdf)

print(f"{len(crops)} Produkte erkannt")
