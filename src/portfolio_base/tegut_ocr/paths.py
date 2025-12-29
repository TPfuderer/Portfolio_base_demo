from pathlib import Path

# ======================================================
# 📦 Package root: src/portfolio_base
# ======================================================
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# ======================================================
# 📁 Data directory (TEMP runtime data)
# ======================================================
DATA_DIR = PACKAGE_ROOT / "data"

# ======================================================
# 🧠 YOLO model
# ======================================================
YOLO_MODEL = PACKAGE_ROOT / "models" / "tegut_yolo.pt"

# ======================================================
# 🧪 Safety checks (early fail is good)
# ======================================================
if not YOLO_MODEL.exists():
    raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL}")

DATA_DIR.mkdir(parents=True, exist_ok=True)

