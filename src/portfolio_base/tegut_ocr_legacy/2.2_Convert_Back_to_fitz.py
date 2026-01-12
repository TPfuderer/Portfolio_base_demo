# ===============================
# 🔄 MakeSense YOLO → Absolute Koordinaten → CROP_ready
# ===============================

from pathlib import Path
import cv2

# === 🧩 MANUELLE KW ===
KW = 46   # << hier ändern

# === 📂 Ordnerstruktur ===
BASE_ROOT = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\TEGUT OCR")
base_dir = BASE_ROOT / f"KW{KW:02d}"

labels_rel_dir = base_dir / "labels_edited"   # Eingabe: MakeSense-relative YOLO Labels
pages_dir      = base_dir / "pages"           # PNG-Seiten
labels_out_dir = base_dir / "CROP_ready"      # Ausgabe: absolute Koordinaten
labels_out_dir.mkdir(exist_ok=True)

print("📄 Relative (MakeSense) Labels:", labels_rel_dir)
print("🖼 Pages:", pages_dir)
print("📁 Output (CROP_ready):", labels_out_dir)
print("────────────────────────────────────────")

default_conf = 0.999
converted = 0
skipped = 0

# === 🔍 Schleife über alle Labels ===
for label_file in labels_rel_dir.glob("*.txt"):

    page_name = label_file.stem   # page_01
    img_path = pages_dir / f"{page_name}.png"

    if not img_path.exists():
        print(f"⚠️ Kein PNG für {label_file.name}, übersprungen.")
        skipped += 1
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Fehler beim Laden von: {img_path}")
        continue

    h, w = img.shape[:2]

    rel_lines = label_file.read_text(encoding="utf-8").splitlines()
    abs_lines = []

    for line in rel_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"⚠️ Ungültige Zeile in {label_file.name}: {line}")
            continue

        cls, xc, yc, ww, hh = map(float, parts)

        # → relative YOLO → absolute Pixel (fitz-/cv2-kompatibel)
        x1 = int((xc - ww/2) * w)
        y1 = int((yc - hh/2) * h)
        x2 = int((xc + ww/2) * w)
        y2 = int((yc + hh/2) * h)

        abs_lines.append(f"{int(cls)} {default_conf:.4f} {x1} {y1} {x2} {y2}")

    # Speicherung
    out_path = labels_out_dir / label_file.name
    out_path.write_text("\n".join(abs_lines), encoding="utf-8")

    print(f"✅ Konvertiert: {label_file.name} ({len(abs_lines)} Boxen)")
    converted += 1

print("────────────────────────────────────────")
print(f"🎯 Erfolgreich konvertiert: {converted}")
print(f"⚠️ Übersprungen: {skipped}")
print(f"📂 Fertig! Deine CROP_ready Labels liegen in:\n{labels_out_dir}")
