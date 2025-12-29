# ===============================
# 🔄 Convert YOLO (x1,y1,x2,y2,conf) → MakeSense (xc,yc,w,h)
# ===============================

from pathlib import Path
import cv2

# === 🧩 MANUELL KW EINSTELLEN ===
KW = 46  # <<<<<<<<<<<<<<<<<<<<<<<<< HIER ändern (z.B. 47, 48, 49 ...)

# === 📂 Basisordner ===
BASE_ROOT = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\TEGUT OCR")
base_dir = BASE_ROOT / f"KW{KW:02d}"

labels_dir = base_dir / "labels"
pages_dir = base_dir / "pages"
output_dir = base_dir / "labels_makesense"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"📁 KW-Ordner: {base_dir}")
print(f"📄 Labels aus: {labels_dir}")
print(f"🖼 Pages aus:   {pages_dir}")
print("────────────────────────────────────────")

# === 🚀 Konvertierung ===
count_files = 0

for label_file in labels_dir.glob("*.txt"):
    page_name = label_file.stem  # z.B. page_01
    img_path = pages_dir / f"{page_name}.png"

    if not img_path.exists():
        print(f"⚠️ Kein Bild für {label_file.name} → überspringe.")
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Bild kann nicht geladen werden: {img_path}")
        continue

    img_h, img_w = img.shape[:2]

    lines = label_file.read_text(encoding="utf-8").splitlines()
    new_lines = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            print(f"⚠️ Ungültige Zeile in {label_file.name}: {line}")
            continue

        cls, conf, x1, y1, x2, y2 = map(float, parts)

        # === Umrechnung (absolute → relative) ===
        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h

        new_lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    (output_dir / label_file.name).write_text("\n".join(new_lines), encoding="utf-8")
    count_files += 1

print("────────────────────────────────────────")
print(f"✅ {count_files} Dateien erfolgreich konvertiert.")
print(f"📂 MakeSense-kompatible Labels → {output_dir}")

# === optional: Klassen-Datei erzeugen
labels_txt = output_dir / "labels.txt"
if not labels_txt.exists():
    labels_txt.write_text("product\n", encoding="utf-8")
    print("🏷️ labels.txt erzeugt.")
