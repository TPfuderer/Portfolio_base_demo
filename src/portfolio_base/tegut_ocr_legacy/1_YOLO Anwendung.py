# ===============================
# 🚀 Flyer Detection Pipeline V2.2 (Pfad-Update)
# ===============================

from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import fitz
import cv2
import numpy as np
import os
import shutil
from datetime import datetime

# ============================================
# 📂 1) Pfade anpassen (wie du wolltest)
# ============================================

PDF_NEW_DIR = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\TEGUT OCR\PDFs\New")
PDF_ARCHIVE_DIR = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\TEGUT OCR\PDFs\Archive")
MODEL_PATH = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\TEGUT OCR\Model\best(2).pt")
BASE_ROOT = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\TEGUT OCR")

# === Nimm automatisch die *erste PDF* im New-Ordner ===
pdf_files = sorted(PDF_NEW_DIR.glob("*.pdf"))
if not pdf_files:
    raise FileNotFoundError("❌ Keine PDF im 'New' Ordner gefunden!")

pdf_path = pdf_files[0]
print(f"📄 Verarbeite PDF: {pdf_path.name}")

# === KW AUTOMATISCH bestimmen ===
kw = datetime.now().isocalendar()[1]
base_dir = BASE_ROOT / f"KW{kw:02d}"

# === Unterordner wie gewohnt ===
pages_dir       = base_dir / "pages"
detections_dir  = base_dir / "detections_full"
labels_dir      = base_dir / "labels"
crops_dir       = base_dir / "boxes_crops"
filtered_dir    = base_dir / "filtered_boxes"

for d in [pages_dir, detections_dir, labels_dir, crops_dir, filtered_dir]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================
# 📌 2) PDF → PNG konvertieren
# ============================================

print("📄 Konvertiere PDF-Seiten zu PNGs ...")
doc = fitz.open(pdf_path)
page_images = []

for i, page in enumerate(doc, start=1):
    pix = page.get_pixmap(dpi=450)
    page_path = pages_dir / f"page_{i:02d}.png"
    pix.save(page_path)
    page_images.append(page_path)
    print(f"✅ Seite {i} gespeichert: {page_path}")

doc.close()
print("📕 PDF geschlossen.")
# ============================================
# 🧠 3) YOLO laden & ausführen
# ============================================

print("🧠 Lade Modell ...")
model = YOLO(MODEL_PATH)

print("🔍 Erkenne Produkte ...")
results = model.predict(
    source=[str(p) for p in page_images],
    save=True,
    save_txt=True,
    save_conf=True,
    project=str(detections_dir),
    name=f"KW{kw:02d}_detect",
    exist_ok=True
)

# ============================================
# 📦 4) Crops & Linienentfernung
# ============================================

print("📦 Schneide erkannte Produktboxen aus ...")

for result in results:
    img_path = Path(result.path)
    im = Image.open(img_path)
    img_np = np.array(im)

    for j, box in enumerate(result.boxes):
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls  = int(box.cls[0])

        crop = img_np[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=40, maxLineGap=5)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(crop, (x1, y1), (x2, y2), (255,255,255), 2)

        crop_pil = Image.fromarray(crop)
        crop_name = f"{img_path.stem}_box{j+1:03d}_cls{cls}_conf{conf:.2f}.jpg"
        crop_pil.save(crops_dir / crop_name)

print("✅ Crops gespeichert:", crops_dir)

# ============================================
# 🏷 5) Labels speichern
# ============================================

print("🏷️ Speichere Labels ...")
for result in results:
    stem = Path(result.path).stem
    label_file = labels_dir / f"{stem}.txt"
    result.save_txt(label_file, save_conf=True)

print("✅ Labels gespeichert in:", labels_dir)

# ============================================
# 🧹 6) IoU Filter
# ============================================

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

print("🧹 Entferne überlappende Boxen ...")

for result in results:
    img = cv2.imread(str(result.path))
    boxes = [b.xyxy[0].cpu().numpy().astype(int) for b in result.boxes]

    keep = []
    for i, boxA in enumerate(boxes):
        if any(iou(boxA, boxB) > 0.9 for j, boxB in enumerate(boxes) if i != j):
            continue
        keep.append(i)

    for idx in keep:
        x1, y1, x2, y2 = boxes[idx]
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

    out_path = filtered_dir / Path(result.path).name
    cv2.imwrite(str(out_path), img)

print("✅ Gefilterte Boxen gespeichert:", filtered_dir)

# ============================================
# 📦 7) PDF → ARCHIVE verschieben
# ============================================

dest = PDF_ARCHIVE_DIR / pdf_path.name
shutil.move(str(pdf_path), str(dest))
print(f"📦 PDF archiviert nach: {dest}")

print("\n🎉 Pipeline abgeschlossen! Ergebnisse in:", base_dir)
