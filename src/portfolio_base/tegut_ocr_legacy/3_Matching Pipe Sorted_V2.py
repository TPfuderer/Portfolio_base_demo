# ===============================
# 🧩 Produkt-Text-Matching mit eindeutiger Zuordnung (TEGUT OCR Version)
# ===============================

import fitz
import json
import cv2
import pandas as pd
from pathlib import Path

# === 📅 Kalenderwoche definieren ===
KW = 46   # << HIER ÄNDERN

# === 📂 Basisordner der TEGUT OCR Pipeline ===
BASE = Path(r"C:\Users\pfudi\PycharmProjects\PythonProject\outputs\TEGUT OCR")

# === Eingaben ===
pdf_dir    = BASE / "PDFs" / "New"
pages_dir  = BASE / f"KW{KW:02d}" / "pages"
label_dir  = BASE / f"KW{KW:02d}" / "CROP_ready"

# === PDF automatisch auswählen ===
pdf_files = list(pdf_dir.glob("*.pdf"))
if not pdf_files:
    raise FileNotFoundError(f"❌ Kein PDF gefunden in: {pdf_dir}")
pdf_path = pdf_files[0]   # (immer nur 1 Datei im Ordner)

# === Output ===
out_dir  = BASE / f"KW{KW:02d}" / "matched"
out_vis  = out_dir / "visual"

out_dir.mkdir(parents=True, exist_ok=True)
out_vis.mkdir(parents=True, exist_ok=True)

print(f"📄 PDF:       {pdf_path}")
print(f"🖼️ Pages:     {pages_dir}")
print(f"🔵 Labels:    {label_dir}")
print(f"💾 Output:    {out_dir}\n")


# ===============================
# Hilfsfunktionen
# ===============================

def load_yolo_labels(path):
    boxes = []
    try:
        lines = path.read_text().splitlines()
    except Exception as e:
        print(f"❌ Fehler beim Lesen {path.name}: {e}")
        return boxes

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 6:
            continue
        cls, conf, x1, y1, x2, y2 = parts
        boxes.append(dict(
            cls=int(float(cls)),
            conf=float(conf),
            x1=float(x1), y1=float(y1),
            x2=float(x2), y2=float(y2)
        ))
    return boxes


def rect_intersection(a, b):
    """Berechne den Anteil der Textbox, der innerhalb der Produktbox liegt."""
    x_left = max(a["x1"], b["x1"])
    y_top = max(a["y1"], b["y1"])
    x_right = min(a["x2"], b["x2"])
    y_bottom = min(a["y2"], b["y2"])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    text_area  = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter_area / text_area


# ===============================
# PDF öffnen
# ===============================

doc = fitz.open(pdf_path)
print(f"📘 PDF geladen ({len(doc)} Seiten)\n")


# ===============================
# Verarbeitung jeder Seite
# ===============================

page_files = sorted(pages_dir.glob("page_*.png"))
all_matches = []


for page_img_path in page_files:
    page_no = int(page_img_path.stem.split("_")[-1])
    print(f"🟩 === Seite {page_no} ===")

    # --- Bild laden ---
    img = cv2.imread(str(page_img_path))
    if img is None:
        print(f"❌ Bild konnte nicht geladen werden: {page_img_path}")
        continue
    new_h, new_w = img.shape[:2]

    # --- Textlayer ---
    page = doc[page_no - 1]
    text_blocks = page.get_text("blocks")
    page_w_pt, page_h_pt = page.rect.width, page.rect.height
    mat = fitz.Matrix(new_w / page_w_pt, new_h / page_h_pt)

    text_boxes = []
    for b in text_blocks:
        try:
            x0, y0, x1, y1, text = b[:5]
        except:
            continue
        if not str(text).strip():
            continue

        rect = fitz.Rect(x0, y0, x1, y1) * mat
        text_boxes.append({
            "x1": rect.x0, "y1": rect.y0,
            "x2": rect.x1, "y2": rect.y1,
            "text": text.strip()
        })

    print(f"   📝 {len(text_boxes)} Textboxen geladen")

    # --- YOLO-Labels ---
    yolo_path = label_dir / f"page_{page_no:02d}.txt"
    if not yolo_path.exists():
        print(f"⚠️ Keine Labels für Seite {page_no}")
        continue

    yolo_boxes = load_yolo_labels(yolo_path)
    print(f"   📦 {len(yolo_boxes)} YOLO-Boxen geladen")

    # --- Matching ---
    matches = []
    assigned_texts = set()

    for i, yb in enumerate(yolo_boxes):
        y_rect = yb.copy()
        matched_texts = []

        for j, tb in enumerate(text_boxes):
            overlap = rect_intersection(y_rect, tb)
            if overlap >= 0.95:
                matched_texts.append((j, overlap))

        matched_texts.sort(key=lambda x: x[1], reverse=True)

        for j, overlap in matched_texts:
            if j not in assigned_texts:
                assigned_texts.add(j)
                matches.append({
                    "page": page_no,
                    "product_id": i + 1,
                    "conf": y_rect["conf"],
                    "yolo_box": [y_rect["x1"], y_rect["y1"], y_rect["x2"], y_rect["y2"]],
                    "text": text_boxes[j]["text"],
                    "text_box": [
                        text_boxes[j]["x1"], text_boxes[j]["y1"],
                        text_boxes[j]["x2"], text_boxes[j]["y2"]
                    ],
                    "overlap": round(overlap, 3)
                })

        if not matched_texts:
            matches.append({
                "page": page_no,
                "product_id": i + 1,
                "conf": y_rect["conf"],
                "yolo_box": [y_rect["x1"], y_rect["y1"], y_rect["x2"], y_rect["y2"]],
                "text": None,
                "text_box": None,
                "overlap": 0.0
            })

    all_matches.extend(matches)
    print(f"   🔗 {len(matches)} Matches gefunden")

    # --- Debug Visualisierung ---
    img_vis = img.copy()
    for m in matches:
        x1, y1, x2, y2 = map(int, m["yolo_box"])
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 3)

        if m["text_box"]:
            tx1, ty1, tx2, ty2 = map(int, m["text_box"])
            cv2.rectangle(img_vis, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
            cv2.putText(img_vis, str(m["overlap"]), (tx1, ty1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out_img = out_vis / f"page_{page_no:02d}_matched.jpg"
    cv2.imwrite(str(out_img), img_vis)
    print(f"   🖼️ Debug gespeichert → {out_img.name}\n")


# ===============================
# CSV + JSON Speichern
# ===============================

csv_path  = out_dir / f"KW{KW}_matches.csv"
json_path = out_dir / f"KW{KW}_matches.json"

pd.DataFrame(all_matches).to_csv(csv_path, index=False, encoding="utf-8-sig")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(all_matches, f, indent=2, ensure_ascii=False)

print(f"💾 CSV gespeichert:  {csv_path}")
print(f"💾 JSON gespeichert: {json_path}")
print("🎉 Schritt 1 abgeschlossen – Produkt-Text-Matching!\n")


# ===============================
# Gruppierung nach Produkt
# ===============================

group_csv  = out_dir / f"KW{KW}_matches_grouped.csv"
group_json = out_dir / f"KW{KW}_matches_grouped.json"

df = pd.read_csv(csv_path)

grouped = (
    df.groupby(["page", "product_id"], as_index=False)
    .agg({
        "conf": "first",
        "yolo_box": "first",
        "text": lambda x: "\n".join([t for t in x.dropna().astype(str) if t.strip()]),
        "overlap": "max"
    })
)

grouped.rename(columns={"text": "combined_text"}, inplace=True)
grouped.to_csv(group_csv, index=False, encoding="utf-8-sig")
grouped.to_json(group_json, orient="records", indent=2, force_ascii=False)

print(f"📁 Gruppiertes CSV:  {group_csv}")
print(f"📁 Gruppiertes JSON: {group_json}")
print("🎉 Schritt 2 abgeschlossen – Gruppierung der Matches!")
