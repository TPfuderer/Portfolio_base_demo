import sys
from pathlib import Path
from datetime import datetime
import tempfile
import re
import shutil
import zipfile

import numpy as np
import cv2
import streamlit as st
from PIL import Image

# -------------------------------------------------
# src/ zum Python-Pfad hinzufügen (GANZ AM ANFANG!)
# -------------------------------------------------
SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_ROOT))

# -------------------------------------------------
# Projekt-Imports
# -------------------------------------------------
from portfolio_base.tegut_ocr.yolo_detect import detect_products
from portfolio_base.tegut_ocr.ocr_easy import extract_text_easyocr

# ======================================================
# 🚀 Kurze Anleitung für schnelle Demos
# ======================================================
st.markdown(
    """
    ### 🚀 So testest du es
    - ⏱️ **30 Sek**: PDF hochladen
    - ⏱️ **20–40 Sek**: **Produkte erkennen** (YOLO)
    - ⏱️ **15 Sek**: Optional MakeSense-Export oder Label-Import
    - ⏱️ **10–20 Sek**: OCR auf ausgewählten Produkt-Crops
    """
)


# ======================================================
# 🔧 Hilfsfunktionen
# ======================================================

def apply_ocr_mask(img: np.ndarray, top=0, bottom=0, left=0, right=0) -> np.ndarray:
    """Weiße Maskierung – OCR sieht diese Bereiche nicht."""
    h, w = img.shape[:2]
    masked = img.copy()

    if top > 0:
        masked[:int(top*h), :] = 255
    if bottom > 0:
        masked[int((1-bottom)*h):, :] = 255
    if left > 0:
        masked[:, :int(left*w)] = 255
    if right > 0:
        masked[:, int((1-right)*w):] = 255

    return masked


def strip_confidence(lines: list[str]) -> list[str]:
    return [" ".join(l.split()[:5]) for l in lines if len(l.split()) >= 5]


def zip_directory(dir_path: Path) -> Path:
    zip_path = dir_path.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(dir_path), "zip", dir_path)
    return zip_path


def export_to_makesense_image_first(run_dir: Path) -> Path:
    pages_dir = run_dir / "pages"
    labels_dir = run_dir / "labels"

    out_dir = run_dir / "makesense_export"
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)

    for img in pages_dir.glob("*.png"):
        shutil.copy(img, images_out / img.name)

    for lbl in labels_dir.glob("*.txt"):
        clean = strip_confidence(lbl.read_text().splitlines())
        (labels_out / lbl.name).write_text("\n".join(clean), encoding="utf-8")

    (labels_out / "labels.txt").write_text("product\n", encoding="utf-8")

    return out_dir


def import_from_makesense_image_first(uploaded_files, run_dir: Path):
    labels_dir = run_dir / "labels"
    for p in labels_dir.glob("*.txt"):
        p.unlink()
    for f in uploaded_files:
        (labels_dir / f.name).write_bytes(f.read())


def recrop_from_yolo_labels(run_dir: Path) -> list[dict]:
    pages_dir = run_dir / "pages"
    labels_dir = run_dir / "labels"
    crops_dir = run_dir / "crops"

    if crops_dir.exists():
        shutil.rmtree(crops_dir)

    raw_dir = crops_dir / "raw"
    ocr_dir = crops_dir / "ocr"
    raw_dir.mkdir(parents=True)
    ocr_dir.mkdir(parents=True)

    crop_infos = []

    for label_file in sorted(labels_dir.glob("*.txt")):
        img_path = pages_dir / f"{label_file.stem}.png"
        if not img_path.exists():
            continue

        img = Image.open(img_path)
        W, H = img.size
        img_np = np.array(img)

        for i, line in enumerate(label_file.read_text().splitlines()):
            cls, xc, yc, w, h = map(float, line.split())
            x1 = int((xc - w/2) * W)
            y1 = int((yc - h/2) * H)
            x2 = int((xc + w/2) * W)
            y2 = int((yc + h/2) * H)

            crop = img_np[y1:y2, x1:x2]
            base = f"{label_file.stem}_box{i+1:03d}_cls{int(cls)}"

            raw_path = raw_dir / f"{base}.jpg"
            ocr_path = ocr_dir / f"{base}.jpg"

            Image.fromarray(crop).save(raw_path)
            Image.fromarray(crop).save(ocr_path)

            crop_infos.append({
                "raw_path": raw_path,
                "ocr_path": ocr_path
            })

    return crop_infos


def get_yolo_page_images(run_dir: Path) -> list[Path]:
    yolo_dir = run_dir / "yolo" / "detect"
    pages = {}

    for img in yolo_dir.glob("*.jpg"):
        m = re.search(r"_page_(\d+)", img.name)
        if m:
            idx = int(m.group(1))
            pages.setdefault(idx, img)

    return [pages[k] for k in sorted(pages)]


# ======================================================
# 1️⃣ PDF Upload
# ======================================================

uploaded_file = st.file_uploader("📄 PDF-Seite hochladen", type=["pdf"])
if uploaded_file is None:
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.read())
    pdf_path = Path(tmp.name)

st.success(f"PDF geladen: {uploaded_file.name}")



# ======================================================
# 2️⃣ Object Detection
# ======================================================

st.info(
    "Die Produkterkennung (YOLO) dauert je nach PDF ca. 20–40 Sekunden.",
    icon="🧠"
)


if st.button("🔍 Produkte erkennen"):
    run_dir, crop_infos = detect_products(pdf_path)
    st.session_state["RUN_DIR"] = run_dir
    st.session_state["crop_paths"] = crop_infos
    st.success(f"{len(crop_infos)} Produkte erkannt")


# ======================================================
# 3️⃣ MakeSense Export / Import
# ======================================================

if "RUN_DIR" in st.session_state:
    if st.button("⬇️ Für MakeSense vorbereiten"):
        out_dir = export_to_makesense_image_first(st.session_state["RUN_DIR"])
        zip_path = zip_directory(out_dir)
        st.download_button("📦 MakeSense ZIP", open(zip_path, "rb"), file_name="makesense_export.zip")

st.markdown("## ⬆️ MakeSense-Labels importieren")

uploaded_labels = st.file_uploader(
    "YOLO-Labels (*.txt)",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_labels and st.button("📥 Labels importieren"):
    import_from_makesense_image_first(uploaded_labels, st.session_state["RUN_DIR"])
    st.session_state["crop_paths"] = recrop_from_yolo_labels(st.session_state["RUN_DIR"])
    st.success("✔ Labels importiert & Crops neu erzeugt")


# ======================================================
# 4️⃣ Crops auswählen
# ======================================================

if "crop_paths" in st.session_state:
    st.markdown("## 🧩 Produkt-Crops auswählen")

    selected = []
    cols = st.columns(4)

    for i, crop in enumerate(st.session_state["crop_paths"]):
        with cols[i % 4]:
            st.image(Image.open(crop["raw_path"]), use_container_width=True)
            if st.checkbox("Für OCR", key=f"sel_{i}"):
                selected.append(crop)

    st.session_state["selected_crops"] = selected


# ======================================================
# 5️⃣ OCR-Vorbereitung + OCR (EIN Crop, Apply-Workflow)
# ======================================================

if st.session_state.get("selected_crops"):
    st.markdown("## 🧠 OCR-Crop auswählen")

    crop_names = [c["raw_path"].name for c in st.session_state["selected_crops"]]
    selected_name = st.selectbox("Aktives OCR-Bild", crop_names)

    active_crop = next(
        c for c in st.session_state["selected_crops"]
        if c["raw_path"].name == selected_name
    )

    st.session_state["active_ocr_crop"] = active_crop

    # -------------------------------
    # OCR-Sichtbereich einstellen
    # -------------------------------
    st.markdown("### 🧹 OCR-Sichtbereich einstellen")

    top = st.slider("Oben ausblenden (%)", 0, 50, 0) / 100
    bottom = st.slider("Unten ausblenden (%)", 0, 50, 0) / 100
    left = st.slider("Links ausblenden (%)", 0, 50, 0) / 100
    right = st.slider("Rechts ausblenden (%)", 0, 50, 0) / 100

    # -------------------------------
    # Anwenden-Button (JETZT erst Bild ändern)
    # -------------------------------
    if st.button("🧹 OCR-Sichtbereich anwenden"):
        img = cv2.imread(str(active_crop["ocr_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masked = apply_ocr_mask(img, top, bottom, left, right)
        st.session_state["ocr_preview_image"] = masked

    # -------------------------------
    # Vorschau anzeigen
    # -------------------------------
    if "ocr_preview_image" in st.session_state:
        st.markdown("### 👁️ OCR-Vorschau")

        st.image(
            st.session_state["ocr_preview_image"],
            caption="OCR sieht genau diesen Bereich",
            width=350
        )

        # -------------------------------
        # OCR ausführen
        # -------------------------------

        st.info(
            "EasyOCR erkennt nun Text auf den ausgewählten Produktbildern (ca. 10–20 Sekunden).",
            icon="🔤"
        )

        if st.button("🔤 OCR starten"):
            res = extract_text_easyocr(st.session_state["ocr_preview_image"])

            st.markdown("### 📑 OCR-Ergebnis")
            st.text(res["text"])
