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
from portfolio_base.app.definitions import (
    apply_ocr_mask,
    zip_directory,
    export_to_makesense_image_first,
    import_from_makesense_image_first,
    recrop_from_yolo_labels,
    get_yolo_page_images,
)

st.markdown(
    """
    <style>
    h2 { padding-top: 0.8rem; }
    hr { margin: 2rem 0; }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================================================
# 🎯 Ziel & Pipeline-Überblick
# ======================================================

st.markdown("## 🎯 Ziel der Demo")

st.markdown(
    """
    Diese Demo zeigt eine **End-to-End Computer-Vision & OCR Pipeline**
    zur **automatischen Produkt-Erkennung in Supermarkt-Flyern**.

    **Use Case**
    - Unstrukturierte PDF-Flyer („Messy Real-World Data“)
    - Automatische Produkt-Boxen (YOLO)
    - Optional: Manuelle Korrektur (MakeSense)
    - Texterkennung (OCR) auf einzelnen Produkten
    """
)

st.markdown("### ⚠️ Limitationen (bewusst gewählt)")
st.markdown(
    """
    - Es wird **nur die erste Seite** des PDFs verarbeitet  
    - Das PDF ist **selbst erstellt / kuratiert** (kein Web-Scraping)
    - OCR ist **produktweise**, nicht seitenweise
    """
)

st.divider()

st.markdown("## 🧭 Pipeline-Struktur")

st.markdown(
    """
    **Input (PDF – 1 Seite)**  
    ⬇️  
    **YOLO: Produkt-Erkennung**  
    ⬇️  
    **(Optional) Manuelle Korrektur in MakeSense**  
    ⬇️  
    **Produkt-Crops**  
    ⬇️  
    **OCR auf ausgewähltem Produkt**
    """
)

st.caption(
    "➡️ Jeder Schritt ist unten als eigener Block umgesetzt. "
    "Die Pipeline kann **ohne manuelle Korrekturen** vollständig durchlaufen werden."
)

st.divider()


# ======================================================
# 1️⃣ PDF-Eingabe
# ======================================================

st.markdown("## 1️⃣ PDF-Eingabe")
st.caption(
    "Eingabe ist eine **einzelne PDF-Seite** mit realem Flyer-Layout "
    "(Rauschen, Preise, Bilder, unterschiedliche Schriftgrößen)."
)


demo_pdf_path = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "input"
    / "pdf_new"
    / "tegut_test.pdf"
)

use_demo = st.checkbox(
    "📂 Demo-PDF verwenden (reale, unaufbereitete Flyer-Daten)",
    value=True
)

uploaded_file = None
pdf_bytes = None
pdf_name = None

# --------------------------------------------------
# Option A: Demo-PDF
# --------------------------------------------------
if use_demo:
    if not demo_pdf_path.exists():
        st.error("❌ Demo-PDF nicht gefunden.")
        st.stop()

    pdf_bytes = demo_pdf_path.read_bytes()
    pdf_name = demo_pdf_path.name

# --------------------------------------------------
# Option B: Eigene Datei hochladen
# --------------------------------------------------
else:
    uploaded_file = st.file_uploader(
        "📄 Eigene PDF-Seite hochladen",
        type=["pdf"]
    )

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.read()
        pdf_name = uploaded_file.name

# --------------------------------------------------
# Abbruch falls nichts gewählt
# --------------------------------------------------
if pdf_bytes is None:
    st.info("⬆️ Wähle ein PDF (Demo oder Upload), um fortzufahren.")
    st.stop()

# --------------------------------------------------
# Einheitlicher Temp-Pfad (für YOLO / OCR)
# --------------------------------------------------
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(pdf_bytes)
    pdf_path = Path(tmp.name)

st.success(f"PDF geladen: {pdf_name}")

# --------------------------------------------------
# PDF-Vorschau (Recruiter!)
# --------------------------------------------------
st.markdown("### 👀 Original-PDF (reale Flyer-Daten)")

st.download_button(
    "⬇️ PDF herunterladen / im Browser öffnen",
    data=pdf_bytes,
    file_name=pdf_name,
    mime="application/pdf"
)

st.caption(
    "Das PDF kann im Browser geöffnet werden. "
    "Es zeigt **echte, unaufbereitete Flyer-Daten** "
    "mit unregelmäßigem Layout (Messy Real-World Data)."
)

st.divider()


# ======================================================
# 2️⃣ Produkt-Erkennung (YOLO)
# ======================================================

st.markdown("## 2️⃣ Produkt-Erkennung (YOLO)")
st.caption(
    "Ein vortrainiertes YOLO-Modell erkennt Produkt-Bounding-Boxes "
    "auf der PDF-Seite."
)

st.info(
    "Die Produkterkennung (YOLO) dauert je nach PDF ca. 20–40 Sekunden.",
    icon="🧠"
)


if st.button("🔍 Produkte erkennen"):
    run_dir, crop_infos = detect_products(pdf_path)
    st.session_state["RUN_DIR"] = run_dir
    st.session_state["crop_paths"] = crop_infos
    st.success(f"{len(crop_infos)} Produkte erkannt")

st.divider()

# ======================================================
# 3️⃣ Optional: Manuelle Korrektur (MakeSense)
# ======================================================

st.markdown("## 3️⃣ Optional: Manuelle Korrektur (MakeSense)")
st.caption(
    "Dieser Schritt ist **optional**. "
    "Er wird nur benötigt, wenn Bounding Boxes manuell korrigiert werden sollen."
)


use_makesense = st.selectbox(
    "Möchtest du die automatischen YOLO-Labels manuell anpassen?",
    [
        "❌ Nein – automatische Erkennung reicht aus",
        "✏️ Ja – Boxen manuell in MakeSense anpassen",
    ]
)

if use_makesense.startswith("✏️"):
    st.caption(
        """
        **MakeSense AI** ist ein webbasiertes Annotationstool für YOLO-Labels.  
        Nutze es **nur**, wenn du Bounding Boxes manuell korrigieren oder ergänzen willst.
        """
    )

    st.markdown(
        "🔗 **MakeSense öffnen:** "
        "[https://www.makesense.ai](https://www.makesense.ai)",
        unsafe_allow_html=True
    )

    action = st.selectbox(
        "MakeSense-Aktion",
        [
            "— bitte wählen —",
            "📤 Pre-Labeling: Bilder & Labels exportieren (für MakeSense)",
            "📥 Post-Labeling: Bearbeitete Labels re-uploaden",
        ]
    )

    # --------------------------------------------------
    # Pre-Labeling: Export zu MakeSense
    # --------------------------------------------------
    if action.startswith("📤") and "RUN_DIR" in st.session_state:
        st.info(
            "Exportiert erkannte Seiten + YOLO-Labels "
            "→ Unpack ZIP und upload in makesense manuell.",
            icon="⬆️"
        )

        if st.button("📦 MakeSense-Export erstellen"):
            out_dir = export_to_makesense_image_first(
                st.session_state["RUN_DIR"]
            )
            zip_path = zip_directory(out_dir)

            st.download_button(
                "⬇️ MakeSense ZIP herunterladen",
                open(zip_path, "rb"),
                file_name="makesense_export.zip"
            )

    # --------------------------------------------------
    # Post-Labeling: Re-Upload aus MakeSense
    # --------------------------------------------------
    if action.startswith("📥"):
        st.info(
            "Lade hier die **YOLO-Label-Dateien (*.txt)** hoch, "
            "die du in MakeSense bearbeitet hast.",
            icon="⬇️"
        )

        uploaded_labels = st.file_uploader(
            "YOLO-Labels aus MakeSense",
            type=["txt"],
            accept_multiple_files=True
        )

        if uploaded_labels and st.button("🔄 Labels re-importieren"):
            import_from_makesense_image_first(
                uploaded_labels,
                st.session_state["RUN_DIR"]
            )

            st.session_state["crop_paths"] = recrop_from_yolo_labels(
                st.session_state["RUN_DIR"]
            )

            st.success("✔ Labels importiert & Produkt-Crops aktualisiert")

st.divider()


# ======================================================
# 5️⃣ OCR auf Einzelprodukt
# ======================================================

st.markdown("## 5️⃣ OCR auf Einzelprodukt")
st.caption(
    "Texterkennung erfolgt **nur auf einem ausgewählten Produkt-Crop**, "
    "nicht auf der gesamten Seite."
)

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
# 5️⃣ OCR – Ein Crop auswählen & sofortige Vorschau
# ======================================================

if st.session_state.get("selected_crops"):
    st.markdown("## 🧠 OCR – Einzelnes Produkt auswählen")

    # --------------------------------------------------
    # Genau EIN Crop auswählen
    # --------------------------------------------------
    crop_map = {
        c["raw_path"].name: c
        for c in st.session_state["selected_crops"]
    }

    selected_name = st.selectbox(
        "Produkt für OCR auswählen",
        list(crop_map.keys())
    )

    active_crop = crop_map[selected_name]
    st.session_state["active_ocr_crop"] = active_crop

    # --------------------------------------------------
    # Original Crop sofort anzeigen
    # --------------------------------------------------
    st.markdown("### 👁️ Produkt-Crop (Original)")

    img_orig = cv2.imread(str(active_crop["raw_path"]))
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

    st.image(
        img_orig,
        caption="Original Crop – ungefiltert",
        width=300
    )

    # --------------------------------------------------
    # OCR-Sichtbereich einstellen (Live Preview)
    # --------------------------------------------------
    st.markdown("### 🧹 OCR-Sichtbereich (Live-Vorschau)")

    top = st.slider("Oben ausblenden (%)", 0, 50, 0) / 100
    bottom = st.slider("Unten ausblenden (%)", 0, 50, 0) / 100
    left = st.slider("Links ausblenden (%)", 0, 50, 0) / 100
    right = st.slider("Rechts ausblenden (%)", 0, 50, 0) / 100

    masked = apply_ocr_mask(
        img_orig,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
    )

    st.markdown("### 👁️ OCR sieht diesen Bereich")

    st.image(
        masked,
        caption="Live-OCR-Vorschau",
        width=300
    )

    # --------------------------------------------------
    # OCR explizit starten
    # --------------------------------------------------
    if st.button("🔤 OCR starten"):
        res = extract_text_easyocr(masked)

        st.markdown("### 📑 OCR-Ergebnis")
        st.text(res["text"])

