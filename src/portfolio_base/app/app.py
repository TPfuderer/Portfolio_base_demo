import sys
from pathlib import Path
import tempfile

import cv2
import streamlit as st
from PIL import Image
import json


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
st.markdown(
    """
    <style>
    .pipeline-box {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px auto;
        width: 80%;
        text-align: center;
        font-size: 1.05rem;
        font-weight: 500;
    }
    .pipeline-arrow {
        text-align: center;
        font-size: 1.8rem;
        margin: 6px 0;
        color: #9CA3AF;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* ================================
       🔘 Große Action-Buttons
       ================================ */
    div.stButton > button {
        width: 100%;
        padding: 0.9rem 1.2rem;
        font-size: 1.15rem;
        font-weight: 600;
        border-radius: 12px;
    }

    /* Extra Betonung für Hauptaktionen */
    div.stButton > button:has(span:contains("Produkte erkennen")),
    div.stButton > button:has(span:contains("OCR starten")) {
        font-size: 1.25rem;
        padding: 1.1rem 1.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ======================================================
# 🎯 Ziel & Pipeline-Überblick
# ======================================================

st.markdown(
    "## Demo einer Pipeline zur optical character recognition (OCR) "
    "für Produkterkennung und Texterfassung in Tegut-Supermarktflyern"
)

st.markdown("## 🧭 Pipeline-Struktur")

st.markdown('<div class="pipeline-box">1️⃣ Input (PDF – eine TEGUT Seite)</div>', unsafe_allow_html=True)
st.markdown('<div class="pipeline-arrow">⬇️</div>', unsafe_allow_html=True)

st.markdown('<div class="pipeline-box">2️⃣ YOLO-basierte Produkterkennung</div>', unsafe_allow_html=True)
st.markdown('<div class="pipeline-arrow">⬇️</div>', unsafe_allow_html=True)

st.markdown('<div class="pipeline-box">3️⃣ (Optional) Manuelle Korrektur mit MakeSense</div>', unsafe_allow_html=True)
st.markdown('<div class="pipeline-arrow">⬇️</div>', unsafe_allow_html=True)

st.markdown('<div class="pipeline-box">4️⃣ Produkt-Crops</div>', unsafe_allow_html=True)
st.markdown('<div class="pipeline-arrow">⬇️</div>', unsafe_allow_html=True)

st.markdown('<div class="pipeline-box">5️⃣ Optical character recognition (OCR) auf ausgewähltem Produkt</div>', unsafe_allow_html=True)

st.caption(
    "Jeder Schritt ist unten als eigener Block umgesetzt. "
    "Die Pipeline kann vollständig ohne manuelle Korrekturen durchlaufen werden."
)


st.divider()

st.markdown("## 🎯 Ziel der Demo")

st.markdown(
    """
    Diese Demo zeigt eine Computer-Vision- und optical-character-recognition-(OCR)-Pipeline
    zur automatischen Produkterkennung in Supermarkt-Flyern sowie zur Texterfassung
    auf Produktebene.
    """
)

st.markdown("### ⚠️ Aktuelle Limitationen")

st.markdown(
    """
    - Es wird **nur die erste Seite** des PDFs verarbeitet  
    - Das PDF ist **manuell bereitgestellt / kuratiert** (kein Web-Scraping)
    - Optical character recognition (OCR) erfolgt **produktweise**, nicht seitenweise
    - Es wird ein **kleines YOLOv8-Modell** aufgrund von Serverbeschränkungen verwendet
    - Das Modell ist **ausschließlich auf Tegut-Flyern trainiert** und kann bei stark
      abweichenden Layouts (z. B. saisonalen oder Weihnachtsflyern) versagen
    - Erweiterungen wie OCR-Fehlerkorrektur, automatische Preiserkennung oder
      zusätzliche Crop-Refinements sind **noch nicht implementiert**
    """
)

st.divider()


# ======================================================
# 1️⃣ PDF-Eingabe
# ======================================================

st.markdown("## 1️⃣ PDF-Eingabe")
st.caption(
    "Eingabe ist eine **einzelne PDF-Seite** mit realem Supermarkt-Flyer-Layout "
    "(Rauschen, Preise, Bilder, unterschiedliche Schriftgrößen)."
)

st.write(
    "Optional kannst du ein **aktuelles Tegut-Flyer-PDF** verwenden. "
    "Bei **starken Layout-Änderungen** (z. B. saisonale Sonderausgaben) "
    "kann die Erkennungsqualität eingeschränkt sein."
)

st.markdown(
    "🔗 **Beispiel (öffentliches PDF):**  \n"
    "[Tegut Flyer KW20 2025 – Thüringen]"
    "(https://static.tegut.com/fileadmin/tegut_upload/Dokumente/"
    "Aktuelle_Flugbl%C3%A4tter/tegut_FB-KW20-2025_thueringen.pdf)"
)

demo_pdf_path = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "input"
    / "pdf_new"
    / "tegut_test.pdf"
)

use_demo = st.checkbox(
    "📂 Demo-PDF verwenden (lokal, kuratiert)",
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


st.markdown("### 🔍 Produkterkennung starten")

if st.button("🔍 Produkte erkennen", type="primary"):
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
# 4️⃣ Produkt-Crops
# ======================================================

st.markdown("## 4️⃣ Produkt-Crops")
st.caption(
    "Oben: vollständige Seite mit YOLO-Bounding-Boxes (inkl. Confidence). "
    "Darunter: einzelne Produkt-Crops."
)

# --------------------------------------------------
# 🧠 Ganze Seite mit YOLO-Boxen anzeigen
# --------------------------------------------------
if "RUN_DIR" in st.session_state:
    yolo_vis_dir = Path(st.session_state["RUN_DIR"]) / "yolo" / "detect"

    yolo_images = sorted(
        list(yolo_vis_dir.glob("*.png")) +
        list(yolo_vis_dir.glob("*.jpg")) +
        list(yolo_vis_dir.glob("*.jpeg"))
    )

    if yolo_images:
        st.markdown("### 🧠 YOLO-Ergebnis (ganze Seite)")

        st.image(
            Image.open(yolo_images[0]),
            use_container_width=True,
            caption="YOLO: erkannte Produkte inkl. Confidence"
        )
    else:
        st.warning("Kein YOLO-Visualisierungsbild gefunden.")

if "crop_paths" in st.session_state:

    selected = []
    cols = st.columns(4)

    for i, crop in enumerate(st.session_state["crop_paths"]):
        with cols[i % 4]:
            st.image(
                Image.open(crop["raw_path"]),
                use_container_width=True
            )
            if st.checkbox("Für OCR auswählen", key=f"crop_sel_{i}"):
                selected.append(crop)

    st.session_state["selected_crops"] = selected
else:
    st.info("Bitte zuerst Produkterkennung (Schritt 2) ausführen.")


# ======================================================
# 5️⃣ Optical character recognition (OCR)
# ======================================================

st.markdown("## 5️⃣ Optical character recognition (OCR)")
st.caption(
    "OCR wird auf genau **einem ausgewählten Produkt-Crop** durchgeführt."
)

if st.session_state.get("selected_crops"):

    crop_map = {
        c["raw_path"].name: c
        for c in st.session_state["selected_crops"]
    }

    selected_name = st.selectbox(
        "Produkt für OCR auswählen",
        list(crop_map.keys())
    )

    active_crop = crop_map[selected_name]

    # --------------------------------------------------
    # Bild laden (intern, nicht anzeigen)
    # --------------------------------------------------
    img_orig = cv2.imread(str(active_crop["raw_path"]))
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)



    # --------------------------------------------------
    # OCR-Sichtbereich einstellen
    # --------------------------------------------------
    st.markdown("### 🧹 OCR-Sichtbereich")

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
    # --------------------------------------------------
    # EINZIGES sichtbares Bild
    # --------------------------------------------------
    st.markdown("### 👁️ OCR sieht diesen Bereich")
    st.write("Kann genutzt werden um Störtext auszublenden")

    st.image(
        masked,
        caption="Live-OCR-Vorschau",
        width=300
    )

    # --------------------------------------------------
    # OCR starten
    # --------------------------------------------------
    st.info("OCR mit EasyOCR dauert ca. 20 Sekunden.")

    if st.button("🔤 OCR starten", type="primary"):
        res = extract_text_easyocr(masked)

        st.session_state["ocr_result"] = res
        st.session_state["ocr_masked"] = masked

        st.markdown("### 📑 OCR-Ergebnis")
        st.text(res["text"])


    else:
        st.info("Bitte zuerst mindestens ein Produkt in Schritt 4 auswählen.")

    # ==================================================
    # 📤 Export (JSON / Bild)
    # ==================================================
    if "ocr_result" in st.session_state:

        st.markdown("### 📤 Ergebnis exportieren")

        col1, col2 = st.columns(2)
        with col1:
            export_json = st.checkbox("📄 JSON exportieren", value=True)
        with col2:
            export_image = st.checkbox("🖼️ OCR-Bild exportieren", value=False)

        # -------------------------
        # Export ausführen
        # -------------------------
        if st.button("💾 Export erstellen", type="primary"):

            run_dir = Path(st.session_state["RUN_DIR"])
            stem = active_crop["raw_path"].stem

            # ---------- JSON ----------
            if export_json:
                export_data = {
                    "source": {
                        "flyer": pdf_name,
                        "page": 1,
                        "pipeline_version": "v1.0"
                    },
                    "product": {
                        "image_file": active_crop["raw_path"].name,
                        "bbox": active_crop.get("bbox", {})
                    },
                    "ocr": {
                        "engine": "easyocr",
                        "language": ["de"],
                        "masked_area": {
                            "top": top,
                            "bottom": bottom,
                            "left": left,
                            "right": right,
                        },
                        "text_raw": st.session_state["ocr_result"]["text"],
                        "confidence": st.session_state["ocr_result"].get("confidence")
                    }
                }

                json_bytes = json.dumps(export_data, indent=2, ensure_ascii=False)
                st.session_state["export_json_bytes"] = json_bytes
                st.session_state["export_json_name"] = f"{stem}_ocr.json"

            # ---------- IMAGE ----------
            if export_image:
                from PIL import Image
                import io

                img = Image.fromarray(st.session_state["ocr_masked"])
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)

                st.session_state["export_img_bytes"] = buf
                st.session_state["export_img_name"] = f"{stem}_ocr_masked.png"

            st.success("✅ Export vorbereitet – Download unten verfügbar")

    # ==================================================
    # ⬇️ Downloads (IMMER stabil)
    # ==================================================
    if "export_json_bytes" in st.session_state:
        st.download_button(
            "⬇️ JSON herunterladen",
            data=st.session_state["export_json_bytes"],
            file_name=st.session_state["export_json_name"],
            mime="application/json"
        )

    if "export_img_bytes" in st.session_state:
        st.download_button(
            "⬇️ OCR-Bild herunterladen",
            data=st.session_state["export_img_bytes"],
            file_name=st.session_state["export_img_name"],
            mime="image/png"
        )

