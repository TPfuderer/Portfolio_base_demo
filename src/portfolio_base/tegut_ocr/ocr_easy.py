from pathlib import Path
from typing import Union
import numpy as np
import easyocr
import cv2

# ======================================================
# 🧠 EasyOCR Reader (einmal initialisieren!)
# ======================================================

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(
            lang_list=["de", "en"],
            gpu=False,
            recog_network="latin_g2",   # besser für Zahlen & Sonderzeichen
            quantize=False              # höhere Präzision
        )
    return _reader


# ======================================================
# 🔍 Preprocessing (EasyOCR-freundlich, minimal)
# ======================================================

def preprocess_for_easyocr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Kontrast-Normalisierung
    gray = cv2.normalize(
        gray, None,
        alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    # Leichtes Entrauschen
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray


# ======================================================
# 🧹 OCR-spezifisches Cleanup (kein Parsing!)
# ======================================================

def _clean_easyocr_text(text: str) -> str:
    replacements = {
        " ,": ",",
        " .": ".",
        "€ .": "€",
        "glnstiger": "günstiger",
        "Glnstiger": "Günstiger",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    # Mehrfach-Leerzeichen entfernen
    text = " ".join(text.split())

    return text.strip()


# ======================================================
# 🔍 Public OCR API
# ======================================================

def extract_text_easyocr(
    image: Union[np.ndarray, Path],
    min_confidence: float = 0.3
) -> dict:
    """
    OCR für Flyer-Crops (Deutsch, Zahlen, Sonderzeichen)

    Parameters
    ----------
    image : np.ndarray | Path
        Bild als Array oder Pfad
    min_confidence : float
        Mindest-Confidence für Textzeilen

    Returns
    -------
    dict mit:
        text: str
        lines: list[str]
        confidences: list[float]
        mean_confidence: float
    """

    # ----------------------------------------------
    # Bild laden
    # ----------------------------------------------
    if isinstance(image, Path):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Bild konnte nicht geladen werden: {image}")
    else:
        img = image.copy()

    # ----------------------------------------------
    # Preprocessing
    # ----------------------------------------------
    gray = preprocess_for_easyocr(img)

    # ----------------------------------------------
    # OCR
    # ----------------------------------------------
    reader = _get_reader()

    results = reader.readtext(
        gray,
        detail=1,
        paragraph=False,
        decoder="greedy",
        text_threshold=0.6,
        low_text=0.4,
        contrast_ths=0.1,
        adjust_contrast=0.5,
        allowlist="0123456789.,€% abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüÄÖÜß"
        #                ↑ wichtiges Leerzeichen hier
    )

    # ----------------------------------------------
    # Ergebnisse sammeln
    # ----------------------------------------------
    lines: list[str] = []
    confidences: list[float] = []

    for _, text, conf in results:
        if conf < min_confidence:
            continue

        cleaned = _clean_easyocr_text(text)
        if cleaned:
            lines.append(cleaned)
            confidences.append(float(conf))

    full_text = " ".join(lines)
    mean_conf = float(np.mean(confidences)) if confidences else 0.0

    return {
        "text": full_text,
        "lines": lines,
        "confidences": confidences,
        "mean_confidence": mean_conf,
    }
