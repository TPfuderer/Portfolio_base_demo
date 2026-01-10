from pathlib import Path
from typing import Union
import numpy as np
import easyocr

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
    """
    Minimal, stabile OCR-Vorverarbeitung OHNE OpenCV.
    Liefert uint8-Graustufenbild für EasyOCR.
    """

    # ---------------------------
    # 1) RGB → Graustufen
    # ---------------------------
    if img.ndim == 3:
        gray = (
            0.299 * img[..., 0] +
            0.587 * img[..., 1] +
            0.114 * img[..., 2]
        )
    else:
        gray = img.copy()

    gray = gray.astype(np.float32)

    # ---------------------------
    # 2) Kontrast-Normalisierung
    # ---------------------------
    min_val = gray.min()
    max_val = gray.max()

    if max_val > min_val:
        gray = (gray - min_val) / (max_val - min_val)
        gray = gray * 255.0

    gray = gray.astype(np.uint8)

    # ---------------------------
    # 3) Sehr leichtes Glätten
    # (billiger Ersatz für GaussianBlur)
    # ---------------------------
    gray = (
        gray +
        np.roll(gray, 1, axis=0) +
        np.roll(gray, -1, axis=0) +
        np.roll(gray, 1, axis=1) +
        np.roll(gray, -1, axis=1)
    ) // 5

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
    from PIL import Image
    if isinstance(image, Path):
        img = np.array(Image.open(image).convert("RGB"))
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
