from pathlib import Path
import shutil
import tempfile
import threading
from typing import Union
import zipfile
import zlib

import cv2
import easyocr
import numpy as np


# ======================================================
# EasyOCR Reader (initialize once, safely)
# ======================================================

_reader = None
_reader_lock = threading.Lock()
EASYOCR_MODEL_DIR = Path(tempfile.gettempdir()) / "easyocr_models"


def _build_reader():
    EASYOCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return easyocr.Reader(
        lang_list=["de", "en"],
        gpu=False,
        recog_network="latin_g2",  # better for digits and symbols
        quantize=False,             # higher precision
        model_storage_directory=str(EASYOCR_MODEL_DIR),
    )


def _clear_corrupted_easyocr_cache():
    # A partial/corrupted model archive can cause zlib/zip errors on unzip.
    if EASYOCR_MODEL_DIR.exists():
        shutil.rmtree(EASYOCR_MODEL_DIR, ignore_errors=True)


def _get_reader():
    global _reader

    if _reader is not None:
        return _reader

    with _reader_lock:
        if _reader is not None:
            return _reader

        try:
            _reader = _build_reader()
        except (zlib.error, zipfile.BadZipFile, EOFError):
            _clear_corrupted_easyocr_cache()
            _reader = _build_reader()

    return _reader


# ======================================================
# Preprocessing (minimal, OCR friendly)
# ======================================================

def preprocess_for_easyocr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


# ======================================================
# OCR cleanup (no semantic parsing)
# ======================================================

def _clean_easyocr_text(text: str) -> str:
    replacements = {
        " ,": ",",
        " .": ".",
        "\u20ac .": "\u20ac",
        "glnstiger": "g\u00fcnstiger",
        "Glnstiger": "G\u00fcnstiger",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = " ".join(text.split())
    return text.strip()


# ======================================================
# Public OCR API
# ======================================================

def extract_text_easyocr(
    image: Union[np.ndarray, Path],
    min_confidence: float = 0.3,
) -> dict:
    """
    OCR for flyer crops.

    Parameters
    ----------
    image : np.ndarray | Path
        Image array or image path
    min_confidence : float
        Minimum confidence threshold for text lines
    """

    # ----------------------------------------------
    # Load image
    # ----------------------------------------------
    if isinstance(image, Path):
        img_bgr = cv2.imread(str(image))
        if img_bgr is None:
            raise ValueError(f"Image could not be loaded: {image}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img = image.copy()

    debug_path = Path(tempfile.gettempdir()) / "debug_ocr_input.png"
    cv2.imwrite(str(debug_path), img)

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
        allowlist=(
            "0123456789.,\u20ac% "
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc\u00df"
        ),
    )

    # ----------------------------------------------
    # Collect outputs
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
