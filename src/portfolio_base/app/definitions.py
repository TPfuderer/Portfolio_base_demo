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