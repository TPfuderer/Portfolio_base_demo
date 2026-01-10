from pathlib import Path
from datetime import datetime
from uuid import uuid4

import fitz
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

from portfolio_base.tegut_ocr.paths import DATA_DIR, YOLO_MODEL


# ======================================================
# 🧠 Public API
# ======================================================

def detect_products(
    pdf_path: Path,
    dpi: int = 450,
    min_conf: float = 0.8
):
    """
    Detect products on a SINGLE PDF page.
    Creates its own isolated run_dir (demo & cloud safe).
    """

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    # --------------------------------------------------
    # 🔹 Create isolated run directory
    # --------------------------------------------------
    run_dir = _create_run_dir(DATA_DIR)

    pages_dir    = run_dir / "pages"
    yolo_dir     = run_dir / "yolo"
    crops_dir    = run_dir / "crops"
    labels_dir   = run_dir / "labels"
    filtered_dir = run_dir / "filtered"

    for d in [pages_dir, yolo_dir, crops_dir, labels_dir, filtered_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1️⃣ PDF → image (ONE page demo)
    # --------------------------------------------------
    page_images = _pdf_to_images(pdf_path, dpi, pages_dir)

    # --------------------------------------------------
    # 2️⃣ YOLO inference
    # --------------------------------------------------
    results = _run_yolo(page_images, yolo_dir, min_conf)

    # --------------------------------------------------
    # 3️⃣ Cropping
    # --------------------------------------------------
    crop_infos = _extract_crops(results, crops_dir, min_conf)

    # --------------------------------------------------
    # 4️⃣ Save labels
    # --------------------------------------------------
    _save_labels(results, labels_dir)

    # --------------------------------------------------
    # 5️⃣ Debug visualization
    # --------------------------------------------------
    _draw_filtered_boxes(results, filtered_dir, min_conf)

    return run_dir, crop_infos


# ======================================================
# 🔧 Internals
# ======================================================

def _create_run_dir(base_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"run_{ts}_{uuid4().hex[:6]}"
    run_dir = base_dir / "output" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _pdf_to_images(pdf_path: Path, dpi: int, pages_dir: Path) -> list[Path]:
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=dpi)
        out = pages_dir / f"{pdf_path.stem}_page_{i:02d}.png"
        pix.save(out)
        pages.append(out)

        break  # 🚨 DEMO: only ONE page

    doc.close()
    return pages


def _run_yolo(page_images: list[Path], yolo_dir: Path, min_conf: float):
    model = YOLO(YOLO_MODEL)

    return model.predict(
        source=[str(p) for p in page_images],
        conf=min_conf,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(yolo_dir),
        name="detect",
        exist_ok=True
    )


def _extract_crops(results, crops_dir: Path, min_conf: float) -> list[dict]:
    crop_infos = []

    raw_dir = crops_dir / "raw"
    ocr_dir = crops_dir / "ocr"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ocr_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        img_path = Path(result.path)
        img = Image.open(img_path)
        img_np = np.array(img)

        for i, box in enumerate(result.boxes):
            conf = float(box.conf[0])
            if conf < min_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])

            crop = img_np[y1:y2, x1:x2]

            base = f"{img_path.stem}_box{i+1:03d}_cls{cls}"
            raw_path = raw_dir / f"{base}.jpg"
            ocr_path = ocr_dir / f"{base}.jpg"

            Image.fromarray(crop).save(raw_path)
            Image.fromarray(crop).save(ocr_path)

            crop_infos.append({
                "raw_path": raw_path,
                "ocr_path": ocr_path,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "cls": cls,
                "page": img_path.name
            })

    return crop_infos


def _save_labels(results, labels_dir: Path):
    for result in results:
        stem = Path(result.path).stem
        result.save_txt(labels_dir / f"{stem}.txt", save_conf=True)


def _draw_filtered_boxes(results, out_dir: Path, min_conf: float):
    for result in results:
        img = cv2.imread(result.path)

        for box in result.boxes:
            if float(box.conf[0]) < min_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(str(out_dir / Path(result.path).name), img)
