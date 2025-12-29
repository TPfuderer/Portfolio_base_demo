from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import fitz
import cv2
import numpy as np

from portfolio_base.tegut_ocr.paths import YOLO_MODEL


# ======================================================
# 🧠 Public API
# ======================================================

def detect_products(
    pdf_path: Path,
    run_dir: Path,
    dpi: int = 450,
    min_conf: float = 0.8
) -> list[dict]:
    """
    Detect products on a SINGLE PDF page.
    All outputs are written strictly inside run_dir.
    """

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    # --------------------------------------------------
    # Directory layout (APP controls run_dir!)
    # --------------------------------------------------
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
    # 4️⃣ Save labels (absolute pixel format)
    # --------------------------------------------------
    _save_labels(results, labels_dir)

    # --------------------------------------------------
    # 5️⃣ Debug visualization (optional)
    # --------------------------------------------------
    _draw_filtered_boxes(results, filtered_dir, min_conf)

    return crop_infos


# ======================================================
# 🔧 Internals
# ======================================================

def _pdf_to_images(pdf_path: Path, dpi: int, out_dir: Path) -> list[Path]:
    """
    Render ONLY the first page (demo-friendly).
    """
    doc = fitz.open(pdf_path)
    page = doc[0]

    pix = page.get_pixmap(dpi=dpi)
    out = out_dir / f"{pdf_path.stem}_page_01.png"
    pix.save(out)

    doc.close()
    return [out]


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


def _extract_crops(
    results,
    crops_dir: Path,
    min_conf: float
) -> list[dict]:

    raw_dir = crops_dir / "raw"
    ocr_dir = crops_dir / "ocr"
    raw_dir.mkdir(exist_ok=True)
    ocr_dir.mkdir(exist_ok=True)

    crop_infos = []

    for result in results:
        img_path = Path(result.path)
        img = Image.open(img_path)
        img_np = np.array(img)

        for j, box in enumerate(result.boxes):
            conf = float(box.conf[0])
            if conf < min_conf:
                continue

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().numpy()
            )
            cls = int(box.cls[0])

            crop = img_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            base = f"{img_path.stem}_box{j+1:03d}_cls{cls}"

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
    """
    Save labels in ABSOLUTE pixel format:
    cls conf x1 y1 x2 y2
    """
    for result in results:
        stem = Path(result.path).stem
        out = labels_dir / f"{stem}.txt"

        lines = []
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().numpy()
            )
            lines.append(f"{cls} {conf:.4f} {x1} {y1} {x2} {y2}")

        out.write_text("\n".join(lines), encoding="utf-8")


def _draw_filtered_boxes(results, out_dir: Path, min_conf: float):
    """
    Optional debug output: draws kept boxes only.
    """
    for result in results:
        img = cv2.imread(result.path)
        if img is None:
            continue

        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < min_conf:
                continue

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().numpy()
            )
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out_path = out_dir / Path(result.path).name
        cv2.imwrite(str(out_path), img)
