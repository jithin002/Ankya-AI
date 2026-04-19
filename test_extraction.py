"""
test_extraction.py
==================
Standalone text extraction test harness extracted from icr_pipeline3.py.

Covers ONLY the extraction / pre-processing stack:
  ┌─────────────────────────────────────────────┐
  │  Image ──► preprocess_for_ocr               │
  │        ──► EasyOCR layout detection         │
  │        ──► TrOCR per-line refinement        │
  │        ──► Pytesseract ensemble (fallback)  │
  │        ──► merge_nearby_horizontal_blocks   │
  │        ──► coalesce_blocks                  │
  │        ──► split_text_by_question           │
  └─────────────────────────────────────────────┘

No scoring, grammar models, LLM or SentenceTransformer is loaded here.

Usage
-----
Run directly:
    python test_extraction.py path/to/image.jpg

Or import and call:
    from test_extraction import extract_text_from_image
    result = extract_text_from_image("path/to/image.jpg")
    print(result["full_text"])
    for seg in result["segments"]:
        print(seg)
"""

import os
import re
import sys
import time
import tempfile
from typing import List, Dict, Tuple, Optional

# ── Optional: set Tesseract path if not on PATH ─────────────────────────────
# Uncomment and fix this if pytesseract raises a "tesseract not found" error:
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
from PIL import Image
import easyocr
import pytesseract
from pytesseract import Output
import torch
from Levenshtein import ratio as lev_ratio


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Extraction] Using device: {DEVICE}")


# ─────────────────────────────────── Preprocessing ──────────────────────────

def deskew_image(img_gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """Rotate image to correct skew using Hough line detection."""
    try:
        edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 200)
        if lines is None:
            return img_gray, 0.0
        angles = []
        for rho_theta in lines:
            _, theta = rho_theta[0]
            angle_deg = (theta * 180.0 / np.pi) - 90.0
            if angle_deg > 90:
                angle_deg -= 180
            if angle_deg < -90:
                angle_deg += 180
            angles.append(angle_deg)
        if not angles:
            return img_gray, 0.0
        median_angle = float(np.median(angles))
        h, w = img_gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
        rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated, median_angle
    except Exception:
        return img_gray, 0.0


def remove_ruled_lines(img_bgr: np.ndarray) -> np.ndarray:
    """Erase horizontal ruled lines from notebook paper using morphological ops."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Use adaptive threshold instead of OTSU to handle uneven lighting/shadows
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(binary.shape[1] // 8), 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cleaned = cv2.subtract(binary, horizontal_lines)
    result = cv2.bitwise_not(cleaned)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Full preprocessing chain:
      1. CLAHE contrast boost (faded ink recovery)
      2. Morphological ruled-line removal
    Returns clean BGR image ready for EasyOCR.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    img_clahe = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return remove_ruled_lines(img_clahe)


def preprocess_image_for_ocr(image_path: str) -> np.ndarray:
    """
    Secondary preprocessing (used in ensemble fallback):
      bilateral filter → deskew → adaptive threshold → morphological open
    Returns a BGR image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    deskewed, angle = deskew_image(denoised)
    print(f"  [deskew] corrected angle: {angle:.2f}°")
    try:
        thresh = cv2.adaptiveThreshold(deskewed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
    except Exception:
        _, thresh = cv2.threshold(deskewed, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────── Text Utilities ─────────────────────────

def normalize_text(text: str) -> str:
    text = text.replace('\r', ' ').replace('\n', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def clean_ocr_noise(text: str) -> str:
    """Strip out patterns that come from ruled lines being misread by OCR."""
    text = re.sub(r'([^\w\s]){3,}', ' ', text)
    text = re.sub(r'([.,\'"`_^=-]\s+){2,}', ' ', text)
    text = re.sub(r'\s[^\w\s]{1,2}\s', ' ', text)
    text = re.sub(r'(?<=\s)[\'"`_]+|[\'"`_]+(?=\s)', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


_ENCIRCLED_DIGITS = {chr(9312 + i): str(i + 1) for i in range(20)}  # ①→1 … ⑳→20


def normalize_encircled_digits(text: str) -> str:
    """Replace Unicode circled-digit chars with plain-digit equivalents."""
    for ch, digit in _ENCIRCLED_DIGITS.items():
        text = text.replace(ch, f' {digit}) ')
    return text


# ─────────────────────────────────── OCR Engines ────────────────────────────

def pytesseract_lines_from_image(img_bgr: np.ndarray) -> List[Dict]:
    """Run Pytesseract and return line-level blocks."""
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    data = pytesseract.image_to_data(pil, output_type=Output.DICT, lang='eng')
    n_boxes = len(data['level'])
    lines: Dict[tuple, dict] = {}
    for i in range(n_boxes):
        text = str(data['text'][i]).strip()
        if not text:
            continue
        conf = float(data['conf'][i]) if float(data['conf'][i]) >= 0 else 0.0
        key = (
            data.get('block_num', [0] * n_boxes)[i],
            data.get('par_num', [0] * n_boxes)[i],
            data.get('line_num', [0] * n_boxes)[i],
        )
        if key not in lines:
            lines[key] = {"words": [], "confs": [], "bboxes": []}
        lines[key]["words"].append(text)
        lines[key]["confs"].append(conf)
        lines[key]["bboxes"].append((data['left'][i], data['top'][i], data['width'][i], data['height'][i]))

    out = []
    for _, v in lines.items():
        text = " ".join(v["words"])
        xs, ys, ws, hs = zip(*v["bboxes"])
        bbox = (
            min(xs),
            min(ys),
            max(x + w for x, w in zip(xs, ws)) - min(xs),
            max(y + h for y, h in zip(ys, hs)) - min(ys),
        )
        out.append({"text": text, "conf": float(np.mean(v["confs"])), "bbox": bbox})
    return out


def easyocr_blocks_from_image(img_bgr: np.ndarray, ocr_reader) -> List[Dict]:
    """Run EasyOCR and return block-level results with unified bbox format."""
    easy_res = ocr_reader.readtext(img_bgr, detail=1)
    out = []
    for bbox, text, conf in easy_res:
        xs, ys = zip(*bbox)
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        out.append({"text": text, "conf": float(conf) * 100.0, "bbox": (x, y, w, h), "cy": y + h / 2.0})
    return out


# ─────────────────────────────────── Ensemble Merger ────────────────────────

def bbox_iou(boxA: tuple, boxB: tuple) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = (boxA[2] * boxA[3]) + (boxB[2] * boxB[3]) - inter
    return inter / union if union > 0 else 0.0


def ensemble_ocr_preprocessed(img_bgr: np.ndarray, ocr_reader) -> List[Dict]:
    """
    Merge EasyOCR and Pytesseract results:
    - Prefer Pytesseract text when IoU overlap >= 0.15 AND
      (Pytesseract confidence is higher OR Levenshtein similarity >= 0.85)
    - Otherwise keep EasyOCR.
    - Append any Pytesseract-only blocks that had no overlap.
    """
    pyt_lines = pytesseract_lines_from_image(img_bgr)
    easy_blocks = easyocr_blocks_from_image(img_bgr, ocr_reader)

    merged = []
    used_pyt: set = set()
    for e in easy_blocks:
        ebox = e['bbox']
        overlaps = [(pi, p) for pi, p in enumerate(pyt_lines) if bbox_iou(ebox, p['bbox']) > 0.15]
        if overlaps:
            m_text = " ".join([p['text'] for _, p in overlaps]).strip()
            m_conf = float(np.mean([p['conf'] for _, p in overlaps]))
            sim = lev_ratio(e['text'].lower(), m_text.lower())
            if m_conf >= e['conf'] or sim > 0.85:
                res = {"text": m_text, "conf": m_conf, "bbox": ebox, "cy": e['cy']}
            else:
                res = e
            for pi, _ in overlaps:
                used_pyt.add(pi)
            merged.append(res)
        else:
            merged.append(e)

    for pi, p in enumerate(pyt_lines):
        if pi not in used_pyt:
            merged.append({"text": p['text'], "conf": p['conf'], "bbox": p['bbox'], "cy": p['bbox'][1] + p['bbox'][3] / 2})

    return sorted(merged, key=lambda x: x['cy'])


# ─────────────────────────────────── TrOCR ──────────────────────────────────

def load_trocr(device: str = DEVICE):
    """Load TrOCR (small-handwritten) with optional LoRA adapter."""
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    base = "microsoft/trocr-small-handwritten"
    print("[TrOCR] Loading base model...")
    t0 = time.time()
    try:
        model = VisionEncoderDecoderModel.from_pretrained(base).to(device)
        processor = ViTImageProcessor.from_pretrained(base)
        try:
            tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False)

        # LoRA adapter (fine_tuned_trocr_small)
        local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "fine_tuned_trocr_small"))
        adapter_path = None
        if os.path.exists(os.path.join(local_dir, "adapter_config.json")):
            adapter_path = local_dir
        elif os.path.exists(local_dir):
            checkpoints = sorted(
                [os.path.join(local_dir, d) for d in os.listdir(local_dir) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[-1])
            )
            for cp in reversed(checkpoints):
                if os.path.exists(os.path.join(cp, "adapter_config.json")):
                    adapter_path = cp
                    break

        if adapter_path:
            print(f"[TrOCR] Loading LoRA from {adapter_path} ...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
            print("[TrOCR] LoRA merged.")
        else:
            print("[TrOCR] No LoRA adapter found, using base model.")

        if device == "cuda":
            model = model.half()

        print(f"[TrOCR] Loaded in {time.time()-t0:.1f}s")
        return {"model": model, "processor": processor, "tokenizer": tokenizer}
    except Exception as exc:
        print(f"[TrOCR] Load failed: {exc}")
        return None


def trocr_recognize_pil(pil_img: Image.Image, bundle: dict, max_new_tokens: int = 256, num_beams: int = 4) -> str:
    if not bundle:
        return ""
    model = bundle["model"]
    processor = bundle["processor"]
    tokenizer = bundle["tokenizer"]
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(DEVICE)
    if DEVICE == "cuda":
        pixel_values = pixel_values.half()
    with torch.inference_mode():
        ids = model.generate(
            pixel_values, 
            max_new_tokens=max_new_tokens, 
            num_beams=num_beams, 
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True).strip()


def recognize_with_trocr_per_line(
    image_path: str,
    easyocr_blocks: List[Dict],
    trocr_bundle: dict,
    pad: int = 6,
    num_beams: int = 8,
) -> Tuple[str, List[Tuple[str, float, tuple]]]:
    """Use EasyOCR bboxes as region proposals; run TrOCR on each crop."""
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise FileNotFoundError(image_path)
    H, W = img_cv.shape[:2]

    merged = merge_nearby_horizontal_blocks(easyocr_blocks, gap_thresh=30)

    # Filter tiny / marginal boxes
    min_area = max(80, 0.0005 * (W * H))
    filtered = []
    for b in merged:
        x, y, w, h = b['bbox']
        if w * h < min_area:
            continue
        filtered.append(b)
    filtered = sorted(filtered, key=lambda b: b['bbox'][1])

    def _preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
        # Preprocessing: Convert to grayscale and apply CLAHE to boost contrast.
        # Avoid harsh binarization and morphological ops that destroy thin handwriting.
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            pass
        # Return as BGR since TrOCR processor converts it back to Pil/RGB anyway
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    results = []
    for b in filtered:
        x, y, w, h = b['bbox']
        x0, y0 = int(max(0, x - pad)), int(max(0, y - pad))
        x1, y1 = int(min(W, x + w + pad)), int(min(H, y + h + pad))
        crop = img_cv[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        pil = Image.fromarray(_preprocess_crop(crop)).convert("L").convert("RGB")
        try:
            txt = trocr_recognize_pil(pil, trocr_bundle, max_new_tokens=256, num_beams=num_beams)
            if not txt or not txt.strip():
                txt = b.get("text", "")
        except Exception as exc:
            print(f"  [TrOCR crop error] {exc}")
            txt = b.get("text", "")
        results.append((txt, float(b.get("conf", 0.0)), (x0, y0, x1 - x0, y1 - y0)))

    joined = "\n".join(r[0] for r in results)
    return joined, results


# ─────────────────────────────────── Block Merging ──────────────────────────

def merge_nearby_horizontal_blocks(blocks: List[Dict], gap_thresh: int = 20) -> List[Dict]:
    """Merge text blocks that are on the same horizontal line."""
    if not blocks:
        return []
    blocks = sorted(blocks, key=lambda b: (b['cy'], b['bbox'][0]))
    merged = [blocks[0].copy()]
    for b in blocks[1:]:
        cur = merged[-1]
        if abs(b['cy'] - cur['cy']) <= max(12, 0.4 * cur['bbox'][3]):
            cur_r = cur['bbox'][0] + cur['bbox'][2]
            if b['bbox'][0] - cur_r <= gap_thresh:
                nl = min(cur['bbox'][0], b['bbox'][0])
                nr = max(cur_r, b['bbox'][0] + b['bbox'][2])
                nt = min(cur['bbox'][1], b['bbox'][1])
                nb = max(cur['bbox'][1] + cur['bbox'][3], b['bbox'][1] + b['bbox'][3])
                cur['text'] += " " + b['text']
                cur['conf'] = (cur['conf'] + b['conf']) / 2
                cur['bbox'] = (nl, nt, nr - nl, nb - nt)
                cur['cy'] = (cur['cy'] + b['cy']) / 2
            else:
                merged.append(b.copy())
        else:
            merged.append(b.copy())
    return merged


def coalesce_blocks(blocks: List[Dict], y_tol: int = 12) -> List[Dict]:
    """Merge blocks that are vertically close into paragraph-level blocks."""
    if not blocks:
        return []
    groups = [[blocks[0]]]
    for b in blocks[1:]:
        if abs(b["cy"] - groups[-1][-1]["cy"]) <= y_tol:
            groups[-1].append(b)
        else:
            groups.append([b])

    out = []
    for g in groups:
        text = " ".join(x["text"] for x in g).strip()
        conf = float(np.mean([x["conf"] for x in g]))
        xs, ys, ws, hs = zip(*[x['bbox'] for x in g])
        x0, y0 = min(xs), min(ys)
        x1 = max(x + w for x, w in zip(xs, ws))
        y1 = max(y + h for y, h in zip(ys, hs))
        out.append({
            "text": text, "conf": conf,
            "bbox": (x0, y0, x1 - x0, y1 - y0),
            "cy": float(np.mean([x['cy'] for x in g])),
        })
    return sorted(out, key=lambda x: x['cy'])


# ─────────────────────────────────── Question Splitter ──────────────────────

def split_text_by_question(blocks: List[Dict]) -> List[Dict]:
    """
    Split OCR blocks into per-question segments.

    Recognised patterns (after whitespace or start of text):
      classic:   1)  (1)  [1]
      dotted:    1.  Q1.  Q1:
      written:   Q1  Q1:  question 1
      encircled: ①  ②  (normalised to '1) ' first)

    Returns list of:
      { 'q_label': '16', 'text': '...answer text...', 'y_pct': 0.35 }
    Falls back to a single 'full' segment if nothing detected.
    """
    Q_PAT = re.compile(
        r'(?:(?:^|(?<=\s))'
        r'(?:'
        r'(?:Q|q|question\s*)(\d{1,3})([ab]?)[:\.]?'   # Q1  Q1. Q1:
        r'|(?:\()(\d{1,3})([ab]?)[\)\]]'               # (1)  [1]
        r'|(\d{1,3})([ab]?)[\)\]]'                     # 1)  1]
        r'|(\d{1,3})([ab]?)\.(?=\s)'                   # 1.
        r'))'
    )

    segments: List[Dict] = []
    current_label: Optional[str] = None
    current_lines: List[str] = []
    current_y = 0.0

    ys = [b.get('y', b.get('cy', 0)) for b in blocks]
    first_y = min(ys) if ys else 0
    last_y = max(ys) if ys else 1
    span = (last_y - first_y) or 1

    for block in blocks:
        raw = block.get('text', '').strip()
        text = normalize_encircled_digits(raw)
        y = block.get('y', block.get('cy', 0))
        y_pct = (y - first_y) / span

        m = Q_PAT.search(text)
        if m:
            if current_lines:
                segments.append({
                    'q_label': current_label or 'full',
                    'text': clean_ocr_noise(' '.join(current_lines)),
                    'y_pct': current_y,
                })
            q_num = next((g for g in m.groups()[0::2] if g is not None), None)
            q_sub = next((g for g in m.groups()[1::2] if g is not None), '') or ''
            current_label = f"{q_num}{q_sub}" if q_num else 'full'
            current_y = y_pct
            rest = text[m.end():].strip()
            current_lines = [rest] if rest else []
        else:
            current_lines.append(text)

    if current_lines:
        segments.append({
            'q_label': current_label or 'full',
            'text': clean_ocr_noise(' '.join(current_lines)),
            'y_pct': current_y,
        })

    if not segments:
        all_text = clean_ocr_noise(' '.join(
            normalize_encircled_digits(b.get('text', '')) for b in blocks
        ))
        return [{'q_label': 'full', 'text': all_text, 'y_pct': 0.0}]

    return segments


# ─────────────────────────────────── Main Entry ──────────────────────────────

def extract_text_from_image(
    image_path: str,
    use_trocr: bool = True,
    save_preprocessed: bool = False,
) -> Dict:
    """
    Full extraction pipeline.

    Parameters
    ----------
    image_path : str
        Path to the image file (JPG / PNG / etc.)
    use_trocr : bool
        Whether to attempt TrOCR refinement (True by default).
        Set to False for speed or if TrOCR dependencies aren't available.
    save_preprocessed : bool
        If True, saves the preprocessed image next to the source for inspection.

    Returns
    -------
    dict with keys:
        full_text  : str          – joined cleaned text from whole page
        blocks     : List[Dict]   – raw OCR blocks (text, conf, bbox, cy)
        segments   : List[Dict]   – per-question segments (q_label, text, y_pct)
        avg_conf   : float        – mean OCR confidence (0-100)
        ocr_mode   : str          – 'trocr', 'ensemble_fallback'
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"\n{'='*60}")
    print(f" Extracting: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    # ── 0. Pre-process: CLAHE + ruled-line removal ─────────────────────────
    raw = cv2.imread(image_path)
    if raw is None:
        raise ValueError(f"cv2 could not read: {image_path}")

    cleaned = preprocess_for_ocr(raw)
    ocr_source_path = image_path  # default: original

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        cv2.imwrite(tmp.name, cleaned)
        ocr_source_path = tmp.name
    print(f"  [preprocess] saved cleaned image → {ocr_source_path}")

    if save_preprocessed:
        base, ext = os.path.splitext(image_path)
        out_path = base + "_preprocessed" + ext
        cv2.imwrite(out_path, cleaned)
        print(f"  [preprocess] also saved to → {out_path}")

    # ── 1. EasyOCR layout ─────────────────────────────────────────────────
    print("  [OCR] Loading EasyOCR ...")
    t0 = time.time()
    ocr_reader = easyocr.Reader(['en'], gpu=(DEVICE == "cuda"))
    print(f"  [OCR] EasyOCR ready in {time.time()-t0:.1f}s")

    easy_layout = easyocr_blocks_from_image(cv2.imread(ocr_source_path), ocr_reader)
    print(f"  [OCR] EasyOCR detected {len(easy_layout)} blocks")

    ocr_mode = "ensemble_fallback"
    blocks: List[Dict] = []

    # ── 2. TrOCR refinement (optional) ───────────────────────────────────
    if use_trocr:
        try:
            trocr_bundle = load_trocr(DEVICE)
            if trocr_bundle:
                joined_text, per_line = recognize_with_trocr_per_line(ocr_source_path, easy_layout, trocr_bundle)
                blocks = [
                    {"text": normalize_text(t), "conf": float(c), "bbox": b, "cy": b[1] + b[3] / 2}
                    for t, c, b in per_line
                ]
                blocks = [bl for bl in blocks if bl["text"].strip()]
                blocks = sorted(blocks, key=lambda x: x['cy'])
                ocr_mode = "trocr"
                print(f"  [OCR] TrOCR produced {len(blocks)} blocks")
                # Free VRAM
                del trocr_bundle
                import gc; gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
        except Exception as exc:
            print(f"  [TrOCR] failed ({exc}), falling back to ensemble.")

    # ── 3. Ensemble fallback ──────────────────────────────────────────────
    if not blocks:
        print("  [OCR] Running ensemble (EasyOCR + Pytesseract) ...")
        processed_bgr = preprocess_image_for_ocr(ocr_source_path)
        blocks = ensemble_ocr_preprocessed(processed_bgr, ocr_reader)
        for b in blocks:
            b['text'] = normalize_text(b['text'])
        blocks = [b for b in blocks if b['text'].strip()]
        print(f"  [OCR] Ensemble produced {len(blocks)} blocks")

    # Cleanup
    del ocr_reader
    try:
        os.unlink(ocr_source_path)
    except Exception:
        pass

    # ── 4. Merge + coalesce ───────────────────────────────────────────────
    blocks = merge_nearby_horizontal_blocks(blocks, gap_thresh=20)
    blocks = coalesce_blocks(blocks, y_tol=12)
    print(f"  [merge] {len(blocks)} blocks after merging")

    # ── 5. Full text + question splitting ────────────────────────────────
    full_text = clean_ocr_noise(" ".join(b["text"] for b in blocks))
    avg_conf = float(np.mean([b.get("conf", 0) for b in blocks])) if blocks else 0.0

    segments = split_text_by_question(blocks)
    print(f"  [split] {len(segments)} question segment(s) detected")

    return {
        "full_text": full_text,
        "blocks": blocks,
        "segments": segments,
        "avg_conf": avg_conf,
        "ocr_mode": ocr_mode,
    }


# ─────────────────────────────────── CLI ─────────────────────────────────────

def _pretty_print(result: Dict) -> None:
    print(f"\n{'─'*60}")
    print(f"  OCR mode  : {result['ocr_mode']}")
    print(f"  Avg conf  : {result['avg_conf']:.1f}%")
    print(f"  Blocks    : {len(result['blocks'])}")
    print(f"  Segments  : {len(result['segments'])}")
    print(f"\n  ── FULL TEXT ──────────────────────────────────────────────")
    print(f"  {result['full_text']}")
    print(f"\n  ── SEGMENTS ───────────────────────────────────────────────")
    for seg in result['segments']:
        print(f"  Q{seg['q_label']} (y≈{seg['y_pct']:.2f}): {seg['text'][:120]}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_extraction.py <image_path> [--no-trocr] [--save-preprocessed]")
        sys.exit(1)

    img = sys.argv[1]
    use_trocr = "--no-trocr" not in sys.argv
    save_pre = "--save-preprocessed" in sys.argv

    result = extract_text_from_image(img, use_trocr=use_trocr, save_preprocessed=save_pre)
    _pretty_print(result)
