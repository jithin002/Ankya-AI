# pipelines/icr_pipeline2.py
import os
import json
import re
import tempfile
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import easyocr
import spacy
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# language_tool_python removed (we use model-based grammar)
from Levenshtein import ratio as lev_ratio
import Levenshtein
import hashlib
import shelve
import cv2
import pytesseract
from pytesseract import Output

# ----------------------- Config / Models -----------------------
nlp = spacy.load("en_core_web_sm")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Initialize OCR (use GPU if available)
OCR_READER = easyocr.Reader(['en'], gpu=(device == "cuda"))

# Sentence embeddings model
SENT_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# LLM grader: flan-t5-small
LLM_MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME).to(device)

# Grammar correction model (public model)
GRAMMAR_MODEL_NAME = "prithivida/grammar_error_correcter_v1"
print("Loading grammar-correction model on device:", device)
_g_tokenizer = AutoTokenizer.from_pretrained(GRAMMAR_MODEL_NAME)
_g_model = AutoModelForSeq2SeqLM.from_pretrained(GRAMMAR_MODEL_NAME).to(device)
_grammar_cache = shelve.open("grammar_cache.db")

# ----------------------- Utilities -----------------------
def normalize_text(text: str) -> str:
    text = text.replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------- Image preprocessing: deskew + denoise ----------------
def deskew_image(img_gray):
    """Estimate skew angle using Hough lines and rotate to deskew."""
    try:
        edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180.0, 200)
        if lines is None:
            return img_gray, 0.0
        angles = []
        for rho_theta in lines:
            rho, theta = rho_theta[0]
            angle_deg = (theta * 180.0 / np.pi) - 90.0
            if angle_deg > 90:
                angle_deg -= 180
            if angle_deg < -90:
                angle_deg += 180
            angles.append(angle_deg)
        if not angles:
            return img_gray, 0.0
        median_angle = float(np.median(angles))
        (h, w) = img_gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated, median_angle
    except Exception as e:
        # fallback: return original
        print("deskew_image error:", e)
        return img_gray, 0.0

def preprocess_image_for_ocr(image_path: str, save_tmp: bool = False):
    ###Loads image, converts to gray, denoises, deskews and applies adaptive thresholding.
    ###Returns the preprocessed BGR image (uint8) suitable for EasyOCR/pytesseract.
    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # denoise
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    # deskew
    deskewed, angle = deskew_image(denoised)
    # adaptive threshold (binary)
    try:
        thresh = cv2.adaptiveThreshold(deskewed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 8)
    except Exception:
        # fallback global threshold
        _, thresh = cv2.threshold(deskewed, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # morphological open/close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    processed_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    ###edits###
    if save_tmp:
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(tmpf.name, processed_bgr)
        return tmpf.name
    return processed_bgr
    ####
    try:
        upscaled = cv2.resize(processed_bgr, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    except Exception:
        upscaled = processed_bgr

    if save_tmp:
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(tmpf.name, upscaled)
        return tmpf.name
    return upscaled

###edit on 29-11-25###
"""def preprocess_image_for_ocr(image_path: str, save_tmp: bool = False):

    ####Enhanced preprocessing: gray -> CLAHE -> denoise -> deskew -> remove ruled lines
    -> adaptive threshold -> closing -> upscale
    ####
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE Contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 2. Denoise
    gray = cv2.medianBlur(gray, 3)

    # 3. Deskew
    deskewed, angle = deskew_image(gray)

    # 4. Remove ruled lines
    edges = cv2.Canny(deskewed, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150,
                            minLineLength=int(0.4 * deskewed.shape[1]),
                            maxLineGap=20)
    mask = np.zeros_like(deskewed)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)

    try:
        deskewed_no_lines = cv2.inpaint(deskewed, mask, 3, cv2.INPAINT_TELEA)
    except:
        deskewed_no_lines = deskewed

    # 5. Adaptive Threshold
    try:
        thresh = cv2.adaptiveThreshold(deskewed_no_lines, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 25, 9)
    except:
        _, thresh = cv2.threshold(deskewed_no_lines, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. Morphological Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 7. Upscale ×2.0
    upscaled = cv2.resize(closed, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    processed_bgr = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

    if save_tmp:
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(tmpf.name, processed_bgr)
        return tmpf.name

    return processed_bgr"""
 

# ---------------- Ensemble OCR: EasyOCR + pytesseract ----------------
def pytesseract_lines_from_image(img_bgr):
    """
    Runs pytesseract on image and returns list of dicts:
    [{'text':..., 'conf':float, 'bbox':(x,y,w,h)}, ...]
    """
    # ensure image is RGB for pytesseract
    if isinstance(img_bgr, np.ndarray):
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    else:
        pil = Image.open(img_bgr).convert("RGB")
    data = pytesseract.image_to_data(pil, output_type=Output.DICT, lang='eng')
    n_boxes = len(data['level'])
    lines = {}
    for i in range(n_boxes):
        text = str(data['text'][i]).strip()
        if text == "":
            continue
        left = int(data['left'][i]); top = int(data['top'][i]); width = int(data['width'][i]); height = int(data['height'][i])
        conf_str = data['conf'][i]
        try:
            conf = float(conf_str)
        except:
            conf = -1.0
        # group by line_num
        block_num = data.get('block_num', [0]*n_boxes)[i]
        par_num = data.get('par_num', [0]*n_boxes)[i]
        line_num = data.get('line_num', [0]*n_boxes)[i]
        key = (block_num, par_num, line_num)
        if key not in lines:
            lines[key] = {"words": [], "confs": [], "bboxes": []}
        lines[key]["words"].append(text)
        lines[key]["confs"].append(conf)
        lines[key]["bboxes"].append((left, top, width, height))
    out = []
    for k, v in lines.items():
        text = " ".join(v["words"])
        xs = [b[0] for b in v["bboxes"]]
        ys = [b[1] for b in v["bboxes"]]
        ws = [b[2] for b in v["bboxes"]]
        hs = [b[3] for b in v["bboxes"]]
        left = min(xs); top = min(ys)
        right = max([x+w for x,w in zip(xs, ws)])
        bottom = max([y+h for y,h in zip(ys, hs)])
        bbox = (left, top, right-left, bottom-top)
        avg_conf = float(np.mean([c for c in v["confs"] if c >= 0])) if v["confs"] else -1.0
        out.append({"text": text, "conf": avg_conf, "bbox": bbox})
    return out

def easyocr_blocks_from_image(img_bgr, ocr_reader):
    """
    Runs EasyOCR and returns list of dicts: {'text', 'conf', 'bbox'}
    bbox format converted to (x,y,w,h)
    """
    # pass numpy image directly
    easy_res = ocr_reader.readtext(img_bgr, detail=1)  # list of (bbox, text, conf)
    out = []
    for bbox, text, conf in easy_res:
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
        x = min(xs); y = min(ys); w = max(xs)-x; h = max(ys)-y
        out.append({"text": text, "conf": float(conf), "bbox": (x, y, w, h)})
    return out

def bbox_iou(boxA, boxB):
    # box = (x,y,w,h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    union = areaA + areaB - interArea
    if union == 0:
        return 0.0
    return interArea / union

def ensemble_ocr_preprocessed(img_bgr, ocr_reader):
    """
    Run EasyOCR + pytesseract on the preprocessed BGR image (numpy array).
    Merge blocks by bbox overlap + confidences. Return list of blocks:
    [{'text', 'conf', 'bbox', 'cy'} ..] sorted top->bottom by center y.
    """
    pyt_lines = pytesseract_lines_from_image(img_bgr)
    easy_blocks = easyocr_blocks_from_image(img_bgr, ocr_reader)

    merged = []
    used_pyt = set()

    for e in easy_blocks:
        ebox = e['bbox']
        overlaps = []
        for pi, p in enumerate(pyt_lines):
            iou = bbox_iou(ebox, p['bbox'])
            if iou > 0.15:
                overlaps.append((pi, p))
        if overlaps:
            merged_texts = [p['text'] for (_, p) in overlaps]
            merged_pyt_text = " ".join(merged_texts).strip()
            merged_pyt_conf = float(np.mean([p['conf'] if p['conf'] >= 0 else 0.0 for (_, p) in overlaps])) if overlaps else 0.0
            sim = lev_ratio(e['text'].lower(), merged_pyt_text.lower()) if merged_pyt_text else 0.0
            if merged_pyt_conf >= e['conf'] or sim > 0.85:
                chosen_text = merged_pyt_text
                chosen_conf = merged_pyt_conf
            else:
                chosen_text = e['text']
                chosen_conf = e['conf']
            for (pi, _) in overlaps:
                used_pyt.add(pi)
            cy = ebox[1] + ebox[3] / 2
            merged.append({"text": chosen_text, "conf": float(chosen_conf), "bbox": ebox, "cy": cy})
        else:
            cy = ebox[1] + ebox[3] / 2
            merged.append({"text": e['text'], "conf": float(e['conf']), "bbox": ebox, "cy": cy})

    # add remaining pytesseract lines
    for pi, p in enumerate(pyt_lines):
        if pi in used_pyt:
            continue
        bbox = p['bbox']
        cy = bbox[1] + bbox[3] / 2
        merged.append({"text": p['text'], "conf": float(p['conf']) if p['conf'] >= 0 else 0.0, "bbox": bbox, "cy": cy})

    merged = sorted(merged, key=lambda x: x['cy'])
    return merged

# ---------------- Main OCR entrypoint ----------------
def ocr_image_to_blocks(image_path: str, save_debug_image: bool = False) -> List[Dict]:
    """
    Full preprocess + ensemble OCR entry point.
    Returns list of paragraphs/blocks: {'text','conf','bbox','cy'}
    """
    processed_bgr = preprocess_image_for_ocr(image_path, save_tmp=False)
    blocks = ensemble_ocr_preprocessed(processed_bgr, OCR_READER)
    for b in blocks:
        b['text'] = normalize_text(b['text'])
    blocks = [b for b in blocks if len(b['text'].strip()) > 0]
    if save_debug_image:
        try:
            debug_path = image_path + ".preproc_debug.png"
            visualize_blocks_on_image(processed_bgr, blocks, debug_path)
            print("[DEBUG] wrote preproc debug:", debug_path)
        except Exception as e:
            print("visualize debug failed:", e)
    return blocks

def visualize_blocks_on_image(img_bgr, blocks, out_path):
    """Draw rectangles and text on the preprocessed image and save for debugging."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    for i, b in enumerate(blocks):
        x, y, w, h = [int(v) for v in b['bbox']]
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        txt = b['text'][:80]
        draw.text((x, max(0, y - 15)), f"{i}:{round(b['conf'],2)} {txt}", fill="blue", font=font)
    pil.save(out_path)

def merge_nearby_horizontal_blocks(blocks: List[Dict], gap_thresh: int = 20) -> List[Dict]:
    """
    Merge blocks that are on the same horizontal band and whose horizontal gap is small.
    gap_thresh is in pixels (increase for very spaced handwriting).
    """
    if not blocks:
        return []
    # sort by y then x
    blocks = sorted(blocks, key=lambda b: (b['cy'], b['bbox'][0]))
    merged = []
    cur = blocks[0].copy()
    for b in blocks[1:]:
        # same approximate line?
        if abs(b['cy'] - cur['cy']) <= max(12, 0.4 * cur['bbox'][3]):
            # compute horizontal gap between cur and b
            cur_right = cur['bbox'][0] + cur['bbox'][2]
            gap = b['bbox'][0] - cur_right
            if gap <= gap_thresh:
                # merge: extend bbox, concat text, average conf
                new_left = min(cur['bbox'][0], b['bbox'][0])
                new_right = max(cur['bbox'][0] + cur['bbox'][2], cur_right + gap)
                new_w = max(cur['bbox'][2], (new_right - new_left))
                new_bbox = (new_left, min(cur['bbox'][1], b['bbox'][1]), new_w, max(cur['bbox'][3], b['bbox'][3]))
                cur['text'] = (cur['text'] + " " + b['text']).strip()
                cur['conf'] = float(np.mean([cur.get('conf',0.0), b.get('conf',0.0)]))
                cur['bbox'] = new_bbox
                cur['cy'] = float((cur['cy'] + b['cy'])/2.0)
            else:
                merged.append(cur)
                cur = b.copy()
        else:
            merged.append(cur)
            cur = b.copy()
    merged.append(cur)
    return merged


# ---------------- coalesce_blocks ----------------
def coalesce_blocks(blocks: List[Dict], y_tol: int = 12) -> List[Dict]:

    """Group nearby lines into paragraph blocks based on dynamic y coordinate tolerance.
    If y_tol is None, estimate from median bbox height (robust for different handwriting sizes)."""

    if not blocks:
        return []
    
    # compute median line height -> use fraction of it for tolerance if not provided
    heights = [b['bbox'][3] for b in blocks if 'bbox' in b and len(b['bbox']) >= 4]
    median_h = float(np.median(heights)) if heights else 20.0
    if y_tol is None:
        # lines with center distance <= 0.9 * median_h are considered same paragraph
        y_tol = max(12.0, 0.9 * median_h)

    groups = []
    cur_group = [blocks[0]]
    for b in blocks[1:]:
        if abs(b["cy"] - cur_group[-1]["cy"]) <= y_tol:
            cur_group.append(b)
        else:
            groups.append(cur_group)
            cur_group = [b]
    groups.append(cur_group)
    out = []
    for g in groups:
        text = " ".join([x["text"] for x in g]).strip()
        conf = float(np.mean([x.get("conf", 0.0) for x in g]))
        xs = [x['bbox'][0] for x in g]
        ys = [x['bbox'][1] for x in g]
        ws = [x['bbox'][2] for x in g]
        hs = [x['bbox'][3] for x in g]
        x0 = min(xs); y0 = min(ys)
        x1 = max([x + w for x, w in zip(xs, ws)])
        y1 = max([y + h for y, h in zip(ys, hs)])
        bbox = (x0, y0, x1 - x0, y1 - y0)
        cy = float(np.mean([x['cy'] for x in g]))
        out.append({"text": text, "conf": conf, "bbox": bbox, "cy": cy})
    out = sorted(out, key=lambda x: x['cy'])
    return out

# ---------------- Scoring functions ----------------
def keyword_score(student_text: str, keywords: List[Dict], fuzz_threshold=0.8) -> float:
    st = student_text.lower()
    total_weight = sum(k.get('weight', 1.0) for k in keywords) if keywords else 0.0
    if total_weight == 0:
        return 100.0
    matched = 0.0
    for kw in keywords:
        term = kw['term'].lower()
        w = kw.get('weight', 1.0)
        if term in st:
            matched += w
            continue
        tokens = re.findall(r'\w+', st)
        best = 0.0
        for t in tokens:
            best = max(best, lev_ratio(term, t))
        if best >= fuzz_threshold:
            matched += w
    return float(matched / total_weight * 100.0)

def semantic_score(student_text: str, reference_texts: List[str]) -> float:
    s_emb = SENT_MODEL.encode([student_text], convert_to_numpy=True)
    r_embs = SENT_MODEL.encode(reference_texts, convert_to_numpy=True)
    r_avg = np.mean(r_embs, axis=0, keepdims=True)
    sim = np.dot(s_emb, r_avg.T) / (np.linalg.norm(s_emb, axis=1, keepdims=True) * np.linalg.norm(r_avg, axis=1))
    cos = float(sim[0][0])
    return float(((cos + 1) / 2) * 100.0)

def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def correct_text_with_model(text: str, max_length: int = 512) -> str:
    key = _cache_key(text)
    if key in _grammar_cache:
        return _grammar_cache[key]
    inp = text
    inputs = _g_tokenizer.encode(inp, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        out = _g_model.generate(inputs, max_new_tokens=256, do_sample=False)
    corrected = _g_tokenizer.decode(out[0], skip_special_tokens=True).strip()
    _grammar_cache[key] = corrected
    _grammar_cache.sync()
    return corrected

def grammar_score_model(student_text: str) -> float:
    if not student_text or student_text.strip() == "":
        return 0.0
    to_process = student_text if len(student_text) <= 2000 else student_text[:2000]
    try:
        corrected = correct_text_with_model(to_process)
    except Exception as e:
        print("Grammar model failed:", e)
        return 70.0
    orig = to_process.strip()
    corr = corrected.strip()
    ed = Levenshtein.distance(orig, corr)
    maxlen = max(1, len(orig))
    norm_change = ed / maxlen
    score = max(0.0, 100.0 * (1.0 - (norm_change / 0.30)))
    score = min(100.0, max(0.0, score))
    tokens = len(re.findall(r"\w+", orig))
    if tokens < 3:
        score = min(score, 50.0)
    return float(round(score, 2))

def content_coverage_score(student_text: str, reference_short: List[str]) -> float:
    st = student_text.lower()
    hits = 0
    for kp in reference_short:
        kp_l = kp.lower()
        if kp_l in st:
            hits += 1
        else:
            tokens = re.findall(r'\w+', st)
            best = max((lev_ratio(kp_l, t) for t in tokens), default=0.0)
            if best >= 0.8:
                hits += 1
    if not reference_short:
        return 100.0
    return float((hits / len(reference_short)) * 100.0)

def presentation_score(student_text: str) -> float:
    tokens = re.findall(r'\w+', student_text)
    n = len(tokens)
    if n == 0:
        return 0.0
    if n < 10:
        return 40.0
    if n < 30:
        return 70.0
    if n < 200:
        return 90.0
    return 100.0

def aggregate_scores(k, s, g, c, p):
    return 0.3 * k + 0.4 * s + 0.15 * g + 0.10 * c + 0.05 * p

# ---------------- LLM grader ----------------
LLM_PROMPT = """
You are an exam grader. Given a QUESTION, a MODEL ANSWER (short & long), and a STUDENT ANSWER, produce:
1) a JSON object with fields: {{keyword_pct, semantic_pct, grammar_pct, coverage_pct, presentation_pct, recommended_marks, explanation}}
2) The JSON should be valid and nothing else.

QUESTION:
{question}

REFERENCE_SHORT:
{ref_short}

REFERENCE_LONG:
{ref_long}

STUDENT_ANSWER:
{student}
"""

def llm_grade(question: str, ref_short: List[str], ref_long: str, student_text: str, max_marks: int = 10) -> Dict:
    def _truncate(s, chars=2000):
        if not s:
            return ""
        return s if len(s) <= chars else s[:chars - 3] + "..."
    try:
        prompt = LLM_PROMPT.format(
            question=_truncate(question, 800),
            ref_short=_truncate("\n".join(ref_short), 1200),
            ref_long=_truncate(ref_long, 1200),
            student=_truncate(student_text, 1500)
        )
    except Exception as e:
        print("Error formatting LLM prompt:", e)
        return {"keyword_pct": 0, "semantic_pct": 0, "grammar_pct": 0, "coverage_pct": 0, "presentation_pct": 0, "recommended_marks": 0, "explanation": f"LLM prompt formatting failed: {e}"}
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = llm.generate(**inputs, max_new_tokens=256)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    try:
        jstart = decoded.index("{")
        jend = decoded.rindex("}") + 1
        js = decoded[jstart:jend]
        data = json.loads(js)
        for k in ["keyword_pct", "semantic_pct", "grammar_pct", "coverage_pct", "presentation_pct", "recommended_marks"]:
            if k in data:
                try:
                    data[k] = float(data[k])
                except:
                    pass
        if "explanation" not in data:
            data["explanation"] = decoded
        return data
    except Exception:
        return {"keyword_pct": 0, "semantic_pct": 0, "grammar_pct": 0, "coverage_pct": 0, "presentation_pct": 0, "recommended_marks": 0, "explanation": f"Failed to parse LLM output as JSON. Raw output: {decoded}"}

# ---------------- High-level pipeline ----------------
"""def evaluate_from_image(image_path: str, question: str, reference: Dict, debug_save_image: bool = False) -> Dict:
    blocks = ocr_image_to_blocks(image_path, save_debug_image=debug_save_image)
    paras = coalesce_blocks(blocks)
    student_text = " ".join([p["text"] for p in paras])
    avg_ocr_conf = float(np.mean([p["conf"] for p in paras])) if paras else 0.0

    k = keyword_score(student_text, reference.get("keywords", []))
    s = semantic_score(student_text, [reference.get("reference_long", "")])
    g = grammar_score_model(student_text)
    c = content_coverage_score(student_text, reference.get("reference_short", []))
    p = presentation_score(student_text)
    final_pct = aggregate_scores(k, s, g, c, p)
    rec_marks = final_pct / 100.0 * reference.get("max_marks", 10)

    llm_out = llm_grade(question, reference.get("reference_short", []), reference.get("reference_long", ""), student_text, reference.get("max_marks", 10))
    grader_confidence = 1.0 - abs((llm_out.get("recommended_marks", rec_marks) - rec_marks) / max(1.0, reference.get("max_marks", 10)))
    semantic_conf = s / 100.0
    composite = 0.4 * avg_ocr_conf + 0.4 * semantic_conf + 0.2 * grader_confidence

    record = {
        "image_path": image_path,
        "student_text": student_text,
        "avg_ocr_conf": avg_ocr_conf,
        "component_scores": {"keyword_pct": k, "semantic_pct": s, "grammar_pct": g, "coverage_pct": c, "presentation_pct": p},
        "deterministic_final_pct": final_pct,
        "deterministic_recommended_marks": rec_marks,
        "llm_output": llm_out,
        "grader_confidence": grader_confidence,
        "composite_confidence": composite,
        "route": "teacher_review" if composite < 0.6 else ("auto_accept" if composite >= 0.85 else "partial_review")
    }
    return record"""

def evaluate_from_image(image_path: str, question: str, reference: Dict, debug_save_image: bool = False) -> Dict:
    """
    Runs OCR, coalesce text, generates component scores, runs LLM for explanation,
    computes composite confidence and returns a record for audit.

    Improvements in this version:
    - normalizes avg_ocr_conf to 0..1 before combining confidences
    - robust LLM fallback: if LLM output cannot be parsed or is invalid, use deterministic fallback
    - ensures composite confidence and routing are computed on normalized scales
    """
    # OCR -> blocks -> paragraphs
    blocks = ocr_image_to_blocks(image_path, save_debug_image=debug_save_image)
    blocks = merge_nearby_horizontal_blocks(blocks, gap_thresh=30)
    paras = coalesce_blocks(blocks)
    student_text = " ".join([p["text"] for p in paras])
    # avg_ocr_conf is originally in 0..100 from OCR; keep raw for logging but normalize later
    avg_ocr_conf = float(np.mean([p["conf"] for p in paras])) if paras else 0.0

    # Deterministic component scores
    k = keyword_score(student_text, reference.get("keywords", []))
    s = semantic_score(student_text, [reference.get("reference_long", "")])
    g = grammar_score_model(student_text)
    c = content_coverage_score(student_text, reference.get("reference_short", []))
    p = presentation_score(student_text)
    final_pct = aggregate_scores(k, s, g, c, p)
    rec_marks = final_pct / 100.0 * reference.get("max_marks", 10)

    # Ask LLM for explanation / suggested marks
    llm_out = llm_grade(
        question,
        reference.get("reference_short", []),
        reference.get("reference_long", ""),
        student_text,
        reference.get("max_marks", 10),
    )

    # Compute a grader_confidence proxy (agreement between deterministic and LLM marks)
    # If LLM returned something parsable and numeric:
    grader_confidence = None
    try:
        lm_rec = float(llm_out.get("recommended_marks")) if isinstance(llm_out, dict) and ("recommended_marks" in llm_out) else None
        if lm_rec is not None:
            grader_confidence = 1.0 - abs(lm_rec - rec_marks) / max(1.0, reference.get("max_marks", 10))
        else:
            grader_confidence = 0.0
    except Exception:
        grader_confidence = 0.0

    # If LLM parse failed or returned clearly invalid results, use deterministic fallback
    use_fallback = False
    if not isinstance(llm_out, dict):
        use_fallback = True
    else:
        expl = str(llm_out.get("explanation", "")).lower()
        # detect explicit failure messages or missing/zero recommended_marks (adjust logic as needed)
        if ("failed to parse" in expl) or ("failed" in expl and "parse" in expl) or llm_out.get("recommended_marks", None) in (None, 0):
            use_fallback = True

    if use_fallback:
        llm_out = {
            "keyword_pct": k,
            "semantic_pct": s,
            "grammar_pct": g,
            "coverage_pct": c,
            "presentation_pct": p,
            "recommended_marks": rec_marks,
            "explanation": "LLM parse failed or returned invalid output; using deterministic fallback."
        }
        # grader_confidence should reflect fallback (set to 1.0 because we are using the deterministic value)
        grader_confidence = 1.0

    # Normalize confidences to 0..1 ranges
    avg_ocr_conf_norm = max(0.0, min(1.0, avg_ocr_conf / 100.0))
    semantic_conf = max(0.0, min(1.0, s / 100.0))
    grader_confidence = max(0.0, min(1.0, grader_confidence if grader_confidence is not None else 0.0))

    # Composite confidence: weighted sum (tweak weights if needed)
    composite = 0.4 * avg_ocr_conf_norm + 0.4 * semantic_conf + 0.2 * grader_confidence

    # Build the audit record
    record = {
        "image_path": image_path,
        "student_text": student_text,
        "avg_ocr_conf_raw": avg_ocr_conf,           # keep raw OCR confidence (0..100) for debugging
        "avg_ocr_conf": avg_ocr_conf_norm,         # normalized 0..1
        "component_scores": {
            "keyword_pct": k,
            "semantic_pct": s,
            "grammar_pct": g,
            "coverage_pct": c,
            "presentation_pct": p
        },
        "deterministic_final_pct": final_pct,
        "deterministic_recommended_marks": rec_marks,
        "llm_output": llm_out,
        "grader_confidence": grader_confidence,
        "composite_confidence": composite,
        "route": "teacher_review" if composite < 0.6 else ("auto_accept" if composite >= 0.85 else "partial_review")
    }

    return record


# ---------------- Command-line test ----------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python icr_pipeline2.py <image_path>")
        sys.exit(1)
    img = sys.argv[1]
    reference = {
        "max_marks": 10,
        "keywords": [
            {"term": "Newton's first law", "weight": 1.0},
            {"term": "inertia", "weight": 1.0},
            {"term": "Newton's second law", "weight": 1.0},
            {"term": "F=ma", "weight": 1.0},
            {"term": "Newton's third law", "weight": 1.0}
        ],
        "reference_long": (
            "Newton's laws of motion: "
            "1) (Inertia) A body remains at rest or in uniform motion unless acted on by a net external force. "
            "2) (F=ma) The net force acting on a body equals mass times acceleration. "
            "3) (Action-Reaction) For every action there is an equal and opposite reaction."
        ),
        "reference_short": [
            "inertia",
            "F = m a",
            "action reaction",
            "net force = mass * acceleration"
        ]
    }
    res = evaluate_from_image(img, "Explain Newton's three laws of motion with examples.", reference, debug_save_image=True)
    print(json.dumps(res, indent=2))

