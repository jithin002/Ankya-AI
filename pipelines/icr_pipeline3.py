# pipelines/icr_pipeline3.py (Refactored for Lazy Loading)
import os
import json
import re
import tempfile
import time
import gc
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import easyocr
import spacy
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import VisionEncoderDecoderModel, ViTImageProcessor
from Levenshtein import ratio as lev_ratio
import Levenshtein
import hashlib
import shelve
import cv2
import pytesseract
from pytesseract import Output
import traceback

# ----------------------- Config / Globals -----------------------

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# Spacy is lightweight enough to keep loaded, or could be lazy loaded too.
# For now, we keep it global as it's used frequently and low memory.
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spacy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ----------------------- Model Manager -----------------------

class ModelManager:
    def __init__(self):
        self.device = DEVICE
        self.grammar_cache = shelve.open("grammar_cache.db")
    
    def garbage_collect(self):
        """Force garbage collection and empty CUDA cache."""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
    
    def load_ocr(self):
        self.garbage_collect()  # Clear any residual VRAM before loading
        print("[ModelManager] Loading EasyOCR (GPU)...")
        start = time.time()
        reader = easyocr.Reader(['en'], gpu=(self.device == "cuda"))
        print(f"[ModelManager] EasyOCR loaded in {time.time()-start:.2f}s")
        return reader
    
    def unload_ocr(self, reader):
        del reader
        self.garbage_collect()
        print("[ModelManager] EasyOCR unloaded")

    def load_trocr(self, model_name=None):
        self.garbage_collect()  # Clear any residual VRAM before loading
        print("[ModelManager] Loading TrOCR...")
        start = time.time()
        base_model_name = "microsoft/trocr-small-handwritten"

        try:
            model = VisionEncoderDecoderModel.from_pretrained(base_model_name).to(self.device)
            processor = ViTImageProcessor.from_pretrained(base_model_name)
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
            except:
                tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
            
            # Load LoRA adapter if it exists (either in root or latest checkpoint)
            local_dir = os.path.join(os.path.dirname(__file__), "..", "fine_tuned_trocr_small")
            local_dir = os.path.abspath(local_dir)
            
            adapter_path = None
            if os.path.exists(os.path.join(local_dir, "adapter_config.json")):
                adapter_path = local_dir
            elif os.path.exists(local_dir):
                # Check for checkpoint directories
                checkpoints = [os.path.join(local_dir, d) for d in os.listdir(local_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    # Sort by checkpoint number
                    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                    latest_chkpt = checkpoints[-1]
                    if os.path.exists(os.path.join(latest_chkpt, "adapter_config.json")):
                        adapter_path = latest_chkpt

            if adapter_path:
                print(f"[ModelManager] Found LoRA adapter at {adapter_path}. Loading on top of base model...")
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter_path)
                # Merge LoRA weights into base model — required so generate() works with positional args
                model = model.merge_and_unload()
                print("[ModelManager] LoRA adapter merged successfully.")
            else:
                print("[ModelManager] No local LoRA adapter found. Using base model.")

            # Cast to float16 (half-precision) to halve VRAM usage without accuracy loss
            if self.device == "cuda":
                model = model.half()
            print(f"[ModelManager] TrOCR loaded in {time.time()-start:.2f}s")
            return {"model": model, "processor": processor, "tokenizer": tokenizer}
        except Exception as e:
            print(f"[ModelManager] TrOCR load failed: {e}")
            return None

    def unload_trocr(self, trocr_dict):
        if trocr_dict:
            del trocr_dict["model"]
            del trocr_dict["processor"]
            del trocr_dict["tokenizer"]
            del trocr_dict
        self.garbage_collect()
        print("[ModelManager] TrOCR unloaded")

    def load_sentence_transformer(self):
        self.garbage_collect()  # Clear any residual VRAM before loading
        print("[ModelManager] Loading SentenceTransformer...")
        start = time.time()
        model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        print(f"[ModelManager] SentenceTransformer loaded in {time.time()-start:.2f}s")
        return model

    def unload_sentence_transformer(self, model):
        del model
        self.garbage_collect()
        print("[ModelManager] SentenceTransformer unloaded")

    def load_grammar_model(self):
        self.garbage_collect()  # Clear any residual VRAM before loading
        print("[ModelManager] Loading Grammar Model...")
        start = time.time()
        name = "prithivida/grammar_error_correcter_v1"
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name).to(self.device)
        print(f"[ModelManager] Grammar Model loaded in {time.time()-start:.2f}s")
        return tokenizer, model

    def unload_grammar_model(self, tokenizer, model):
        del tokenizer
        del model
        self.garbage_collect()
        print("[ModelManager] Grammar Model unloaded")

    def load_llm(self):
        self.garbage_collect()  # Clear any residual VRAM before loading
        print("[ModelManager] Loading LLM (TinyLlama)...")
        start = time.time()
        name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(
                name, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            print(f"[ModelManager] LLM loaded in {time.time()-start:.2f}s")
            return tokenizer, model
        except Exception as e:
            print(f"[ModelManager] LLM load failed: {e}")
            return None, None

    def unload_llm(self, tokenizer, model):
        if tokenizer: del tokenizer
        if model: del model
        self.garbage_collect()
        print("[ModelManager] LLM unloaded")
        
    def close(self):
        self.grammar_cache.close()

# Singleton instance
MM = ModelManager()

# ----------------------- OCR Functions -----------------------

def trocr_recognize_pil(pil_img: Image.Image, trocr_bundle, max_new_tokens: int = 256, num_beams: int = 4) -> str:
    if not trocr_bundle:
        return ""
    model = trocr_bundle["model"]
    processor = trocr_bundle["processor"]
    tokenizer = trocr_bundle["tokenizer"]
    
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(DEVICE)
    # Cast to float16 if on CUDA to match the model's dtype
    if DEVICE == "cuda":
        pixel_values = pixel_values.half()
    with torch.inference_mode():
        generated_ids = model.generate(
            pixel_values, 
            max_new_tokens=max_new_tokens, 
            num_beams=num_beams, 
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    return text

def recognize_with_trocr_per_line(image_path: str, easyocr_blocks: List[Dict], trocr_bundle, pad: int = 6, num_beams: int = 8) -> Tuple[str, List[Tuple[str, float, tuple]]]:
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise FileNotFoundError(image_path)
    H, W = img_cv.shape[:2]

    # 1) Merge nearby horizontal pieces
    merged = merge_nearby_horizontal_blocks(easyocr_blocks, gap_thresh=30)

    # 2) Heuristics: remove tiny boxes (relaxed – handwriting near margins is valid)
    filtered = []
    min_area = max(80, 0.0005 * (W * H))  # was 200/0.002 – relaxed to catch small circled numbers
    for b in merged:
        x, y, w, h = b['bbox']
        area = w * h
        if area < min_area: continue
        filtered.append(b)

    filtered = sorted(filtered, key=lambda x: x['bbox'][1])

    # Preprocessing helpers
    def preprocess_crop_for_trocr(crop_bgr):
        # Preprocessing: Convert to grayscale and apply CLAHE to boost contrast.
        # Avoid harsh binarization and morphological ops that destroy thin handwriting.
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
        except: pass
        # Return as BGR since TrOCR processor converts it back to Pil/RGB anyway
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    results = []
    for b in filtered:
        x, y, w, h = b['bbox']
        x0 = int(max(0, x - pad)); y0 = int(max(0, y - pad))
        x1 = int(min(W, x + w + pad)); y1 = int(min(H, y + h + pad))
        crop = img_cv[y0:y1, x0:x1]
        if crop.size == 0: continue
        
        proc = preprocess_crop_for_trocr(crop)
        pil = Image.fromarray(proc).convert("L").convert("RGB")
        try:
            txt = trocr_recognize_pil(pil, trocr_bundle, max_new_tokens=256, num_beams=num_beams)
            if not txt or len(txt.strip()) == 0:
                txt = b.get("text", "")
        except Exception as e:
            print("TrOCR error:", e)
            txt = b.get("text", "")
        results.append((txt, float(b.get("conf", 0.0)), (x0, y0, x1-x0, y1-y0)))

    joined = "\n".join([r[0] for r in results])
    return joined, results

# ---------------- Utilities & Preprocessing ----------------

def normalize_text(text: str) -> str:
    text = text.replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_ruled_lines(img_bgr) -> object:
    """Use OpenCV morphological transforms to erase horizontal ruled lines from notebook paper.
    Returns a clean BGR image with only the handwriting remaining."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Use adaptive threshold instead of OTSU to handle uneven lighting/shadows
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    # Detect horizontal lines using a wide, flat kernel
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(binary.shape[1] // 8), 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    # Remove them from the original binary image
    cleaned = cv2.subtract(binary, horizontal_lines)
    # Restore handwriting on white background
    result = cv2.bitwise_not(cleaned)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def clean_ocr_noise(text: str) -> str:
    """Strip out patterns that come from ruled lines being misread by OCR."""
    import re as _re
    text = _re.sub(r'([^\w\s]){3,}', ' ', text)
    text = _re.sub(r'([.,\'"`_^=-]\s+){2,}', ' ', text)
    text = _re.sub(r'\s[^\w\s]{1,2}\s', ' ', text)
    text = _re.sub(r'(?<=\s)[\'"`_]+|[\'"`_]+(?=\s)', ' ', text)
    text = _re.sub(r'\s+', ' ', text).strip()
    return text


# ── Encircled digit normaliser ────────────────────────────────────────────────
# OCR engines often produce ① ② ③ (Unicode Enclosed Alphanumerics) for circled
# handwritten question numbers.  Map them to "1) " so the question-splitter
# regex can detect segment boundaries reliably.
_ENCIRCLED_DIGITS = {chr(9312 + i): str(i + 1) for i in range(20)}  # ①→1 … ⑳→20

def normalize_encircled_digits(text: str) -> str:
    """Replace circled-digit unicode chars with plain-digit equivalents."""
    for ch, digit in _ENCIRCLED_DIGITS.items():
        text = text.replace(ch, f' {digit}) ')
    return text


def preprocess_for_ocr(img_bgr):
    """Full preprocessing chain: CLAHE contrast boost + ruled-line removal.
    Returns a clean, high-contrast BGR image ready for EasyOCR."""
    import tempfile
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Step 1: CLAHE Adaptive Histogram Equalization to boost faded ink contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Step 2: Convert back to BGR color for ruled-line removal
    img_clahe = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Step 3: Morphological ruled-line removal
    img_clean = remove_ruled_lines(img_clahe)
    return img_clean


DEFAULT_Q_PATTERN = re.compile(
    r'(?:^|\s)(?:(\d{1,2})([ab]?)\)|(?:\()(\d{1,2})[ab]?(?:\)))',
    re.IGNORECASE | re.MULTILINE
)

def split_text_by_question(blocks: list) -> list:
    """Given OCR blocks (each with a 'text' and 'y' centroid), split into segments
    whenever a question number pattern is detected. Returns a list of dicts:
      { 'q_label': '16', 'text': '...full answer text...', 'y_pct': 0.35 }
    If no question numbers detected, returns one segment covering the whole page.

    Recognised patterns (start-of-text or after whitespace):
      classic:   1)  (1)  [1]
      dotted:    1.  Q1.  Q1:
      written:   Q1  Q1:  question 1
      encircled: ①  ②  (already normalised to '1) ' by normalize_encircled_digits)
    """
    # Expanded pattern – covers: 1) (1) [1] 1. Q1 Q1. Q1: question1
    Q_PAT = re.compile(
        r'(?:(?:^|(?<=\s))'
        r'(?:'
        r'(?:Q|q|question\s*)(\d{1,3})([ab]?)[:\.]?'   # Q1  Q1. Q1:  question1
        r'|(?:\()(\d{1,3})([ab]?)[\)\]]'               # (1)  [1]
        r'|(\d{1,3})([ab]?)[\)\]]'                      # 1)  1]
        r'|(\d{1,3})([ab]?)\.(?=\s)'                    # 1.  (dot followed by space)
        r'))'
    )

    segments = []
    current_label = None
    current_lines = []
    current_y = 0.0
    first_block_y = None
    last_block_y = 1.0

    if blocks:
        ys = [b.get('y', 0) for b in blocks]
        first_block_y = min(ys) if ys else 0
        last_block_y = max(ys) if ys else 1

    for block in blocks:
        raw_text = block.get('text', '').strip()
        # Normalise encircled digits BEFORE regex matching
        text = normalize_encircled_digits(raw_text)
        y = block.get('y', 0)
        y_pct = (y - first_block_y) / (last_block_y - first_block_y + 1) if blocks else 0

        match = Q_PAT.search(text)
        if match:
            # Save previous segment
            if current_lines:
                segments.append({
                    'q_label': current_label or 'full',
                    'text': clean_ocr_noise(' '.join(current_lines)),
                    'y_pct': current_y
                })
            # Extract whichever capture group matched
            q_num = next((g for g in match.groups()[0::2] if g is not None), None)
            q_sub = next((g for g in match.groups()[1::2] if g is not None), '') or ''
            current_label = f"{q_num}{q_sub}" if q_num else 'full'
            current_y = y_pct
            rest = text[match.end():].strip()
            current_lines = [rest] if rest else []
        else:
            current_lines.append(text)

    # Flush last segment
    if current_lines:
        segments.append({
            'q_label': current_label or 'full',
            'text': clean_ocr_noise(' '.join(current_lines)),
            'y_pct': current_y
        })

    # If nothing was detected, return single segment with all text
    if not segments:
        all_text = clean_ocr_noise(' '.join(
            normalize_encircled_digits(b.get('text', '')) for b in blocks
        ))
        return [{'q_label': 'full', 'text': all_text, 'y_pct': 0.0}]

    return segments

def deskew_image(img_gray):
    try:
        edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180.0, 200)
        if lines is None: return img_gray, 0.0
        angles = []
        for rho_theta in lines:
            _, theta = rho_theta[0]
            angle_deg = (theta * 180.0 / np.pi) - 90.0
            if angle_deg > 90: angle_deg -= 180
            if angle_deg < -90: angle_deg += 180
            angles.append(angle_deg)
        if not angles: return img_gray, 0.0
        median_angle = float(np.median(angles))
        (h, w) = img_gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated, median_angle
    except:
        return img_gray, 0.0

def preprocess_image_for_ocr(image_path: str):
    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    deskewed, _ = deskew_image(denoised)
    try:
        thresh = cv2.adaptiveThreshold(deskewed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
    except:
        _, thresh = cv2.threshold(deskewed, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

# ---------------- Ensemble OCR Helpers ----------------

def pytesseract_lines_from_image(img_bgr):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    data = pytesseract.image_to_data(pil, output_type=Output.DICT, lang='eng')
    n_boxes = len(data['level'])
    lines = {}
    for i in range(n_boxes):
        text = str(data['text'][i]).strip()
        if not text: continue
        conf = float(data['conf'][i]) if float(data['conf'][i]) >= 0 else 0.0
        key = (data.get('block_num',[0]*n_boxes)[i], data.get('par_num',[0]*n_boxes)[i], data.get('line_num',[0]*n_boxes)[i])
        if key not in lines: lines[key] = {"words": [], "confs": [], "bboxes": []}
        lines[key]["words"].append(text); lines[key]["confs"].append(conf)
        lines[key]["bboxes"].append((data['left'][i], data['top'][i], data['width'][i], data['height'][i]))
    
    out = []
    for k, v in lines.items():
        text = " ".join(v["words"])
        xs, ys, ws, hs = zip(*v["bboxes"])
        bbox = (min(xs), min(ys), max(x+w for x,w in zip(xs,ws))-min(xs), max(y+h for y,h in zip(ys,hs))-min(ys))
        out.append({"text": text, "conf": float(np.mean(v["confs"])), "bbox": bbox})
    return out

def easyocr_blocks_from_image(img_bgr, ocr_reader):
    easy_res = ocr_reader.readtext(img_bgr, detail=1)
    out = []
    for bbox, text, conf in easy_res:
        xs, ys = zip(*bbox)
        x, y = min(xs), min(ys)
        w, h = max(xs)-x, max(ys)-y
        out.append({"text": text, "conf": float(conf)*100.0, "bbox": (x,y,w,h), "cy": y+h/2.0})
    return out

def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2]); yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB-xA) * max(0, yB-yA)
    union = (boxA[2]*boxA[3]) + (boxB[2]*boxB[3]) - interArea
    return interArea / union if union > 0 else 0.0

def ensemble_ocr_preprocessed(img_bgr, ocr_reader):
    pyt_lines = pytesseract_lines_from_image(img_bgr)
    easy_blocks = easyocr_blocks_from_image(img_bgr, ocr_reader)
    
    merged = []
    used_pyt = set()
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
            for pi, _ in overlaps: used_pyt.add(pi)
            merged.append(res)
        else:
            merged.append(e)
    
    for pi, p in enumerate(pyt_lines):
        if pi not in used_pyt:
            merged.append({"text": p['text'], "conf": p['conf'], "bbox": p['bbox'], "cy": p['bbox'][1]+p['bbox'][3]/2})
            
    return sorted(merged, key=lambda x: x['cy'])

# ---------------- Main OCR Entrypoint ----------------

def ocr_image_to_blocks(image_path: str, ocr_reader, trocr_bundle=None, save_debug_image: bool = False) -> List[Dict]:
    img_orig = cv2.imread(image_path)
    if img_orig is None: raise FileNotFoundError(image_path)

    try:
        # 1. Layout with EasyOCR
        easy_layout = easyocr_blocks_from_image(img_orig, ocr_reader)
        
        # 2. Try TrOCR if available
        if trocr_bundle:
            try:
                joined_text, per_line = recognize_with_trocr_per_line(image_path, easy_layout, trocr_bundle)
                blocks = [{"text": normalize_text(t), "conf": float(c), "bbox": b, "cy": b[1]+b[3]/2} for t, c, b in per_line]
                blocks = [b for b in blocks if b["text"].strip()]
                return sorted(blocks, key=lambda x: x['cy'])
            except Exception as e:
                print("TrOCR failed, fallback to ensemble:", e)
    except Exception as e:
        print("EasyOCR layout failed:", e)

    # 3. Fallback to Ensemble
    processed_bgr = preprocess_image_for_ocr(image_path)
    blocks = ensemble_ocr_preprocessed(processed_bgr, ocr_reader)
    for b in blocks: b['text'] = normalize_text(b['text'])
    return [b for b in blocks if b['text'].strip()]

# ---------------- Coalesce Blocks ----------------

def merge_nearby_horizontal_blocks(blocks, gap_thresh=20):
    if not blocks: return []
    blocks = sorted(blocks, key=lambda b: (b['cy'], b['bbox'][0]))
    merged = []
    cur = blocks[0].copy()
    for b in blocks[1:]:
        if abs(b['cy'] - cur['cy']) <= max(12, 0.4*cur['bbox'][3]):
            cur_r = cur['bbox'][0] + cur['bbox'][2]
            if b['bbox'][0] - cur_r <= gap_thresh:
                # Merge
                nl = min(cur['bbox'][0], b['bbox'][0])
                nr = max(cur_r, b['bbox'][0]+b['bbox'][2])
                nt = min(cur['bbox'][1], b['bbox'][1])
                nb = max(cur['bbox'][1]+cur['bbox'][3], b['bbox'][1]+b['bbox'][3])
                cur['text'] += " " + b['text']
                cur['conf'] = (cur['conf'] + b['conf'])/2
                cur['bbox'] = (nl, nt, nr-nl, nb-nt)
                cur['cy'] = (cur['cy'] + b['cy'])/2
            else:
                merged.append(cur); cur = b.copy()
        else:
            merged.append(cur); cur = b.copy()
    merged.append(cur)
    return merged

def coalesce_blocks(blocks, y_tol=12):
    if not blocks: return []
    groups = []
    cur_group = [blocks[0]]
    for b in blocks[1:]:
        if abs(b["cy"] - cur_group[-1]["cy"]) <= y_tol:
            cur_group.append(b)
        else:
            groups.append(cur_group); cur_group = [b]
    groups.append(cur_group)
    
    out = []
    for g in groups:
        text = " ".join([x["text"] for x in g]).strip()
        conf = float(np.mean([x["conf"] for x in g]))
        xs, ys, ws, hs = zip(*[x['bbox'] for x in g])
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(x+w for x,w in zip(xs,ws)), max(y+h for y,h in zip(ys,hs))
        out.append({"text": text, "conf": conf, "bbox": (x0, y0, x1-x0, y1-y0), "cy": np.mean([x['cy'] for x in g])})
    return sorted(out, key=lambda x: x['cy'])

# ---------------- Scoring ----------------

def keyword_score(student_text: str, keywords: List[Dict]) -> float:
    st = student_text.lower()
    total = sum(k.get('weight', 1.0) for k in keywords) or 1.0
    matched = 0.0
    for kw in keywords:
        term = kw['term'].lower()
        if term in st:
            matched += kw.get('weight', 1.0)
        else:
            # fuzzy
            tokens = re.findall(r'\w+', st)
            if any(lev_ratio(term, t) >= 0.8 for t in tokens):
                matched += kw.get('weight', 1.0)
    return min(100.0, (matched / total) * 100.0)

def semantic_score(student_text: str, reference_texts: List[str], model) -> float:
    if not model or not student_text.strip(): return 0.0
    s_emb = model.encode([student_text], convert_to_numpy=True)
    r_embs = model.encode(reference_texts, convert_to_numpy=True)
    r_avg = np.mean(r_embs, axis=0, keepdims=True)
    sim = np.dot(s_emb, r_avg.T) / (np.linalg.norm(s_emb, axis=1, keepdims=True) * np.linalg.norm(r_avg, axis=1))
    return float(((sim[0][0] + 1) / 2) * 100.0)

def grammar_score_model(student_text: str, tokenizer, model) -> float:
    if not student_text.strip() or not tokenizer or not model: return 0.0
    text = student_text[:2000] # truncate
    
    # Check cache
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if key in MM.grammar_cache:
        corrected = MM.grammar_cache[key]
    else:
        try:
            inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            with torch.no_grad():
                out = model.generate(inputs, max_new_tokens=256, do_sample=False)
            corrected = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            MM.grammar_cache[key] = corrected
            MM.grammar_cache.sync()
        except:
            return 70.0 # fallback

    ed = Levenshtein.distance(text.strip(), corrected)
    norm = ed / max(1, len(text))
    score = max(0.0, 100.0 * (1.0 - (norm / 0.30)))
    if len(text.split()) < 3: score = min(score, 50.0)
    return min(100.0, score)

def content_coverage_score(student_text: str, reference_short: List[str]) -> float:
    st = student_text.lower()
    hits = 0
    for kp in reference_short:
        kpl = kp.lower()
        if kpl in st: hits += 1
        else:
            tokens = re.findall(r'\w+', st)
            if any(lev_ratio(kpl, t) >= 0.8 for t in tokens): hits += 1
    return (hits / len(reference_short) * 100.0) if reference_short else 100.0

def presentation_score(student_text: str) -> float:
    n = len(re.findall(r'\w+', student_text))
    if n == 0: return 0.0
    if n < 10: return 40.0
    if n < 30: return 70.0
    if n < 200: return 90.0
    return 100.0

# ---------------- LLM Grading ----------------

def llm_grade(question, ref_short, ref_long, student, tokenizer, model) -> Dict:
    if not tokenizer or not model: return None
    
    # -------------------------------------------------------------------------
    # TODO: FUTURE IMPLEMENTATION - LLM SCORING
    # Currently, we only use the LLM to generate qualitative feedback text.
    # In the future, we should instruct the LLM to also output a score (0-10)
    # or specific component scores (grammar, keywords, etc.) in a structured format 
    # (like JSON) once we have a model capable of reliable structured output 
    # or improve the prompting strategy for this small model.
    # -------------------------------------------------------------------------

    def trunc(s, n): return s[:n] if s else ""
    
    # STRICT Prompt engineering to prevent "coding mode"
    sys_msg = (
        "You are a Teacher's Assistant grading an exam. "
        "Your goal is to provide constructive feedback to the student based on the reference answer. "
        "Do NOT write any code (Python, etc.). "
        "Do NOT talk about programming. "
        "Focus ONLY on the subject matter (Physics, History, etc.)."
    )
    
    user_msg = f"""
[Question]
{trunc(question, 200)}

[Reference Answer]
{trunc(ref_long, 350)}

[Key Points to Look For]
{trunc(", ".join(ref_short), 250)}

[Student's Answer]
{trunc(student, 500)}

[Instruction]
Write a short feedback comment (2-3 sentences) to the student.
- If the answer is correct, praise them and mention what they got right.
- If the answer is incorrect or incomplete, explain clearly what is missing based on the Reference Answer.
- Speak directly to the student ("You mentioned...", "You missed...").
- DO NOT WRITE CODE.
"""

    # TinyLlama format
    prompt = f"<|system|>\n{sys_msg}</s>\n<|user|>\n{user_msg}</s>\n<|assistant|>\n"

    pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(DEVICE)
    
    try:
        with torch.no_grad():
            # slightly lower temp for more focused output
            out = model.generate(
                **inputs, 
                max_new_tokens=250, # Shorter to prevent rambling
                do_sample=False, 
                temperature=0.2, 
                repetition_penalty=1.2, # Reduce loops
                pad_token_id=pad_token_id
            )
        
        # Decode only new tokens
        generated_ids = out[0][inputs['input_ids'].shape[1]:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"[LLM FEEDBACK] {decoded}") # Debug
        
        # Clean up if model still tries to chat
        if "Student Answer:" in decoded:
            decoded = decoded.split("Student Answer:")[-1].strip()
        
        return {
            "explanation": decoded,
            "recommended_marks": 0.0 
        }
            
    except Exception as e:
        print(f"LLM Error: {e}")
        traceback.print_exc()
    return None

# ---------------- Pipeline Orchestrator ----------------

def evaluate_from_image(image_path: str, question: str, reference: Dict, debug_save_image: bool = False,
                        qa_dataset: list = None, manual_q_overrides: dict = None) -> list:
    """Evaluate student answers from an image. Returns a LIST of result dicts,
    one per detected question-answer on the page. Uses the full qa_dataset when available
    so it can automatically pick the correct rubric for each detected answer."""
    print(f"\n--- Processing {os.path.basename(image_path)} ---")
    
    # 0. Pre-processing: CLAHE contrast boost + ruled-line removal
    raw_img = cv2.imread(image_path)
    if raw_img is not None:
        cleaned_img = preprocess_for_ocr(raw_img)
        import tempfile
        tmp_cleaned = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(tmp_cleaned.name, cleaned_img)
        tmp_cleaned.close()
        ocr_source_path = tmp_cleaned.name
    else:
        ocr_source_path = image_path
    
    # 1. OCR Stage
    print("Stage 1: OCR")
    ocr_reader = MM.load_ocr()
    trocr_bundle = MM.load_trocr()
    
    blocks = ocr_image_to_blocks(ocr_source_path, ocr_reader, trocr_bundle, debug_save_image)
    
    MM.unload_trocr(trocr_bundle)
    MM.unload_ocr(ocr_reader)
    if ocr_source_path != image_path:
        try: os.unlink(ocr_source_path)
        except: pass
    
    blocks = merge_nearby_horizontal_blocks(blocks)
    paras = coalesce_blocks(blocks)

    full_student_text = clean_ocr_noise(" ".join([p["text"] for p in paras]))
    avg_ocr = np.mean([p.get("conf", 0) for p in paras]) if paras else 0.0
    print(f"OCR Text ({len(full_student_text)} chars): {full_student_text[:100]}...")

    # ── Multi-Answer Splitting ──────────────────────────────────────────────
    segments = split_text_by_question(paras)
    print(f"[MultiAnswer] {len(segments)} segment(s) detected")

    # ── Auto-Rubric via SentenceTransformer for ALL segments ───────────────
    matched_rubrics = {}
    if qa_dataset:
        import gc as _gc
        from sentence_transformers import SentenceTransformer as _ST, util as _stu
        _gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Apply manual overrides first (no semantic search needed for these)
        override_map = manual_q_overrides or {}
        segs_needing_rag = []
        for seg in segments:
            q_label = seg["q_label"]
            if q_label in override_map:
                forced_num = override_map[q_label]
                forced_item = next((it for it in qa_dataset if it["number"] == forced_num), None)
                if forced_item:
                    seg["auto_matched_q"] = forced_item["number"]
                    seg["rag_confidence"] = 1.0  # teacher override = perfect confidence
                    matched_rubrics[q_label] = {
                        "question": forced_item["question"],
                        "reference": forced_item["rubric"]
                    }
                    print(f"[MANUAL] '{q_label}' -> Q{forced_item['number']} (teacher override)")
                    continue  # skip RAG for this segment
            segs_needing_rag.append(seg)

        if segs_needing_rag:
            sent_model = MM.load_sentence_transformer()
            for seg in segs_needing_rag:
                st = seg["text"]
                if not st.strip():
                    continue
                se = sent_model.encode(st, convert_to_tensor=True)
                best, best_item = -1.0, None
                for item in qa_dataset:
                    combined = f"{item.get('question','')} {item.get('answer','')}"
                    re_emb = sent_model.encode(combined, convert_to_tensor=True)
                    from sentence_transformers import util as _stu
                    sc = float(_stu.cos_sim(se, re_emb)[0][0])
                    if sc > best:
                        best = sc
                        best_item = item
                seg["auto_matched_q"] = best_item["number"] if (best_item and best > 0.10) else None
                seg["rag_confidence"] = best
                if best_item and best > 0.10:
                    matched_rubrics[seg["q_label"]] = {
                        "question": best_item["question"],
                        "reference": best_item["rubric"]
                    }
                    print(f"[RAG] '{seg['q_label']}' -> Q{best_item['number']} ({best:.3f})")
            MM.unload_sentence_transformer(sent_model)
        _gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── Grade Each Segment Sequentially ────────────────────────────────────
    all_results = []
    for seg in segments:
        seg_text = seg["text"]
        if not seg_text.strip():
            continue
        q_label = seg["q_label"]
        y_pct = seg["y_pct"]

        if q_label in matched_rubrics:
            seg_question = matched_rubrics[q_label]["question"]
            seg_reference = matched_rubrics[q_label]["reference"]
        else:
            seg_question = question
            seg_reference = reference

        # Semantic
        sent_model = MM.load_sentence_transformer()
        s_score = semantic_score(seg_text, [seg_reference.get("reference_long", "")], sent_model)
        MM.unload_sentence_transformer(sent_model)

        # Grammar
        g_tok, g_mod = MM.load_grammar_model()
        g_score = grammar_score_model(seg_text, g_tok, g_mod)
        MM.unload_grammar_model(g_tok, g_mod)

        # Deterministic
        k_score = keyword_score(seg_text, seg_reference.get("keywords", []))
        c_score = content_coverage_score(seg_text, seg_reference.get("reference_short", []))
        p_score = presentation_score(seg_text)
        max_m = seg_reference.get("max_marks", reference.get("max_marks", 10))
        final_pct = 0.3 * k_score + 0.4 * s_score + 0.15 * g_score + 0.10 * c_score + 0.05 * p_score
        rec_marks = final_pct / 100.0 * max_m

        # LLM
        llm_tok, llm_mod = MM.load_llm()
        llm_out = llm_grade(seg_question, seg_reference.get("reference_short", []),
                            seg_reference.get("reference_long", ""), seg_text, llm_tok, llm_mod)
        MM.unload_llm(llm_tok, llm_mod)

        if llm_out and isinstance(llm_out, dict) and "recommended_marks" in llm_out:
            l_marks = float(llm_out.get("recommended_marks", 0))
            conf = max(0.0, 1.0 - abs(l_marks - rec_marks) / max(max_m, 1))
        else:
            conf = 0.0
            llm_out = {"explanation": "LLM failed.", "recommended_marks": 0}

        composite = 0.4 * (avg_ocr / 100.0) + 0.4 * (s_score / 100.0) + 0.2 * conf

        all_results.append({
            "q_label": q_label,
            "y_pct": y_pct,
            "auto_matched_question": seg.get("auto_matched_q"),
            "rag_confidence": seg.get("rag_confidence"),
            "student_text": seg_text,
            "full_page_text": full_student_text,
            "avg_ocr_conf": avg_ocr,
            "component_scores": {
                "keyword_pct": k_score, "semantic_pct": s_score,
                "grammar_pct": g_score, "coverage_pct": c_score, "presentation_pct": p_score
            },
            "deterministic_final_pct": final_pct,
            "deterministic_recommended_marks": rec_marks,
            "llm_output": llm_out,
            "composite_confidence": composite,
            "image_path": image_path,
            "matched_question_text": seg_question,
            "rubric": seg_reference,
        })

    if not all_results:
        all_results.append({
            "q_label": "full", "y_pct": 0.0, "auto_matched_question": None,
            "student_text": full_student_text, "full_page_text": full_student_text,
            "avg_ocr_conf": avg_ocr, "component_scores": {},
            "deterministic_recommended_marks": 0,
            "llm_output": {"explanation": "No text found on page.", "recommended_marks": 0},
            "composite_confidence": 0, "image_path": image_path,
            "matched_question_text": "", "rubric": reference,
        })

    return all_results

if __name__ == "__main__":

    # Test
    examples_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
    img_path = os.path.join(examples_dir, "sample4.jpg")
    
    # Create dummy image if not exists, just to test pipeline logic flow
    if not os.path.exists(img_path):
        print(f"Creating dummy image at {img_path}")
        os.makedirs(examples_dir, exist_ok=True)
        img = Image.new('RGB', (800, 600), color = (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((10,10), "Newton's first law states that an object at rest stays at rest.", fill=(0,0,0))
        img.save(img_path)

    ref = {
        "max_marks": 10,
        "keywords": [{"term": "Newton", "weight": 1.0}],
        "reference_long": "Newton's first law of motion.",
        "reference_short": ["inertia"]
    }
    
    print("Running pipeline implementation test...")
    res = evaluate_from_image(img_path, "Explain Newton's law", ref)
    print(json.dumps(res, indent=2))
