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
        print("[ModelManager] Loading EasyOCR...")
        start = time.time()
        reader = easyocr.Reader(['en'], gpu=(self.device == "cuda"))
        print(f"[ModelManager] EasyOCR loaded in {time.time()-start:.2f}s")
        return reader
    
    def unload_ocr(self, reader):
        del reader
        self.garbage_collect()
        print("[ModelManager] EasyOCR unloaded")

    def load_trocr(self, model_name=None):
        print("[ModelManager] Loading TrOCR...")
        start = time.time()
        # prefer local fine-tuned checkpoint
        # if model_name is None:
        #     local_dir = os.path.join(os.path.dirname(__file__), "..", "fine_tuned_trocr_small")
        #     local_dir = os.path.abspath(local_dir)
        #     if os.path.exists(local_dir):
        #         model_name = local_dir
        #     else:
        #         model_name = "microsoft/trocr-small-handwritten"
        if model_name is None:
            model_name = "microsoft/trocr-small-handwritten"
        
        try:
            model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
            processor = ViTImageProcessor.from_pretrained(model_name)
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            except:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
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
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False)
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    return text

def recognize_with_trocr_per_line(image_path: str, easyocr_blocks: List[Dict], trocr_bundle, pad: int = 6, num_beams: int = 8) -> Tuple[str, List[Tuple[str, float, tuple]]]:
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise FileNotFoundError(image_path)
    H, W = img_cv.shape[:2]

    # 1) Merge nearby horizontal pieces
    merged = merge_nearby_horizontal_blocks(easyocr_blocks, gap_thresh=30)

    # 2) Heuristics: remove tiny boxes
    filtered = []
    min_area = max(200, 0.002 * (W * H))
    top_margin = 0.03 * H
    bottom_margin = 0.97 * H
    for b in merged:
        x, y, w, h = b['bbox']
        area = w * h
        if area < min_area: continue
        if y < top_margin or (y + h) > bottom_margin:
            if not (w > 0.6 * W and h > 0.02 * H): continue
        filtered.append(b)

    filtered = sorted(filtered, key=lambda x: x['bbox'][1])

    # Preprocessing helpers
    def preprocess_crop_for_trocr(crop_bgr):
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
        except: pass
        gray = cv2.medianBlur(gray, 3)
        try:
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
        except:
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        return th

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
    
    # Manual TinyLlama template to ensure correct formatting
    # <|system|>...</s><|user|>...</s><|assistant|>
    
    def trunc(s, n): return s[:n] if s else ""
    
    sys_msg = "You are an exam grader. Grade the student answer based on the reference. Return ONLY a valid JSON object."
    user_msg = f"""Question: {trunc(question, 200)}
Key Points: {trunc(", ".join(ref_short), 250)}
Reference: {trunc(ref_long, 350)}
Student Answer: {trunc(student, 500)}

Return a JSON object with these exact keys:
{{
  "keyword_pct": <0-100>,
  "semantic_pct": <0-100>,
  "grammar_pct": <0-100>,
  "coverage_pct": <0-100>,
  "presentation_pct": <0-100>,
  "recommended_marks": <0-10>,
  "explanation": "<short feedback>"
}}"""

    prompt = f"<|system|>\n{sys_msg}</s>\n<|user|>\n{user_msg}</s>\n<|assistant|>\n"

    pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(DEVICE)
    
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False, temperature=0.1, pad_token_id=pad_token_id)
        
        # Decode only new tokens
        generated_ids = out[0][inputs['input_ids'].shape[1]:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"[LLM RAW] {decoded}") # Debug
        
        # Parse JSON
        jstart = decoded.find("{")
        jend = decoded.rfind("}") + 1
        if jstart >= 0 and jend > jstart:
            json_str = decoded[jstart:jend]
            try:
                return json.loads(json_str)
            except:
                import ast
                try:
                    return ast.literal_eval(json_str)
                except:
                    pass
    except Exception as e:
        print(f"LLM Error: {e}")
    return None

# ---------------- Pipeline Orchestrator ----------------

def evaluate_from_image(image_path: str, question: str, reference: Dict, debug_save_image: bool = False) -> Dict:
    print(f"\n--- Processing {os.path.basename(image_path)} ---")
    
    # 1. OCR Stage
    print("Stage 1: OCR")
    ocr_reader = MM.load_ocr()
    trocr_bundle = MM.load_trocr() # optional, depends if we want TrOCR
    
    blocks = ocr_image_to_blocks(image_path, ocr_reader, trocr_bundle, debug_save_image)
    
    MM.unload_trocr(trocr_bundle)
    MM.unload_ocr(ocr_reader)
    
    blocks = merge_nearby_horizontal_blocks(blocks)
    paras = coalesce_blocks(blocks)
    student_text = " ".join([p["text"] for p in paras])
    avg_ocr = np.mean([p.get("conf",0) for p in paras]) if paras else 0.0
    print(f"OCR Text ({len(student_text)} chars): {student_text[:50]}...")

    # 2. Semantic Stage
    print("Stage 2: Semantic Analysis")
    sent_model = MM.load_sentence_transformer()
    s_score = semantic_score(student_text, [reference.get("reference_long", "")], sent_model)
    MM.unload_sentence_transformer(sent_model)

    # 3. Grammar Stage
    print("Stage 3: Grammar Analysis")
    g_tok, g_mod = MM.load_grammar_model()
    g_score = grammar_score_model(student_text, g_tok, g_mod)
    MM.unload_grammar_model(g_tok, g_mod)

    # 4. Deterministic Scores (No model needed)
    k_score = keyword_score(student_text, reference.get("keywords", []))
    c_score = content_coverage_score(student_text, reference.get("reference_short", []))
    p_score = presentation_score(student_text)
    
    final_pct = 0.3*k_score + 0.4*s_score + 0.15*g_score + 0.10*c_score + 0.05*p_score
    rec_marks = final_pct / 100.0 * reference.get("max_marks", 10)

    # 5. LLM Stage
    print("Stage 4: LLM Grading")
    llm_tok, llm_mod = MM.load_llm()
    llm_out = llm_grade(
        question, 
        reference.get("reference_short", []), 
        reference.get("reference_long", ""), 
        student_text, 
        llm_tok, 
        llm_mod
    )
    MM.unload_llm(llm_tok, llm_mod)

    # Combine results
    if llm_out and isinstance(llm_out, dict) and "recommended_marks" in llm_out:
        l_marks = float(llm_out["recommended_marks"])
        conf = 1.0 - abs(l_marks - rec_marks)/10.0
    else:
        conf = 0.0
        llm_out = {"explanation": "LLM failed or fallback used", "raw_output": str(llm_out)}

    composite = 0.4*(avg_ocr/100.0) + 0.4*(s_score/100.0) + 0.2*conf
    
    res = {
        "image_path": image_path,
        "student_text": student_text,
        "avg_ocr_conf": avg_ocr,
        "component_scores": {
            "keyword_pct": k_score, "semantic_pct": s_score, 
            "grammar_pct": g_score, "coverage_pct": c_score, "presentation_pct": p_score
        },
        "deterministic_final_pct": final_pct,
        "deterministic_recommended_marks": rec_marks,
        "llm_output": llm_out,
        "composite_confidence": composite
    }
    return res

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
