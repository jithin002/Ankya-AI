# pipelines/icr_pipeline.py
import os
import json
import re
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np
import easyocr
import spacy
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import language_tool_python
from Levenshtein import ratio as lev_ratio
import torch
import Levenshtein
import hashlib
import shelve
import time


nlp = spacy.load("en_core_web_sm")
"""ft = language_tool_python.LanguageToolPublicAPI('en-US')"""

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Initialize OCR (use GPU if available)
OCR_READER = easyocr.Reader(['en'], gpu=(device=="cuda"))

# Sentence embeddings model
SENT_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# LLM grader: flan-t5-small (or flan-t5-base if you prefer)
LLM_MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME).to(device)

########## UTILITIES ##########

def normalize_text(text: str) -> str:
    text = text.replace('\r',' ').replace('\n',' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def ocr_image_to_blocks(image_path: str) -> List[Dict]:
    """
    Uses EasyOCR to extract text lines with bounding boxes.
    Returns list of dicts: {text, bbox, confidence, center_y}
    """
    res = OCR_READER.readtext(image_path, detail=1)  # detail gives bbox & conf
    blocks = []
    for bbox, text, conf in res:
        # bbox is list of 4 points; compute box center y
        ys = [p[1] for p in bbox]
        cy = sum(ys)/len(ys)
        blocks.append({
            "text": text,
            "bbox": bbox,
            "conf": conf,
            "cy": cy
        })
    # sort top->bottom
    blocks = sorted(blocks, key=lambda x: x["cy"])
    return blocks

def coalesce_blocks(blocks: List[Dict], y_tol: int = 12) -> List[Dict]:
    """
    Group nearby lines into paragraph blocks based on y coordinate tolerance.
    """
    if not blocks:
        return []
    groups = []
    cur_group = [blocks[0]]
    for b in blocks[1:]:
        if abs(b["cy"] - cur_group[-1]["cy"]) <= y_tol:
            cur_group.append(b)
        else:
            groups.append(cur_group)
            cur_group = [b]
    groups.append(cur_group)
    # join text and average conf
    out = []
    for g in groups:
        text = " ".join([x["text"] for x in g])
        conf = float(np.mean([x["conf"] for x in g]))
        bbox = [min(min(p[0] for p in x["bbox"]) for x in g),
                min(min(p[1] for p in x["bbox"]) for x in g),
                max(max(p[0] for p in x["bbox"]) for x in g),
                max(max(p[1] for p in x["bbox"]) for x in g)]
        cy = np.mean([x["cy"] for x in g])
        out.append({"text": text, "conf": conf, "bbox": bbox, "cy": cy})
    return out

########## SCORING ##########

def keyword_score(student_text: str, keywords: List[Dict], fuzz_threshold=0.8) -> float:
    st = student_text.lower()
    total_weight = sum(k.get('weight',1.0) for k in keywords) if keywords else 0.0
    if total_weight == 0:
        return 100.0
    matched = 0.0
    for kw in keywords:
        term = kw['term'].lower()
        w = kw.get('weight',1.0)
        if term in st:
            matched += w
            continue
        # fuzzy token-level
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
    return float(((cos + 1)/2)*100.0)

"""def grammar_score(student_text: str) -> float:
    issues = ft.check(student_text)
    sents = list(nlp(student_text).sents)
    avg_issues = len(issues) / max(1, len(sents))
    score = max(0.0, 100.0 - min(100.0, (avg_issues / 3.0) * 100.0))
    return float(score)"""

GRAMMAR_MODEL_NAME = "prithivida/grammar_error_correcter_v1"

# device should already be computed earlier in file
# device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading grammar-correction model on device:", device)
_g_tokenizer = AutoTokenizer.from_pretrained(GRAMMAR_MODEL_NAME)
_g_model = AutoModelForSeq2SeqLM.from_pretrained(GRAMMAR_MODEL_NAME).to(device)

# small persistent cache to avoid repeated model calls
_grammar_cache = shelve.open("grammar_cache.db")

def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def correct_text_with_model(text: str, max_length: int = 512) -> str:
    """
    Returns the model's corrected version of `text`.
    Uses cache to reduce repeated calls.
    """
    key = _cache_key(text)
    if key in _grammar_cache:
        return _grammar_cache[key]

    # prepare prompt (some models expect a prefix like "fix: " - not for all; adapt if needed)
    inp = text
    inputs = _g_tokenizer.encode(inp, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    # generate
    with torch.no_grad():
        out = _g_model.generate(inputs, max_new_tokens=256, do_sample=False)
    corrected = _g_tokenizer.decode(out[0], skip_special_tokens=True).strip()
    # store in cache
    _grammar_cache[key] = corrected
    _grammar_cache.sync()
    return corrected

def grammar_score_model(student_text: str) -> float:
    """
    Compute a 0..100 grammar score:
      - 100: no corrections required
      - lower as model makes more edits
    Approach:
      - compute normalized edit distance (char-level or token-level)
      - map to score with a saturating function
    """
    if not student_text or student_text.strip()=="":
        return 0.0
    # optionally truncate extremely long text for speed
    to_process = student_text if len(student_text) <= 2000 else student_text[:2000]

    try:
        corrected = correct_text_with_model(to_process)
    except Exception as e:
        print("Grammar model failed:", e)
        # fallback to simple heuristic (no grammar check)
        # return neutral 70 to not overly penalize
        return 70.0

    # compute normalized Levenshtein ratio (1.0 -> identical)
    # use ratio = 1 - (edits / max_len)
    orig = to_process.strip()
    corr = corrected.strip()
    # use char-level Levenshtein distance (fast)
    ed = Levenshtein.distance(orig, corr)
    maxlen = max(1, len(orig))
    norm_change = ed / maxlen  # 0.0 -> identical, larger -> more changes

    # map to score: no change -> 100, huge change -> closer to 0
    # we consider that ~30% of characters changed is very bad
    score = max(0.0, 100.0 * (1.0 - (norm_change / 0.30)))
    # clamp
    if score > 100.0:
        score = 100.0
    if score < 0.0:
        score = 0.0

    # optional: small penalty for extremely short answers
    tokens = len(re.findall(r"\w+", orig))
    if tokens < 3:
        score = min(score, 50.0)

    return float(round(score, 2))

# --- END: model-based grammar scoring ---

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
    return float((hits / len(reference_short))*100.0)

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

def aggregate_scores(k,s,g,c,p):
    return 0.3*k + 0.4*s + 0.15*g + 0.10*c + 0.05*p

########## LLM-BASED GRADER ##########

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

"""def llm_grade(question: str, ref_short: List[str], ref_long: str, student_text: str, max_marks: int=10) -> Dict:
    prompt = LLM_PROMPT.format(question=question,
                               ref_short="\n".join(ref_short),
                               ref_long=ref_long,
                               student=student_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = llm.generate(**inputs, max_new_tokens=256)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # try to extract JSON object from decoded (if model returns text + json)
    # naive approach: find first '{' and last '}'
    try:
        jstart = decoded.index("{")
        jend = decoded.rindex("}")+1
        js = decoded[jstart:jend]
        data = json.loads(js)
        return data
    except Exception as e:
        # fallback: return an explanation and 0s
        return {"keyword_pct":0,"semantic_pct":0,"grammar_pct":0,"coverage_pct":0,"presentation_pct":0,"recommended_marks":0,"explanation":decoded}
"""

def llm_grade(question: str, ref_short: List[str], ref_long: str, student_text: str, max_marks: int=10) -> Dict:
    """
    Safely format the prompt and call the LLM.
    - Escapes literal braces in the JSON description (done in LLM_PROMPT).
    - Truncates long texts to avoid tokenizer truncation.
    - Returns a dict; on parse failure returns explanation text inside 'explanation'.
    """
    # truncate long inputs to avoid hitting model token limit (adjust sizes as needed)
    def _truncate(s, chars=2000):
        if not s:
            return ""
        return s if len(s) <= chars else s[:chars-3] + "..."
    try:
        prompt = LLM_PROMPT.format(
            question=_truncate(question, 800),
            ref_short=_truncate("\n".join(ref_short), 1200),
            ref_long=_truncate(ref_long, 1200),
            student=_truncate(student_text, 1500)
        )
    except Exception as e:
        # log formatting error and return safe fallback
        print("Error formatting LLM prompt:", str(e))
        return {
            "keyword_pct": 0,
            "semantic_pct": 0,
            "grammar_pct": 0,
            "coverage_pct": 0,
            "presentation_pct": 0,
            "recommended_marks": 0,
            "explanation": f"LLM prompt formatting failed: {e}"
        }

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = llm.generate(**inputs, max_new_tokens=256)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    # try to extract JSON object from the decoded output
    try:
        jstart = decoded.index("{")
        jend = decoded.rindex("}")+1
        js = decoded[jstart:jend]
        data = json.loads(js)
        # ensure numeric fields exist and coerce types
        for k in ["keyword_pct","semantic_pct","grammar_pct","coverage_pct","presentation_pct","recommended_marks"]:
            if k in data:
                try:
                    data[k] = float(data[k])
                except:
                    pass
        if "explanation" not in data:
            data["explanation"] = decoded
        return data
    except Exception as e:
        # parsing failed; return whole decoded text in explanation
        return {
            "keyword_pct": 0,
            "semantic_pct": 0,
            "grammar_pct": 0,
            "coverage_pct": 0,
            "presentation_pct": 0,
            "recommended_marks": 0,
            "explanation": f"Failed to parse LLM output as JSON. Raw output: {decoded}"
        }

########## HIGH-LEVEL PIPELINE ##########

def evaluate_from_image(image_path: str, question: str, reference: Dict) -> Dict:
    """
    Runs OCR, coalesce text, generates component scores, runs LLM for explanation,
    computes composite confidence and returns a record for audit.
    reference = {"reference_long": str, "reference_short": [str], "keywords": [{"term":..., "weight":...}], "max_marks": int}
    """
    blocks = ocr_image_to_blocks(image_path)
    paras = coalesce_blocks(blocks)
    # join all paras as student's answer (simple approach: you can map to QIDs later)
    student_text = " ".join([p["text"] for p in paras])
    avg_ocr_conf = float(np.mean([p["conf"] for p in paras])) if paras else 0.0

    # deterministic scores
    k = keyword_score(student_text, reference.get("keywords", []))
    s = semantic_score(student_text, [reference.get("reference_long","")])
    #g = grammar_score(student_text)
    g = grammar_score_model(student_text)
    c = content_coverage_score(student_text, reference.get("reference_short", []))
    p = presentation_score(student_text)
    final_pct = aggregate_scores(k,s,g,c,p)
    rec_marks = final_pct/100.0 * reference.get("max_marks",10)

    # LLM grade for explanation
    llm_out = llm_grade(question, reference.get("reference_short", []), reference.get("reference_long",""), student_text, reference.get("max_marks",10))
    # grader confidence proxy: if LLM returned recommended_marks, compare with deterministic rec_marks
    grader_confidence = 1.0 - abs((llm_out.get("recommended_marks", rec_marks) - rec_marks) / max(1.0, reference.get("max_marks",10)))
    # semantic_conf normalized 0..1
    semantic_conf = s/100.0
    composite = 0.4*avg_ocr_conf + 0.4*semantic_conf + 0.2*grader_confidence

    record = {
        "image_path": image_path,
        "student_text": student_text,
        "avg_ocr_conf": avg_ocr_conf,
        "component_scores": {"keyword_pct":k,"semantic_pct":s,"grammar_pct":g,"coverage_pct":c,"presentation_pct":p},
        "deterministic_final_pct": final_pct,
        "deterministic_recommended_marks": rec_marks,
        "llm_output": llm_out,
        "grader_confidence": grader_confidence,
        "composite_confidence": composite,
        "route": "teacher_review" if composite < 0.6 else ("auto_accept" if composite >= 0.85 else "partial_review")
    }
    return record

if __name__ == "__main__":
    # test run with an example image in examples/
    import sys
    if len(sys.argv) < 2:
        print("Usage: python icr_pipeline.py examples/sample.jpg")
        sys.exit(1)
    img = sys.argv[1]
    # simple example reference
    reference = {
        "max_marks": 10,
        "keywords": [{"term":"bayes theorem","weight":1.0},{"term":"posterior","weight":1.0}],
        "reference_long": "Bayes theorem: P(A|B) = P(B|A)P(A)/P(B). It gives the posterior probability given prior and likelihood.",
        "reference_short": ["posterior probability","prior","likelihood","P(A|B)=P(B|A)P(A)/P(B)"]
    }
    res = evaluate_from_image(img, "State Bayes theorem and explain.", reference)
    print(json.dumps(res, indent=2))
