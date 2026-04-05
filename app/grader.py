# app/grader.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pipelines.icr_pipeline3 import evaluate_from_image
from pipelines.auto_rubric import generate_rubric_from_answer_key, extract_text_from_pdf
from typing import Optional, List
import uuid, os, json, re, tempfile
import torch, gc

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows requests from any origin (ideal for local dev like Vite)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

@app.post("/grade-image/")
async def grade_image(
    file: UploadFile = File(...),
    question: str = Form("Explain Newton's first law of motion."),
    ref_long: str = Form("Newton's first law states that an object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force (inertia)."),
    ref_short: str = Form("inertia, object at rest stays at rest, object in motion stays in motion, external force"),
    keywords: str = Form("Newton, Inertia, Force, Motion, Rest"),
    max_marks: int = Form(10)
):
    # Parse inputs from form
    ref_short_list = [x.strip() for x in ref_short.split(",") if x.strip()]
    keywords_list = [{"term": x.strip(), "weight": 1.0} for x in keywords.split(",") if x.strip()]

    reference = {
        "max_marks": max_marks,
        "keywords": keywords_list,
        "reference_long": ref_long,
        "reference_short": ref_short_list
    }

    contents = await file.read()
    filename_lower = file.filename.lower()
    is_pdf = filename_lower.endswith('.pdf')
    temp_paths = []
    
    import tempfile
    
    if is_pdf:
        try:
            import fitz
            doc = fitz.open(stream=contents, filetype="pdf")
            for i in range(len(doc)):
                page = doc[i]
                pix = page.get_pixmap(dpi=150)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                tmp.close()
                pix.save(tmp.name)
                temp_paths.append(tmp.name)
        except ImportError:
            raise HTTPException(status_code=500, detail="PyMuPDF is required for PDFs. Please run `pip install pymupdf`")
    else:
        ext = os.path.splitext(file.filename)[1] or '.jpg'
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(contents)
        tmp.close()
        temp_paths.append(tmp.name)

    all_results = []
    
    try:
        for idx, t_path in enumerate(temp_paths):
            res = evaluate_from_image(t_path, question, reference)
            res["page_num"] = idx + 1
            all_results.append(res)
    finally:
        for t_path in temp_paths:
            if os.path.exists(t_path):
                try:
                    os.unlink(t_path)
                except:
                    pass

    # write audit log
    log_path = os.path.join(LOG_DIR, f"{uuid.uuid4().hex}.json")
    with open(log_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    return {"results": all_results}


@app.post("/generate-rubric/")
async def generate_rubric(
    question: str = Form(...),
    answer_key: UploadFile = File(...),
    question_paper: Optional[UploadFile] = File(None)
):
    try:
        # Process Answer Key
        ak_content = await answer_key.read()
        ak_text = ""
        if answer_key.filename.lower().endswith('.pdf'):
            ak_text = extract_text_from_pdf(ak_content)
        else:
            ak_text = ak_content.decode('utf-8')

        # Process Question Paper if provided
        qp_text = ""
        if question_paper:
            qp_content = await question_paper.read()
            if question_paper.filename.lower().endswith('.pdf'):
                qp_text = extract_text_from_pdf(qp_content)
            else:
                qp_text = qp_content.decode('utf-8')

        # Generate Rubric
        rubric = generate_rubric_from_answer_key(question, ak_text, qp_text)
        return rubric
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _parse_numbered_text(pdf_text_pages: list) -> dict:
    """
    Extract numbered items from PDF text pages into {number: text} dict.
    Supports formats: '1. text', '1) text', 'Q1. text', 'Q1: text', '1: text'
    """
    result = {}
    # Matches: 1. | 1) | Q1. | Q1: | 1: at start of line (after optional whitespace)
    Q_LINE_RE = re.compile(
        r'^\s*(?:Q|q)?\s*(\d{1,3})\s*[.):\]]\s*(.*)', re.IGNORECASE
    )
    for page_text in pdf_text_pages:
        if not page_text.strip():
            continue
        lines = page_text.split("\n")
        current_q = None
        current_text = []
        for line in lines:
            m = Q_LINE_RE.match(line)
            if m:
                if current_q is not None:
                    result[current_q] = "\n".join(current_text).strip()
                current_q = int(m.group(1))
                current_text = [m.group(2).strip()] if m.group(2).strip() else []
            else:
                if current_q is not None and line.strip():
                    current_text.append(line.strip())
        if current_q is not None:
            result[current_q] = "\n".join(current_text).strip()
    return result


def _ocr_pdf_bytes_to_text_pages(pdf_bytes: bytes) -> list:
    """
    Render each PDF page as an image and run EasyOCR on it.
    Used when fitz.get_text() returns empty (scanned PDFs).
    Handles 2-column layouts by splitting page vertically and reading each half.
    Returns a list of strings, one per page.
    """
    import fitz
    import numpy as np
    import cv2
    import easyocr

    reader = easyocr.Reader(['en'], gpu=False)  # CPU-only, lightweight for PDFs
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []

    for page in doc:
        # Render at 200 DPI for good OCR accuracy
        pix = page.get_pixmap(dpi=200)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        # Convert to BGR for OpenCV
        if pix.n == 4:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        H, W = img_bgr.shape[:2]

        # ── 2-column layout: split at midpoint and read each half separately ──
        # Detect if content is split in two columns by checking text density per half
        left_half  = img_bgr[:, :W//2]
        right_half = img_bgr[:, W//2:]

        def ocr_region(region):
            results = reader.readtext(region, detail=1)
            # Sort by Y then X so text flows naturally
            results_sorted = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))
            return "\n".join([text for _, text, conf in results_sorted if conf > 0.3])

        left_text  = ocr_region(left_half).strip()
        right_text = ocr_region(right_half).strip()

        # Heuristic: if both halves have meaningful content, treat as 2-column
        if len(left_text) > 50 and len(right_text) > 50:
            page_text = left_text + "\n" + right_text
        elif len(left_text) > len(right_text):
            # Single column on the left / full page – re-run on full image
            full_results = reader.readtext(img_bgr, detail=1)
            full_sorted  = sorted(full_results, key=lambda r: (r[0][0][1], r[0][0][0]))
            page_text = "\n".join([t for _, t, c in full_sorted if c > 0.3])
        else:
            full_results = reader.readtext(img_bgr, detail=1)
            full_sorted  = sorted(full_results, key=lambda r: (r[0][0][1], r[0][0][0]))
            page_text = "\n".join([t for _, t, c in full_sorted if c > 0.3])

        pages_text.append(page_text)

    return pages_text


@app.post("/parse-qa-documents/")
async def parse_qa_documents(
    questions_pdf: UploadFile = File(...),
    answers_pdf: UploadFile = File(...),
    max_marks_per_question: int = Form(10)
):
    """
    Accepts a Questions PDF and an Answer Key PDF (digital or scanned).
    Extracts numbered Q&A pairs matched by number, returns a qa_dataset list
    with embedded rubrics ready for per-page grading in the frontend.

    Handles scanned PDFs automatically via EasyOCR fallback.
    Handles 2-column layouts by splitting page images vertically.
    """
    import fitz

    def pdf_to_text_pages(content: bytes) -> list:
        """Try digital text layer first; fall back to OCR if blank."""
        doc = fitz.open(stream=content, filetype="pdf")
        pages = [page.get_text().strip() for page in doc]
        total_chars = sum(len(p) for p in pages)
        if total_chars < 50:
            # Scanned PDF: use EasyOCR
            print("[parse-qa] Digital text layer is empty – switching to OCR fallback")
            return _ocr_pdf_bytes_to_text_pages(content)
        return pages

    q_bytes = await questions_pdf.read()
    a_bytes = await answers_pdf.read()

    questions_map = _parse_numbered_text(pdf_to_text_pages(q_bytes))
    answers_map   = _parse_numbered_text(pdf_to_text_pages(a_bytes))

    # If questions PDF also came back blank, try using answers as both Q and A
    if not questions_map and answers_map:
        questions_map = {k: f"Question {k}" for k in answers_map}

    qa_dataset = []
    for num in sorted(questions_map.keys()):
        q_text = questions_map[num]
        a_text = answers_map.get(num, "")
        section = "A" if num <= 15 else "B" if num <= 22 else "C" if num <= 31 else "D"

        words = re.findall(r"[A-Za-z]{4,}", a_text)
        freq: dict = {}
        for w in words:
            freq[w.lower()] = freq.get(w.lower(), 0) + 1
        keywords = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:6]]

        qa_dataset.append({
            "number": num,
            "section": section,
            "question": q_text,
            "answer": a_text,
            "rubric": {
                "max_marks": max_marks_per_question,
                "reference_long": a_text,
                "reference_short": [s.strip() for s in a_text.split('.') if s.strip()][:4],
                "keywords": [{"term": k, "weight": 1.0} for k in keywords]
            }
        })

    return {"qa_dataset": qa_dataset, "total": len(qa_dataset)}


@app.post("/grade-page/")
async def grade_page(
    file: UploadFile = File(...),
    question: str = Form(""),
    ref_long: str = Form(""),
    ref_short: str = Form(""),
    keywords: str = Form(""),
    max_marks: int = Form(10),
    qa_dataset_json: str = Form(""),
    manual_q_overrides_json: str = Form("")
):
    """
    Grade a single page image. Returns a list of per-answer results.
    If qa_dataset_json is provided, the pipeline auto-detects all questions on
    the page and grades each one in sequence using the full dataset context.
    """
    import tempfile, torch, gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    contents = await file.read()
    ext = os.path.splitext(file.filename)[1] or '.jpg'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(contents)
    tmp.close()

    # Fallback reference for when qa_dataset is not provided
    reference = {
        "max_marks": max_marks,
        "keywords": [{"term": x.strip(), "weight": 1.0} for x in keywords.split(",") if x.strip()],
        "reference_long": ref_long,
        "reference_short": [x.strip() for x in ref_short.split(",") if x.strip()]
    }

    qa_dataset = None
    if qa_dataset_json:
        try:
            qa_dataset = json.loads(qa_dataset_json)
        except Exception:
            qa_dataset = None

    # Parse teacher's manual question overrides: {q_label: question_number}
    manual_q_overrides = {}
    if manual_q_overrides_json:
        try:
            raw = json.loads(manual_q_overrides_json)
            # values may be int or None; convert to int where set
            manual_q_overrides = {k: int(v) for k, v in raw.items() if v is not None}
        except Exception:
            manual_q_overrides = {}

    try:
        from pipelines.icr_pipeline3 import evaluate_from_image
        results = evaluate_from_image(tmp.name, question, reference,
                                      qa_dataset=qa_dataset,
                                      manual_q_overrides=manual_q_overrides)
        return {"results": results}
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

