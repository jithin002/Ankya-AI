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


def _parse_numbered_text(pdf_text_pages):
    """Extract numbered items from PDF text pages into {number: text} dict."""
    result = {}
    for page_text in pdf_text_pages:
        lines = page_text.split("\n")
        current_q = None
        current_text = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and '.' in line[:4]:
                if current_q is not None:
                    result[current_q] = "\n".join(current_text).strip()
                try:
                    num_str, rest = line.split('.', 1)
                    current_q = int(num_str.strip())
                    current_text = [rest.strip()]
                except:
                    continue
            else:
                if current_q is not None:
                    current_text.append(line)
        if current_q is not None:
            result[current_q] = "\n".join(current_text).strip()
    return result


@app.post("/parse-qa-documents/")
async def parse_qa_documents(
    questions_pdf: UploadFile = File(...),
    answers_pdf: UploadFile = File(...),
    max_marks_per_question: int = Form(10)
):
    """
    Accepts a Questions PDF and an Answer Key PDF.
    Extracts numbered Q&A pairs matched by number, returns a qa_dataset
    list with embedded rubrics ready for page-by-page grading in the frontend.
    """
    import fitz, re
    
    def pdf_to_text_pages(content: bytes):
        doc = fitz.open(stream=content, filetype="pdf")
        return [page.get_text().strip() for page in doc]

    q_bytes = await questions_pdf.read()
    a_bytes = await answers_pdf.read()

    questions_map = _parse_numbered_text(pdf_to_text_pages(q_bytes))
    answers_map = _parse_numbered_text(pdf_to_text_pages(a_bytes))

    qa_dataset = []
    for num in sorted(questions_map.keys()):
        q_text = questions_map[num]
        a_text = answers_map.get(num, "")
        section = "A" if num <= 15 else "B" if num <= 22 else "C" if num <= 31 else "D"
        
        words = re.findall(r"[A-Za-z]{4,}", a_text)
        freq = {}
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
    qa_dataset_json: str = Form("")   # Optional: full JSON of QA dataset for auto-RAG selection
):
    """
    Grade a single page image against a rubric.
    
    If qa_dataset_json is provided, the system automatically detects which question
    the student is answering using Semantic Similarity (RAG), then grades against that
    matched rubric — no manual question selection required.
    """
    import tempfile, torch, gc
    from sentence_transformers import SentenceTransformer, util

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save incoming image to temp file
    contents = await file.read()
    ext = os.path.splitext(file.filename)[1] or '.jpg'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(contents)
    tmp.close()

    try:
        # ── Step 1: Run OCR to extract raw student text ──────────────────────
        from pipelines.icr_pipeline3 import MM, ocr_image_to_blocks, merge_nearby_horizontal_blocks, coalesce_blocks, remove_ruled_lines, clean_ocr_noise
        import cv2, numpy as np

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raw_img = cv2.imread(tmp.name)
        if raw_img is not None:
            cleaned_img = remove_ruled_lines(raw_img)
            tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(tmp2.name, cleaned_img)
            tmp2.close()
            ocr_src = tmp2.name
        else:
            ocr_src = tmp.name

        ocr_reader = MM.load_ocr()
        trocr_bundle = MM.load_trocr()
        blocks = ocr_image_to_blocks(ocr_src, ocr_reader, trocr_bundle)
        MM.unload_trocr(trocr_bundle)
        MM.unload_ocr(ocr_reader)
        if ocr_src != tmp.name:
            try: os.unlink(ocr_src)
            except: pass

        blocks = merge_nearby_horizontal_blocks(blocks)
        paras = coalesce_blocks(blocks)
        student_text = " ".join([p["text"] for p in paras])
        student_text = clean_ocr_noise(student_text)
        print(f"[grade-page] OCR extracted {len(student_text)} chars")

        # ── Step 2: Auto-select rubric via Semantic RAG if qa_dataset given ──
        selected_question = question
        reference = {
            "max_marks": max_marks,
            "keywords": [{"term": x.strip(), "weight": 1.0} for x in keywords.split(",") if x.strip()],
            "reference_long": ref_long,
            "reference_short": [x.strip() for x in ref_short.split(",") if x.strip()]
        }
        matched_q_num = None

        if qa_dataset_json and student_text.strip():
            try:
                qa_dataset = json.loads(qa_dataset_json)
                if qa_dataset:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    sent_model = SentenceTransformer('all-MiniLM-L6-v2')
                    student_emb = sent_model.encode(student_text, convert_to_tensor=True)

                    best_score = -1.0
                    best_item = None
                    for item in qa_dataset:
                        # Compare student text to both the question and the answer
                        q_text = item.get("question", "")
                        a_text = item.get("answer", "")
                        combined = f"{q_text} {a_text}"
                        ref_emb = sent_model.encode(combined, convert_to_tensor=True)
                        score = float(util.cos_sim(student_emb, ref_emb)[0][0])
                        if score > best_score:
                            best_score = score
                            best_item = item

                    del sent_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if best_item and best_score > 0.15:
                        matched_q_num = best_item["number"]
                        rubric = best_item["rubric"]
                        selected_question = best_item["question"]
                        reference = {
                            "max_marks": rubric.get("max_marks", max_marks),
                            "keywords": rubric.get("keywords", []),
                            "reference_long": rubric.get("reference_long", ""),
                            "reference_short": rubric.get("reference_short", [])
                        }
                        print(f"[RAG] Auto-matched Q{matched_q_num} (score={best_score:.3f})")
                    else:
                        print(f"[RAG] No confident match (best score={best_score:.3f}), using manual rubric")
            except Exception as e:
                print(f"[RAG] Error during auto-selection: {e}, falling back to manual rubric")

        # ── Step 3: Run the rest of the grading pipeline (without OCR) ───────
        # We already have student_text, so we call evaluate_from_image directly
        from pipelines.icr_pipeline3 import evaluate_from_image
        result = evaluate_from_image(tmp.name, selected_question, reference)
        result["page_num"] = 1
        result["auto_matched_question"] = matched_q_num
        result["rag_confidence"] = best_score if qa_dataset_json else None

        return {"results": [result]}

    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
