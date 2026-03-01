# app/grader.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pipelines.icr_pipeline3 import evaluate_from_image
from pipelines.auto_rubric import generate_rubric_from_answer_key, extract_text_from_pdf
from typing import Optional
import uuid, os, json

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

