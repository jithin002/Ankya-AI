# app/grader.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pipelines.icr_pipeline3 import evaluate_from_image
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
    # save uploaded file
    ext = os.path.splitext(file.filename)[1]
    fname = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(LOG_DIR, fname)
    contents = await file.read()
    with open(path, "wb") as f:
        f.write(contents)

    # Parse inputs from form
    ref_short_list = [x.strip() for x in ref_short.split(",") if x.strip()]
    keywords_list = [{"term": x.strip(), "weight": 1.0} for x in keywords.split(",") if x.strip()]

    reference = {
        "max_marks": max_marks,
        "keywords": keywords_list,
        "reference_long": ref_long,
        "reference_short": ref_short_list
    }

    record = evaluate_from_image(path, question, reference)
    # write audit log
    log_path = os.path.join(LOG_DIR, f"{uuid.uuid4().hex}.json")
    with open(log_path, "w") as f:
        json.dump(record, f, indent=2)
    return record
