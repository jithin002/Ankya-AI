# app/grader.py
from fastapi import FastAPI, UploadFile, File
from pipelines.icr_pipeline3 import evaluate_from_image
import uuid, os, json

app = FastAPI()

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

@app.post("/grade-image/")
async def grade_image(file: UploadFile = File(...)):
    # save uploaded file
    ext = os.path.splitext(file.filename)[1]
    fname = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(LOG_DIR, fname)
    contents = await file.read()
    with open(path, "wb") as f:
        f.write(contents)

    # example reference - in real app, pass question/ref in body
    reference = {
        "max_marks": 10,
        "keywords": [{"term":"bayes theorem","weight":1.0},{"term":"posterior","weight":1.0}],
        "reference_long": "Bayes theorem: P(A|B) = P(B|A)P(A)/P(B). It gives the posterior probability given prior and likelihood.",
        "reference_short": ["posterior probability","prior","likelihood","P(A|B)=P(B|A)P(A)/P(B)"]
    }
    # For demo, set question text statically
    question = "State Bayes theorem and explain."

    record = evaluate_from_image(path, question, reference)
    # write audit log
    log_path = os.path.join(LOG_DIR, f"{uuid.uuid4().hex}.json")
    with open(log_path, "w") as f:
        json.dump(record, f, indent=2)
    return record
