# scripts/batch_eval.py
"""Batch-evaluate the OCR pipeline over images listed in data/labels.csv.
Saves results to eval_records_batch.jsonl and prints aggregate CER/WER.
"""
import os
import csv
import json
import sys
from tqdm import tqdm

# ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipelines.icr_pipeline3 import evaluate_from_image, trocr_init
import sys
from jiwer import wer, cer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
labels_csv = os.path.join(ROOT, "data", "labels.csv")
images_dir = os.path.join(ROOT, "data", "images")
out_path = os.path.join(ROOT, "eval_records_batch.jsonl")

rows = []
# optional model dir arg
model_dir = None
if len(sys.argv) > 1:
    model_dir = sys.argv[1]
    print("Using TrOCR model dir:", model_dir)
    try:
        trocr_init(model_name=model_dir)
    except Exception as e:
        print("Failed to init TrOCR with", model_dir, "falling back to default:", e)
with open(labels_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

results = []
refs = []
hyps = []
for r in tqdm(rows, desc="Evaluating"):
    fname = r.get('filename')
    text = r.get('text', "")
    img_path = os.path.join(images_dir, fname)
    if not os.path.exists(img_path):
        print("Missing image", img_path)
        continue
    rec = evaluate_from_image(img_path, "", {"reference_long": text, "reference_short": [], "keywords": [], "max_marks": 10}, debug_save_image=False)
    results.append(rec)
    refs.append(text)
    hyps.append(rec.get('student_text',''))
    with open(out_path, 'a', encoding='utf-8') as fo:
        fo.write(json.dumps(rec) + "\n")

# compute overall CER/WER with jiwer (cer returns float) - note jiwer.cer expects strings
try:
    total_wer = wer(refs, hyps)
except Exception:
    total_wer = None

try:
    total_cer = cer(refs, hyps)
except Exception:
    total_cer = None

print(f"WER: {total_wer}")
print(f"CER: {total_cer}")
print(f"Saved {len(results)} records to {out_path}")
