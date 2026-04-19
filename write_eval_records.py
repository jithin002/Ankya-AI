"""
write_eval_records.py - converts benchmark_results_fresh.json to eval_records_fresh.jsonl
Run once: .\\venv\\Scripts\\python write_eval_records.py
"""
import os, json, sys
ROOT = os.path.dirname(os.path.abspath(__file__))

src = os.path.join(ROOT, "benchmark_results_fresh.json")
dst = os.path.join(ROOT, "eval_records_fresh.jsonl")

with open(src, "r", encoding="utf-8") as f:
    bm = json.load(f)

lines = []
for sample_name, data in bm.items():
    cs = data.get("component_scores", {})
    record = {
        "image_path":    os.path.join(ROOT, "examples", sample_name),
        "sample_name":   sample_name,
        "q_label":       "full",
        "student_text":  "",
        "avg_ocr_conf":  data.get("avg_ocr_conf", 0),
        "component_scores": {
            "keyword_pct":      cs.get("keyword_pct", 0),
            "semantic_pct":     cs.get("semantic_pct", 0),
            "grammar_pct":      cs.get("grammar_pct", 0),
            "coverage_pct":     cs.get("coverage_pct", 0),
            "presentation_pct": cs.get("presentation_pct", 0),
        },
        "deterministic_final_pct": round(
            0.30 * cs.get("keyword_pct", 0) +
            0.40 * cs.get("semantic_pct", 0) +
            0.15 * cs.get("grammar_pct", 0) +
            0.10 * cs.get("coverage_pct", 0) +
            0.05 * cs.get("presentation_pct", 0), 4
        ),
        "deterministic_recommended_marks": data.get("ai_marks", 0),
        "composite_confidence": data.get("composite_confidence", 0),
        "timings_s":    data.get("timings_s", {}),
        "human_marks":  data.get("human_marks"),
        "cer":          data.get("cer"),
        "wer":          data.get("wer"),
        "llm_explanation": "",
        "rag_confidence": None,
        "auto_matched_question": None,
    }
    lines.append(record)

with open(dst, "w", encoding="utf-8") as f:
    for rec in lines:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Written {len(lines)} records to {dst}")
