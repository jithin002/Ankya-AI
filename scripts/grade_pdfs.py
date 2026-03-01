import os
import sys
import tempfile
import json
import fitz

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.icr_pipeline3 import evaluate_from_image, MM
from pipelines.auto_rubric import generate_rubric_from_answer_key, extract_text_from_pdf

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    questions_pdf = os.path.join(base_dir, "answer_sheets", "Questions.pdf")
    answer_key_pdf = os.path.join(base_dir, "answer_sheets", "Answer_Key.pdf")
    sample_answers_pdf = os.path.join(base_dir, "answer_sheets", "Sample answers.pdf")

    print("[1/4] Extracting text from Questions and Answer Key PDFs...")
    qp_text = extract_text_from_pdf(questions_pdf)
    ak_text = extract_text_from_pdf(answer_key_pdf)
    
    # We use a generic prompt since we don't know the exact question right now.
    target_question = "Grade the student's answer based on the provided answer key."
    
    print(f"Extracted {len(qp_text)} chars from Questions (truncating to 1000 for LLM)")
    print(f"Extracted {len(ak_text)} chars from Answer Key (truncating to 1500 for LLM)")

    print("\n[2/4] Generating Rubric with TinyLlama...")
    rubric = generate_rubric_from_answer_key(target_question, ak_text, qp_text)
    
    print("\n--- Generated Rubric ---")
    print(json.dumps(rubric, indent=2))
    print("------------------------\n")

    print("[3/4] Converting Sample Answers PDF to images for OCR...")
    temp_paths = []
    doc = fitz.open(sample_answers_pdf)
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(dpi=150)
        # Create temp file but immediately close the auto-opened handle 
        # so PyMuPDF can write to it without permission errors on Windows.
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp.close()  
        pix.save(tmp.name)
        temp_paths.append(tmp.name)
        
    print(f"Extracted {len(temp_paths)} pages from Sample Answers.")

    print("\n[4/4] Evaluating Sample Answers...")
    all_results = []
    total_marks = 0
    total_max_marks = rubric.get("max_marks", 10) * len(temp_paths)
    
    for idx, t_path in enumerate(temp_paths):
        print(f"\n--- Grading Page {idx + 1}/{len(temp_paths)} ---")
        res = evaluate_from_image(t_path, target_question, rubric)
        res["page_num"] = idx + 1
        all_results.append(res)
        
        # Determine marks based on dashboard logic
        llm_res = res.get("llm_output", {})
        if not isinstance(llm_res, dict): llm_res = {}
        rec_marks = res.get("deterministic_recommended_marks", 0.0)
        llm_marks = llm_res.get("recommended_marks", 0.0)
        conf = res.get("composite_confidence", 0.0)
        
        display_marks = llm_marks if (conf > 0.6 and llm_marks > 0) else rec_marks
        total_marks += display_marks
        
        print(f"\nPage {idx + 1} Results:")
        print(f"Extracted OCR Text: {res.get('student_text', '')[:150]}...")
        print(f"Marks Awarded: {display_marks:.1f} / {rubric.get('max_marks', 10)}")
        print(f"LLM Feedback: {llm_res.get('explanation', 'No feedback')}")
        print(f"Scores: Keywords: {res['component_scores'].get('keyword_pct',0):.1f}%, Semantic: {res['component_scores'].get('semantic_pct',0):.1f}%")

    print("\n==============================================")
    print(f"FINAL GRADE: {total_marks:.1f} / {total_max_marks}")
    print(f"PERCENTAGE:  {(total_marks / total_max_marks * 100) if total_max_marks > 0 else 0:.1f}%")
    print("==============================================")

    # Save report
    report_path = os.path.join(base_dir, "analysis", "pdf_grading_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full debug report to: {report_path}")

    # Cleanup temp images
    for t_path in temp_paths:
        if os.path.exists(t_path):
            os.unlink(t_path)

if __name__ == "__main__":
    main()
