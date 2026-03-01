import os
import json
import re

from pipelines.icr_pipeline3 import MM, DEVICE
import torch

def extract_text_from_pdf(pdf_path_or_bytes):
    """Extract raw text from an Answer Key PDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF is required to process PDFs. Please install it using `pip install pymupdf`")
        
    if isinstance(pdf_path_or_bytes, str):
        doc = fitz.open(pdf_path_or_bytes)
    else:
        doc = fitz.open(stream=pdf_path_or_bytes, filetype="pdf")
        
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)

def generate_rubric_from_answer_key(question: str, answer_key_text: str, question_document_text: str = "") -> dict:
    """
    Uses the local LLM to generate a grading rubric from the provided answer key text.
    """
    print("Loading LLM for Auto-Rubric Generation...")
    tokenizer, model = MM.load_llm()
    if not model or not tokenizer:
        raise RuntimeError("Failed to load LLM. Check GPU memory or model path.")
        
    sys_msg = (
        "You are an expert curriculum designer. "
        "Given a Question and its official Answer Key, extract the required grading criteria. "
        "Do NOT write any code. Do NOT talk. Just answer strictly in the requested format."
    )
    
    user_msg = f"""
[Target Question / Topic]
{question}

[Question Paper / Syllabus context]
{question_document_text[:1000] if question_document_text else "Not provided."}

[Official Answer Key extract]
{answer_key_text[:1500]}

[Instruction]
Extract the key information needed to grade a student's answer.
Format your output EXACTLY as this JSON:
{{
  "reference_long": "The full intended answer summarizing the core logic here",
  "reference_short": ["key concept 1", "key concept 2", "key concept 3"],
  "keywords": ["Term1", "Term2", "Term3"],
  "max_marks": 10
}}
"""
    prompt = f"<|system|>\n{sys_msg}</s>\n<|user|>\n{user_msg}</s>\n<|assistant|>\n"
    pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs, 
                max_new_tokens=300, 
                do_sample=False, 
                temperature=0.1,
                pad_token_id=pad_token_id
            )
            
        generated_ids = out[0][inputs['input_ids'].shape[1]:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"[LLM RUBRIC RAW] {decoded}")
        
        # Try to parse JSON from the output
        # Sometimes models wrap in ```json ... ```
        json_match = re.search(r'\{.*\}', decoded, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                # Validate and fill in gaps
                return {
                    "reference_long": data.get("reference_long", "Could not generate long reference."),
                    "reference_short": data.get("reference_short", ["fallback"]),
                    "keywords": [{"term": k, "weight": 1.0} for k in data.get("keywords", ["fallback"])],
                    "max_marks": int(data.get("max_marks", 10))
                }
            except json.JSONDecodeError:
                pass
                
        # Fallback if parsing fails
        return {
            "reference_long": answer_key_text[:500] if len(answer_key_text) > 5 else "Fallback logic.",
            "reference_short": ["concept"],
            "keywords": [{"term": "keyword", "weight": 1.0}],
            "max_marks": 10
        }
        
    except Exception as e:
        print(f"Error generating rubric: {e}")
        return {
            "reference_long": answer_key_text[:500],
            "reference_short": ["error"],
            "keywords": [{"term": "error", "weight": 1.0}],
            "max_marks": 10
        }
    finally:
        MM.unload_llm(tokenizer, model)
