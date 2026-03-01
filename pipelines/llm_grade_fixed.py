import re
import json
from typing import Dict, List, Optional

def llm_grade(question: str, ref_short: List[str], ref_long: str, student: str, tokenizer, model) -> Optional[Dict]:
    """
    Grade student answer using LLM (TinyLlama).
    Returns a dict with scores and feedback, or None if LLM fails.
    """
    if not tokenizer or not model:
        print("[LLM] Tokenizer or model is None, skipping LLM grading")
        return None
    
    def trunc(s, n): 
        return s[:n] if s else ""
    
    # More explicit prompt for JSON-only output
    sys_msg = """You are an AI grading assistant. Analyze the student answer and return ONLY a JSON object.
Do not include any explanatory text before or after the JSON. Just the JSON object."""
    
    user_msg = f"""Question: {trunc(question, 200)}
Key Points Expected: {trunc(", ".join(ref_short), 250)}
Reference Answer: {trunc(ref_long, 350)}
Student Answer: {trunc(student, 500)}

Output ONLY this JSON structure with your assessment:
{{
  "keyword_pct": <number 0-100>,
  "semantic_pct": <number 0-100>,
  "grammar_pct": <number 0-100>,
  "coverage_pct": <number 0-100>,
  "presentation_pct": <number 0-100>,
  "recommended_marks": <number 0-10>,
  "explanation": "<brief feedback string>"
}}"""

    # TinyLlama chat template
    prompt = f"<|system|>\n{sys_msg}</s>\n<|user|>\n{user_msg}</s>\n<|assistant|>\n"

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(device)
    
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs, 
                max_new_tokens=300,
                do_sample=False, 
                temperature=0.1, 
                repetition_penalty=1.2,
                pad_token_id=pad_token_id
            )
        
        # Decode only new tokens
        generated_ids = out[0][inputs['input_ids'].shape[1]:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"[LLM JSON FEEDBACK] {decoded}") # Debug
        
        # Extract JSON
        start = decoded.find('{')
        end = decoded.rfind('}')
        if start != -1 and end != -1:
            json_str = decoded[start:end+1]
            data = json.loads(json_str)
            for k in ["keyword_pct", "semantic_pct", "grammar_pct", "coverage_pct", "presentation_pct", "recommended_marks"]:
                if k in data:
                    try:
                        data[k] = float(data[k])
                    except (ValueError, TypeError):
                        pass
            return data
        else:
            print("No JSON found in LLM output.")
            return None
            
    except Exception as e:
        print(f"LLM Error: {e}")
        import traceback
        traceback.print_exc()
        return None
