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
    prompt = f"<|system|>\n{sys_msg}</s>\n
