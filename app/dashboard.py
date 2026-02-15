import streamlit as st
import os
import tempfile
import json
import pandas as pd
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.icr_pipeline3 import evaluate_from_image

st.set_page_config(
    page_title="ICR Intelligent Grader",
    page_icon="📝",
    layout="wide"
)

# Custom CSS for "Good and Interactive" look
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        color: #333333; /* Ensure text is visible on light bg */
    }
    .metric-card h3 {
        color: #333333;
    }
    .score-big {
        font-size: 3em;
        font-weight: bold;
        color: #4CAF50;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .feedback-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        color: #333333; /* Ensure text is visible on light bg */
    }
</style>
""", unsafe_allow_html=True)

st.title("📝 Ankya AI - Intelligent Classroom Grader")
st.markdown("Upload a student's handwritten answer to grade it automatically using **OCR + AI (Ankya)**.")

# --- Sidebar: Configuration ---
st.sidebar.header("⚙️ Grading Rubric")

with st.sidebar.form("rubric_form"):
    question = st.text_area("Question", "Explain Newton's first law of motion.", height=100)
    
    # Defaults
    def_ref_long = "Newton's first law states that an object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force (inertia)."
    def_ref_short = "inertia, object at rest stays at rest, object in motion stays in motion, external force"
    def_keywords = "Newton, Inertia, Force, Motion, Rest"
    
    ref_long = st.text_area("Reference Answer (Long)", def_ref_long, height=150)
    ref_short = st.text_area("Key Points (comma separated)", def_ref_short)
    keywords = st.text_input("Keywords (comma separated)", def_keywords)
    max_marks = st.number_input("Max Marks", min_value=1, max_value=100, value=10)
    
    submit_rubric = st.form_submit_button("Update Rubric")

# Parse lists
ref_short_list = [x.strip() for x in ref_short.split(",") if x.strip()]
keywords_list = [{"term": x.strip(), "weight": 1.0} for x in keywords.split(",") if x.strip()]

# Reference object
reference = {
    "reference_long": ref_long,
    "reference_short": ref_short_list,
    "keywords": keywords_list,
    "max_marks": max_marks
}

# --- Main Area ---

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Upload Answer")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Student Answer", use_column_width=True)

if uploaded_file and st.button("🚀 Grade Answer", type="primary"):
    with st.spinner("Processing... This may take a moment (loading models)..."):
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        try:
            # Run Pipeline
            # We redirect stdout to capture logs nicely if needed, or just let it run
            result = evaluate_from_image(tmp_path, question, reference)
            
            # --- Results Display ---
            st.success("Grading Complete!")
            
            # Extract scores
            scores = result.get("component_scores", {})
            llm_res = result.get("llm_output", {})
            
            # Handle possible LLM failure/fallback structure for display
            if not isinstance(llm_res, dict): llm_res = {}
            
            final_pct = result.get("deterministic_final_pct", 0.0)
            rec_marks = result.get("deterministic_recommended_marks", 0.0)
            llm_marks = llm_res.get("recommended_marks", 0.0)
            
            # Use LLM recommendation if high confidence, else deterministic
            conf = result.get("composite_confidence", 0.0)
            display_marks = llm_marks if (conf > 0.6 and llm_marks > 0) else rec_marks
            display_pct = (display_marks / max_marks) * 100
            
            with col2:
                st.subheader("2. Grading Results")
                
                # Big Score Card
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Final Grade</h3>
                    <div class="score-big">{display_marks:.1f} / {max_marks}</div>
                    <div style="color:gray">({display_pct:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("") # Spacer

                # Component Breakdown
                st.write("#### Component Analysis")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Keywords", f"{scores.get('keyword_pct', 0):.0f}%")
                    st.progress(int(scores.get('keyword_pct', 0))/100)
                    
                    st.metric("Grammar", f"{scores.get('grammar_pct', 0):.0f}%")
                    st.progress(int(scores.get('grammar_pct', 0))/100)
                    
                with c2:
                    st.metric("Semantic Meaning", f"{scores.get('semantic_pct', 0):.0f}%")
                    st.progress(int(scores.get('semantic_pct', 0))/100)
                    
                    st.metric("Presentation", f"{scores.get('presentation_pct', 0):.0f}%")
                    st.progress(int(scores.get('presentation_pct', 0))/100)

                # Feedback
                st.write("#### 🤖 AI Feedback")
                explanation = llm_res.get("explanation", "No detailed feedback generated.")
                st.markdown(f"""<div class="feedback-box">{explanation}</div>""", unsafe_allow_html=True)

            # Full Details Section
            st.divider()
            with st.expander("📄 See Extracted Text (OCR)"):
                st.text(result.get("student_text", ""))
                
            with st.expander("🔍 View Raw Debug Data (JSON)"):
                st.json(result)
                
        except Exception as e:
            st.error(f"An error occurred during grading: {e}")
            import traceback
            st.code(traceback.format_exc())
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
