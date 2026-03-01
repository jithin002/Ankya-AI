import streamlit as st
import os
import tempfile
import json
import pandas as pd
import sys
import altair as alt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.icr_pipeline3 import evaluate_from_image, MM
from pipelines.auto_rubric import generate_rubric_from_answer_key, extract_text_from_pdf

st.set_page_config(
    page_title="Ankya AI - Intelligent Grader",
    page_icon="📝",
    layout="wide"
)

# Custom CSS for "Good and Interactive" look
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        color: #000000;
    }
    .metric-card h3 {
        color: #000000;
        margin-bottom: 5px;
    }
    .score-big {
        font-size: 3em;
        font-weight: bold;
        color: #2E7D32; /* Darker green for better contrast */
    }
    .stProgress > div > div > div > div {
        background-color: #2E7D32;
    }
    .feedback-box {
        background-color: #f1f8e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
        color: #000000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-card p {
        background-color: #f8f9fa;
        padding: 5px;
        border-radius: 5px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# --- Caching AI Models ---
@st.cache_resource(show_spinner="Loading AI Models into GPU Strategy (First Run Only)...")
def get_model_manager():
    """Returns the ModelManager instance. Streamlit caches this so models stay in GPU memory."""
    return MM

model_manager = get_model_manager()

# --- Session State ---
if 'grading_result' not in st.session_state:
    st.session_state.grading_result = None
if 'current_image_path' not in st.session_state:
    st.session_state.current_image_path = None

if 'def_ref_long' not in st.session_state:
    st.session_state.def_ref_long = "Newton's first law states that an object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force (inertia)."
if 'def_ref_short' not in st.session_state:
    st.session_state.def_ref_short = "inertia, object at rest stays at rest, object in motion stays in motion, external force"
if 'def_keywords' not in st.session_state:
    st.session_state.def_keywords = "Newton, Inertia, Force, Motion, Rest"
if 'def_max_marks' not in st.session_state:
    st.session_state.def_max_marks = 10

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["📝 Grader", "📊 Model Benchmark"])

if page == "📝 Grader":
    st.title("📝 Ankya AI - Intelligent Classroom Grader")
    st.markdown("Upload a student's handwritten answer to grade it automatically using **OCR + AI (Ankya)**.")
    
    # --- Sidebar: Configuration ---
    st.sidebar.header("⚙️ Grading Rubric")
    
    with st.sidebar.expander("🪄 Auto-Generate Rubric (Beta)", expanded=False):
        st.markdown("**1. Upload Knowledge Base**")
        st.caption("Upload an Answer Key and/or Question Paper to formulate the rubric.")
        qp_file = st.file_uploader("Upload Question Paper (PDF/TXT) [Optional]", type=["pdf", "txt"], key="qp_upload")
        ak_file = st.file_uploader("Upload Answer Key (PDF/TXT)", type=["pdf", "txt"], key="ak_upload")
        
        st.markdown("**2. Specify Target Question**")
        st.caption("Since the uploaded document may contain many questions, please paste the SPECIFIC question you want to grade below:")
        ak_question = st.text_area("Question Text", "Explain Newton's first law of motion.", key="ak_q")
        
        if st.button("Generate Rubric", use_container_width=True) and ak_file:
            with st.spinner("Parsing document(s) and generating rubric..."):
                
                # Parse Question Paper
                qp_text = ""
                if qp_file:
                    if qp_file.name.lower().endswith(".pdf"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(qp_file.getbuffer())
                            tmp_path = tmp.name
                        qp_text = extract_text_from_pdf(tmp_path)
                        os.unlink(tmp_path)
                    else:
                        qp_text = qp_file.getvalue().decode("utf-8")
                        
                # Parse Answer Key
                if ak_file.name.lower().endswith(".pdf"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(ak_file.getbuffer())
                        tmp_path = tmp.name
                    ak_text = extract_text_from_pdf(tmp_path)
                    os.unlink(tmp_path)
                else:
                    ak_text = ak_file.getvalue().decode("utf-8")
                    
                generated_rubric = generate_rubric_from_answer_key(ak_question, ak_text, qp_text)
                
                # Update session state with generated rubric
                st.session_state.def_ref_long = generated_rubric.get("reference_long", "")
                st.session_state.def_ref_short = ", ".join(generated_rubric.get("reference_short", []))
                
                kws = generated_rubric.get("keywords", [])
                if isinstance(kws, list) and len(kws) > 0 and isinstance(kws[0], dict):
                    kws = [k["term"] for k in kws]
                st.session_state.def_keywords = ", ".join(kws)
                st.session_state.def_max_marks = generated_rubric.get("max_marks", 10)
                st.success("Rubric updated successfully!")
                
    st.sidebar.markdown("---")
    
    with st.sidebar.form("rubric_form"):
        question = st.text_area("Question", "Explain Newton's first law of motion.", height=100)
        
        ref_long = st.text_area("Reference Answer (Long)", st.session_state.def_ref_long, height=150)
        ref_short = st.text_area("Key Points (comma separated)", st.session_state.def_ref_short)
        keywords = st.text_input("Keywords (comma separated)", st.session_state.def_keywords)
        max_marks = st.number_input("Max Marks", min_value=1, max_value=100, value=st.session_state.def_max_marks)
        
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
    
    st.subheader("1. Upload Answer(s)")
    uploaded_file = st.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])
    
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        if uploaded_file is not None:
            if uploaded_file.name.lower().endswith(".pdf"):
                st.info(f"📄 PDF Document Uploaded: **{uploaded_file.name}**\n\nThe pages will be extracted and graded sequentially when you click Grade.")
            else:
                st.image(uploaded_file, caption="Student Answer", use_column_width=True)
            
            # Reset state if new file uploaded
            if st.session_state.current_image_path != uploaded_file.name:
                st.session_state.grading_result = None
                st.session_state.current_image_path = uploaded_file.name
    
    grade_clicked = False
    with col1:
        if uploaded_file:
            grade_clicked = st.button("🚀 Grade Answer", type="primary", use_container_width=True)

    if grade_clicked:
        with st.spinner("Analyzing handwriting and evaluating answer(s)..."):
            is_pdf = uploaded_file.name.lower().endswith(".pdf")
            temp_paths = []
            
            if is_pdf:
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    for i in range(len(doc)):
                        page = doc[i]
                        pix = page.get_pixmap(dpi=150)
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                        tmp.close()
                        pix.save(tmp.name)
                        temp_paths.append(tmp.name)
                except ImportError:
                    st.error("PyMuPDF is required for PDFs. Please run `pip install pymupdf` in the terminal.")
                    st.stop()
            else:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                tmp.write(uploaded_file.getbuffer())
                tmp.close()
                temp_paths.append(tmp.name)
            
            try:
                all_results = []
                progress_bar = st.progress(0)
                st.write(f"Processing {len(temp_paths)} page(s)...")
                
                for idx, t_path in enumerate(temp_paths):
                    # Run Pipeline using the cached models
                    res = evaluate_from_image(t_path, question, reference)
                    res["page_num"] = idx + 1
                    all_results.append(res)
                    progress_bar.progress((idx + 1) / len(temp_paths))
                
                st.session_state.grading_result = all_results
                st.success("Grading Complete!")
            except Exception as e:
                st.error(f"An error occurred during grading: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                for t_path in temp_paths:
                    if os.path.exists(t_path):
                        os.unlink(t_path)
    
    # --- Results Display ---
    if st.session_state.grading_result is not None:
        results = st.session_state.grading_result
        if not isinstance(results, list):
            results = [results]
        
        total_marks = 0
        df_rows = []
        for res in results:
            llm_res = res.get("llm_output", {})
            if not isinstance(llm_res, dict): llm_res = {}
            rec_marks = res.get("deterministic_recommended_marks", 0.0)
            llm_marks = llm_res.get("recommended_marks", 0.0)
            conf = res.get("composite_confidence", 0.0)
            display_marks = llm_marks if (conf > 0.6 and llm_marks > 0) else rec_marks
            total_marks += display_marks
            
            df_rows.append({
                "Page": res.get("page_num", 1),
                "Marks Awarded": round(display_marks, 1),
                "Max Marks": max_marks,
                "Feedback": llm_res.get("explanation", "No detailed feedback."),
                "Extracted Text": res.get("student_text", "")[:100] + "..."
            })
        
        total_max_marks = max_marks * len(results)
        avg_pct = (total_marks / total_max_marks) * 100 if total_max_marks > 0 else 0
        
        with col2:
            st.subheader("2. Grading Results")
            
            # Export Button
            df_export = pd.DataFrame(df_rows)
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Grading Report (CSV)",
                data=csv,
                file_name='grading_report.csv',
                mime='text/csv',
                help="Download the consolidated rubric assessment for all pages."
            )
            
            # Big Score Card
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Grade</h3>
                <div class="score-big">{total_marks:.1f} / {total_max_marks}</div>
                <div style="color:gray">({avg_pct:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("") # Spacer

            # Page by Page Breakdown Loop
            for res in results:
                page_num = res.get("page_num", 1)
                st.markdown(f"### Page {page_num} Breakdown")
                
                scores = res.get("component_scores", {})
                llm_res = res.get("llm_output", {})
                if not isinstance(llm_res, dict): llm_res = {}

                # Feedback
                st.write("#### 🤖 AI Feedback")
                explanation = llm_res.get("explanation", "No detailed feedback generated.")
                st.markdown(f"""<div class="feedback-box">{explanation}</div>""", unsafe_allow_html=True)

                st.write("")
                
                # Component Breakdown Tabs
                tab1, tab2 = st.tabs(["📊 Score Breakdown", "📄 Extracted Text & Data"])
                
                with tab1:
                    st.write("##### Component Analysis")
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

                with tab2:
                    with st.expander("📄 See Extracted Text (OCR)", expanded=True):
                        st.text(res.get("student_text", ""))
                        
                    with st.expander("🔍 View Raw Debug Data (JSON)"):
                        st.json(res)
                
                st.divider()

elif page == "📊 Model Benchmark":
    st.title("📊 Pipeline Comparison Benchmark")
    st.markdown("Comparing **Pipeline V2 (Intermediate)** vs **Pipeline V3 (New)** on a standard test case.")
    
    benchmark_file = "benchmark_results.json"
    
    if os.path.exists(benchmark_file):
        with open(benchmark_file, "r") as f:
            data = json.load(f)
        
        human_marks = data.get("human_marks", 10.0)
        v2 = data.get("v2", {})
        v3 = data.get("v3", {})
        
        if not v2 or not v3:
            st.warning("Benchmark run incomplete. Please run scripts/benchmark_sample4.py")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Human Ground Truth", f"{human_marks:.1f}")
            with col2:
                v2_marks = v2.get("marks", 0)
                v2_err = abs(v2_marks - human_marks)
                st.metric("V2 Prediction", f"{v2_marks:.1f}", delta=f"-{v2_err:.2f} Error", delta_color="inverse")
            with col3:
                v3_marks = v3.get("marks", 0)
                v3_err = abs(v3_marks - human_marks)
                st.metric("V3 Prediction", f"{v3_marks:.1f}", delta=f"-{v3_err:.2f} Error", delta_color="inverse")
            
            st.divider()
            
            st.subheader("OCR Accuracy (Lower is Better)")
            c1, c2 = st.columns(2)
            with c1:
                # WER Chart
                chart_data = pd.DataFrame({
                    "Pipeline": ["V2", "V3"],
                    "WER": [v2.get("wer", 0), v3.get("wer", 0)]
                })
                c = alt.Chart(chart_data).mark_bar().encode(
                    x='Pipeline',
                    y='WER',
                    color='Pipeline'
                ).properties(title="Word Error Rate (WER)")
                st.altair_chart(c, use_container_width=True)
                
            with c2:
                # CER Chart
                chart_data_cer = pd.DataFrame({
                    "Pipeline": ["V2", "V3"],
                    "CER": [v2.get("cer", 0), v3.get("cer", 0)]
                })
                c_cer = alt.Chart(chart_data_cer).mark_bar().encode(
                    x='Pipeline',
                    y='CER',
                    color='Pipeline'
                ).properties(title="Character Error Rate (CER)")
                st.altair_chart(c_cer, use_container_width=True)

            st.subheader("OCR Output Comparison")
            with st.expander("Show Text Diff"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**V2 Text:**")
                    st.text(v2.get("text", ""))
                with col_b:
                    st.markdown("**V3 Text:**")
                    st.text(v3.get("text", ""))
                
                st.markdown("**Ground Truth:**")
                st.info(data.get("ground_truth_text", ""))
                
    else:
        st.info("Benchmark results not found. Running benchmark...")
        if st.button("Run Benchmark Now (This takes time)"):
            with st.spinner("Running V2 vs V3 benchmark..."):
                import subprocess
                cmd = [sys.executable, os.path.join("scripts", "benchmark_sample4.py")]
                subprocess.run(cmd)
                st.experimental_rerun()
