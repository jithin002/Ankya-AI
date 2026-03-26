import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import io

# ---------------- HELPER FUNCTIONS ----------------
def extract_text_and_images(file):
    """
    Extract text and page images from PDF.
    """
    full_content = []
    images = []

    pdf_bytes = file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page in doc:
        text = page.get_text()
        full_content.append(text.strip())

        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        images.append(image)

    return full_content, images

def parse_solutions(pdf_text_pages):
    """
    Parse solution pages into a dict {question_number: solution_text}.
    Assumes solutions are numbered same as questions.
    """
    solutions_dict = {}
    for page_text in pdf_text_pages:
        lines = page_text.split("\n")
        current_q = None
        current_text = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and '.' in line[:4]:
                if current_q:
                    solutions_dict[current_q] = "\n".join(current_text)
                try:
                    num, q_text = line.split('.', 1)
                    current_q = int(num.strip())
                    current_text = [q_text.strip()]
                except:
                    continue
            else:
                if current_q:
                    current_text.append(line)
        if current_q:
            solutions_dict[current_q] = "\n".join(current_text)
    return solutions_dict

def parse_questions(pdf_text_pages):
    """
    Parse question pages into list of dicts [{number, question}].
    """
    questions = []
    for page_text in pdf_text_pages:
        lines = page_text.split("\n")
        current_q = None
        current_text = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and '.' in line[:4]:
                if current_q:
                    questions.append({"number": current_q, "question": "\n".join(current_text)})
                try:
                    num, q_text = line.split('.', 1)
                    current_q = int(num.strip())
                    current_text = [q_text.strip()]
                except:
                    continue
            else:
                if current_q:
                    current_text.append(line)
        if current_q:
            questions.append({"number": current_q, "question": "\n".join(current_text)})
    return questions

# ---------------- DASHBOARD ----------------
st.set_page_config(page_title="PDF QA Navigator", layout="wide")
st.title("PDF Question & Answer Navigator (Solutions Extracted)")

st.markdown("""
Upload **Questions PDF** and **Solutions PDF**.  
Each solution will be matched to its corresponding question automatically.
""")

# ---------------- FILE UPLOAD ----------------
col1, col2 = st.columns(2)
with col1:
    questions_file = st.file_uploader("Upload Questions PDF", type=["pdf"])
with col2:
    solutions_file = st.file_uploader("Upload Solutions PDF", type=["pdf"])

if questions_file and solutions_file:
    try:
        st.info("Extracting Questions PDF...")
        question_pages, _ = extract_text_and_images(questions_file)
        questions_list = parse_questions(question_pages)

        st.info("Extracting Solutions PDF...")
        solution_pages, solution_images = extract_text_and_images(solutions_file)
        solutions_dict = parse_solutions(solution_pages)

        # Combine questions with solutions
        qa_dataset = []
        for q in questions_list:
            num = q["number"]
            question_text = q["question"]
            answer_text = solutions_dict.get(num, "Solution not found")
            qa_dataset.append({
                "number": num,
                "section": "A" if num <= 15 else "B" if num <= 22 else "C" if num <= 31 else "D",
                "question": question_text,
                "answer": answer_text
            })

        # ---------------- DASHBOARD SUMMARY ----------------
        st.subheader("Dataset Summary")
        section_counts = {"A":0,"B":0,"C":0,"D":0}
        for q in qa_dataset:
            section_counts[q["section"]] += 1
        st.metric("Total Questions", len(qa_dataset))
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Section A", section_counts["A"])
        colB.metric("Section B", section_counts["B"])
        colC.metric("Section C", section_counts["C"])
        colD.metric("Section D", section_counts["D"])

        # ---------------- QUESTION NAVIGATOR ----------------
        st.subheader("Question Navigator")
        section_choice = st.selectbox("Select Section", ["A","B","C","D"])
        section_questions = [q for q in qa_dataset if q["section"] == section_choice]

        if section_questions:
            question_numbers = [q["number"] for q in section_questions]
            question_choice = st.selectbox("Select Question Number", question_numbers)
            selected_q = next(q for q in section_questions if q["number"]==question_choice)

            st.markdown(f"### Question {selected_q['number']}")
            st.write(selected_q["question"])
            st.markdown("### Solution")
            st.write(selected_q["answer"])

        # ---------------- QUICK ACCESS ----------------
        st.subheader("Quick Access")
        user_input = st.text_input("Enter Section and Question Number (Example: A 3)")
        if user_input:
            try:
                sec, num = user_input.split()
                num = int(num)
                filtered = [q for q in qa_dataset if q["section"]==sec.upper()]
                if not filtered or num < 1 or num > len(filtered):
                    raise ValueError("Question number out of range")
                selected_q = filtered[num-1]

                st.markdown(f"### Question {selected_q['number']}")
                st.write(selected_q["question"])
                st.markdown("### Solution")
                st.write(selected_q["answer"])
            except Exception as e:
                st.error(f"Invalid input! Use format like: A 3\nError: {e}")

    except Exception as e:
        st.error(f"Error processing PDFs: {e}")

else:
    st.info("Please upload both Questions PDF and Solutions PDF to continue.")