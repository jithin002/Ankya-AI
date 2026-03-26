import pdfplumber
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import zip_longest

# ---------------- STEP 1: Extract PDF Text ----------------
def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

pdf_text = extract_pdf_text("Dataset1.pdf")
print("\nTEXT LOADED SUCCESSFULLY\n")

# ---------------- STEP 2: CLEAN TEXT ----------------
def clean_text(text):
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"H2O TECH LABS.*", "", text)
    text = re.sub(r"Class VIII.*", "", text)
    text = re.sub(r"Sample Question Paper.*", "", text)
    return text

pdf_text = clean_text(pdf_text)

# ---------------- STEP 3: KEEP ONLY AFTER SECTION A ----------------
def get_question_text(text):
    match = re.search(r"Section A(.*)", text, re.S)
    return match.group(1) if match else text

pdf_text = get_question_text(pdf_text)

# ---------------- STEP 4: EXTRACT QUESTIONS ----------------
def extract_all_questions(text):
    pattern = r"(\d+)\.\s(.*?)(?=\n\d+\.|\Z)"
    matches = re.findall(pattern, text, re.S)

    questions = []

    for num, q in matches:
        q = q.strip()

        if re.search(r"[a-d]\)", q) or "?" in q:
            questions.append((int(num), q))

    return questions

all_questions = extract_all_questions(pdf_text)
print(f"Total Questions Extracted: {len(all_questions)}")

# ---------------- STEP 5: EXTRACT OPTIONS ----------------
def extract_options(question):
    question = question.replace("\n", " ")
    question = re.sub(r"\s+", " ", question)

    matches = re.findall(r"[a-d]\)\s*([^a-d]+?)(?=\s*[a-d]\)|$)", question)
    options = [opt.strip() for opt in matches]

    return options

# ---------------- STEP 6: DIVIDE INTO SECTIONS ----------------
sections = {"A": [], "B": [], "C": [], "D": []}

for num, q in all_questions:
    if 1 <= num <= 15:
        sections["A"].append(q)
    elif 16 <= num <= 22:
        sections["B"].append(q)
    elif 23 <= num <= 31:
        sections["C"].append(q)
    elif 32 <= num <= 34:
        sections["D"].append(q)

# ---------------- STEP 7: DISPLAY COUNT ----------------
print("\nQUESTION COUNT:\n")
for sec in sections:
    print(f"Section {sec}: {len(sections[sec])}")

# ---------------- STEP 8: PREVIEW ----------------
print("\nFIRST 3 QUESTIONS FROM SECTION A:\n")
for i, q in enumerate(sections["A"][:3], 1):
    print(f"{i}. {q}\n")

# ---------------- STEP 9: STRUCTURED DATA ----------------
dataset = []

for sec in sections:
    for q in sections[sec]:
        dataset.append({
            "section": sec,
            "question": q,
            "options": extract_options(q) if sec == "A" else []
        })

# ---------------- STEP 10: TEST OPTIONS ----------------
print("\nTEST OPTIONS:\n")
for q in sections["A"][:3]:
    print("Question:\n", q)
    print("Options:", extract_options(q))
    print()

# ---------------- STEP 11: EMBEDDINGS ----------------
model = SentenceTransformer('all-MiniLM-L6-v2')

questions_text = [item["question"] for item in dataset]
embeddings = model.encode(questions_text)

# ---------------- STEP 12: SEARCH FUNCTION ----------------
def search_question(query, k=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]

    print("\nSEARCH RESULTS:\n")

    for i in top_indices:
        print("Section:", dataset[i]["section"])
        print("Q:", dataset[i]["question"])

        if dataset[i]["options"]:
            print("Options:", dataset[i]["options"])
        print()

# ---------------- STEP 13: TEST SEARCH ----------------
search_question("coal")

# ---------------- STEP 14: DISPLAY ALL QUESTIONS ----------------
print("\n===== ALL QUESTIONS =====\n")

for sec in sections:
    print(f"\n===== Section {sec} =====\n")
    for i, q in enumerate(sections[sec], 1):
        print(f"{i}. {q}\n")

# ---------------- STEP 15: SIDE-BY-SIDE VIEW ----------------
def display_side_by_side(questions):
    half = len(questions)//2
    col1 = questions[:half]
    col2 = questions[half:]

    for q1, q2 in zip_longest(col1, col2, fillvalue=""):
        print(f"{q1:<80} | {q2}")

print("\nSIDE BY SIDE VIEW (Section A):\n")
display_side_by_side(sections["A"])

def extract_solutions(text):
    pattern = r"(\d+)\.\s*(.*?)(?=\n\d+\.|\Z)"
    matches = re.findall(pattern, text, re.S)

    solutions = {}

    for num, sol in matches:
        solutions[int(num)] = sol.strip()

    return solutions

solution_text = extract_pdf_text("Dataset2.pdf")
solution_text = clean_text(solution_text)

solutions_dict = extract_solutions(solution_text)

print("Total Solutions:", len(solutions_dict))

qa_dataset = []

for num, q in all_questions:
    answer = solutions_dict.get(num, "No Answer Found")

    qa_dataset.append({
        "question_no": num,
        "question": q,
        "answer": answer,
        "options": extract_options(q)
    })
    
questions_text = [item["question"] for item in qa_dataset]
embeddings = model.encode(questions_text)

def ask_question(query, k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]

    print("\n🔍 BEST MATCHED ANSWERS:\n")

    for i in top_indices:
        print("Question:", qa_dataset[i]["question"])
        print("Answer:", qa_dataset[i]["answer"])

        if qa_dataset[i]["options"]:
            print("Options:", qa_dataset[i]["options"])

        print("-" * 50)

ask_question("pasteurisation")
ask_question("coal types")
ask_question("electrolysis")
