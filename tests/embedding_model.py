from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
a = m.encode(["Newton's laws of motion"], convert_to_numpy=True)
b = m.encode(["Newton's first law: inertia"], convert_to_numpy=True)
import numpy as np
print("cos:", (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b)))


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1").to("cpu")
s = "This is bad grammer sentence."
inp = tokenizer.encode(s, return_tensors="pt", truncation=True)
out = model.generate(inp, max_new_tokens=128)
print(tokenizer.decode(out[0], skip_special_tokens=True))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to("cpu")
prompt = "Return only JSON: {\"recommended_marks\": 5}\nQuestion: What is X?\nStudent: ... \n"
inp = tokenizer(prompt, return_tensors="pt", truncation=True)
out = model.generate(**inp, max_new_tokens=200, do_sample=False, num_beams=2)
print(tokenizer.decode(out[0], skip_special_tokens=True))