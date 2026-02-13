import os
import sys
import time
import torch
import gc

def print_memory(step_name):
    print(f"\n--- {step_name} ---")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"CUDA Allocated: {allocated:.2f} GB")
        print(f"CUDA Reserved:  {reserved:.2f} GB")
    else:
        print("CUDA not available")
    
    # Try to print RAM usage
    try:
        import psutil
        process = psutil.Process(os.getpid())
        ram_gb = process.memory_info().rss / 1024**3
        print(f"System RAM (Process): {ram_gb:.2f} GB")
        virtual = psutil.virtual_memory()
        print(f"System RAM (Total Used): {virtual.percent}% ({virtual.used / 1024**3:.2f} / {virtual.total / 1024**3:.2f} GB)")
    except ImportError:
        print("psutil not installed, cannot measure RAM")

print("Starting diagnosis...")
print_memory("Start")

print("\n1. Loading EasyOCR...")
try:
    import easyocr
    # Force GPU if available to test VRAM usage
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    print("EasyOCR loaded.")
except Exception as e:
    print(f"FAILED to load EasyOCR: {e}")
print_memory("After EasyOCR")

print("\n2. Loading SentenceTransformer...")
try:
    from sentence_transformers import SentenceTransformer
    sent_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda" if torch.cuda.is_available() else "cpu")
    print("SentenceTransformer loaded.")
except Exception as e:
    print(f"FAILED to load SentenceTransformer: {e}")
print_memory("After SentenceTransformer")

print("\n3. Loading TinyLlama...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    print("TinyLlama loaded.")
except Exception as e:
    print(f"FAILED to load TinyLlama: {e}")
print_memory("After TinyLlama")

print("\n4. Loading Grammar Model...")
try:
    from transformers import AutoModelForSeq2SeqLM
    g_name = "prithivida/grammar_error_correcter_v1"
    g_tokenizer = AutoTokenizer.from_pretrained(g_name)
    g_model = AutoModelForSeq2SeqLM.from_pretrained(g_name).to("cuda" if torch.cuda.is_available() else "cpu")
    print("Grammar Model loaded.")
except Exception as e:
    print(f"FAILED to load Grammar Model: {e}")
print_memory("After Grammar Model")

print("\n5. Initializing TrOCR (Lazy Init Simulation)...")
try:
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor
    # Simulating what happens when TrOCR is needed
    trocr_name = "microsoft/trocr-small-handwritten"
    trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_name).to("cuda" if torch.cuda.is_available() else "cpu")
    trocr_processor = ViTImageProcessor.from_pretrained(trocr_name)
    print("TrOCR loaded.")
except Exception as e:
    print(f"FAILED to load TrOCR: {e}")
print_memory("After TrOCR")

print("\nDiagnosis complete.")
