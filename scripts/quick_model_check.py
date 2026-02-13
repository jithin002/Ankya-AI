"""
Quick test to verify TinyLlama model loads and check VRAM usage.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def test_model_loading():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Initial VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    print("\n Loading TinyLlama-1.1B-Chat...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        print("✓ Model loaded successfully!")
        
        if device == "cuda":
            vram_used = torch.cuda.memory_allocated() / 1024**3
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"VRAM Used: {vram_used:.2f} GB / {vram_total:.2f} GB")
            print(f"VRAM Available: {vram_total - vram_used:.2f} GB")
        
        # Test generation
        print("\nTesting generation...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in JSON format: {\"message\": \"...\"}"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"Model response: {response}")
        print("\n✓ All tests passed!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    exit(0 if success else 1)
