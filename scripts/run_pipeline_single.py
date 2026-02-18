import argparse
import json
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, help="v1 or v3")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--ref_json", required=True)
    args = parser.parse_args()

    ref = json.loads(args.ref_json)

    # Force CPU if CUDA_VISIBLE_DEVICES is empty (monkey patch)
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        import torch
        torch.cuda.is_available = lambda: False
        print("Forced CPU mode via monkey-patch.")

    if args.pipeline == "v1":
        # Import v1
        import pipelines.icr_pipeline as v1
        if hasattr(v1, "OCR_READER") and hasattr(v1.OCR_READER, "device"):
             if v1.OCR_READER.device == "cuda" and not torch.cuda.is_available():
                 print("Reloading EasyOCR for CPU...")
                 import easyocr
                 v1.OCR_READER = easyocr.Reader(['en'], gpu=False)
        from pipelines.icr_pipeline import evaluate_from_image
        
    elif args.pipeline == "v2":
        # Import v2
        from pipelines.icr_pipeline2 import evaluate_from_image
        
    elif args.pipeline == "v3":
        # Import v3
        from pipelines.icr_pipeline3 import evaluate_from_image
    else:
        raise ValueError(f"Unknown pipeline version: {args.pipeline}")

    # Run
    result = evaluate_from_image(args.image, args.question, ref)
    
    # Print JSON to stdout (ensure it's the last thing printed)
    print("\nXXX_JSON_START_XXX")
    print(json.dumps(result))
    print("XXX_JSON_END_XXX")

if __name__ == "__main__":
    main()
