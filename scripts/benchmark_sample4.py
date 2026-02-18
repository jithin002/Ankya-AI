import subprocess
import json
import os
import sys
import Levenshtein
import time

# Constants for Sample 4 (Newton)
IMAGE_PATH = os.path.join("examples", "sample4.jpg")
QUESTION = "Explain Newton's first law of motion."
REF_DATA = {
    "reference_long": "Newton's first law states that an object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force (inertia).",
    "reference_short": ["inertia", "object at rest stays at rest", "object in motion stays in motion", "external force"],
    "keywords": [{"term": "Newton", "weight": 1.0}, {"term": "Inertia", "weight": 1.0}, {"term": "Force", "weight": 1.0}, {"term": "Motion", "weight": 1.0}, {"term": "Rest", "weight": 1.0}],
    "max_marks": 10
}
HUMAN_MARKS = 9.0

# Ground Truth Text for Sample 4 (Manual Transcription)
GROUND_TRUTH_TEXT = "First law of motion (Law of Inertia). An object at rest stays at rest, and an object in motion continues in uniform motion in a straight line unless acted upon by an external force. Second law of motion. Rate of change of momentum of an object is directly proportional to the applied force. F=ma. F-force, m-mass, a-acceleration. Third law of Motion. For every action, there is an equal and opposite reaction. When you apply a force on an object, it applies an equal force back in the opposite direction."

def calculate_wer(ref, hyp):
    r = ref.split()
    h = hyp.split()
    d = Levenshtein.distance(r, h)
    return d / len(r) if len(r) > 0 else 0.0

def calculate_cer(ref, hyp):
    d = Levenshtein.distance(ref, hyp)
    return d / len(ref) if len(ref) > 0 else 0.0

def run_pipeline(version, force_cpu=False):
    print(f"Running Pipeline {version} (CPU={force_cpu})...")
    
    env = os.environ.copy()
    if force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPU
    
    cmd = [
        sys.executable, # Use same python as this script (venv hopefully)
        os.path.join("scripts", "run_pipeline_single.py"),
        "--pipeline", version,
        "--image", IMAGE_PATH,
        "--question", QUESTION,
        "--ref_json", json.dumps(REF_DATA)
    ]
    
    start_time = time.time()
    # Capture output
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Error running {version}:")
        print(result.stderr)
        return None, elapsed
    
    # Parse output by looking for the marker
    content = result.stdout
    data = None
    try:
        start_marker = "XXX_JSON_START_XXX"
        end_marker = "XXX_JSON_END_XXX"
        
        if start_marker in content and end_marker in content:
            json_str = content.split(start_marker)[1].split(end_marker)[0].strip()
            data = json.loads(json_str)
        else:
            print(f"Could not find JSON markers in {version} output.")
            # Fallback: try last line
            lines = content.strip().split("\n")
            data = json.loads(lines[-1])
    except Exception as e:
        print(f"Failed to parse JSON from {version}: {e}")
        # print("Stdout was:", content) # Debug
        
    return data, elapsed

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: {IMAGE_PATH} not found.")
        return

    # Run V2 -> Optimized, can use GPU (Sequential execution prevents OOM)
    print("--- Starting V2 (Intermediate Pipeline) [GPU ALLOWED] ---")
    res_v2, time_v2 = run_pipeline("v2", force_cpu=False)
    
    # Run V3 -> Optimized, can use GPU
    print("--- Starting V3 (New Pipeline) [GPU ALLOWED] ---")
    res_v3, time_v3 = run_pipeline("v3", force_cpu=False)

    # Calculate metrics
    metrics_v2 = {}
    if res_v2:
        txt = res_v2.get("student_text", "")
        metrics_v2["wer"] = calculate_wer(GROUND_TRUTH_TEXT, txt)
        metrics_v2["cer"] = calculate_cer(GROUND_TRUTH_TEXT, txt)
        metrics_v2["time"] = time_v2
        metrics_v2["marks"] = res_v2.get("deterministic_recommended_marks", 0)
        metrics_v2["text"] = txt

    metrics_v3 = {}
    if res_v3:
        txt = res_v3.get("student_text", "")
        metrics_v3["wer"] = calculate_wer(GROUND_TRUTH_TEXT, txt)
        metrics_v3["cer"] = calculate_cer(GROUND_TRUTH_TEXT, txt)
        metrics_v3["time"] = time_v3
        metrics_v3["marks"] = res_v3.get("deterministic_recommended_marks", 0)
        metrics_v3["text"] = txt

    output = {
        "ground_truth_text": GROUND_TRUTH_TEXT,
        "human_marks": HUMAN_MARKS,
        "v2": metrics_v2,
        "v3": metrics_v3
    }
    
    out_file = "benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nBenchmark complete. Saved to {out_file}")
    print("V2 WER:", metrics_v2.get("wer"))
    print("V3 WER:", metrics_v3.get("wer"))

if __name__ == "__main__":
    main()
