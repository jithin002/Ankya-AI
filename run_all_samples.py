import os
import subprocess

samples = [
    "examples/sample6.png",
    "examples/sample7.png",
    "examples/sample8.jpeg"
]

print("Running extraction on all samples to show current baseline...\n")

for sample in samples:
    print(f"--- Extraction for {os.path.basename(sample)} ---")
    
    # Run the test_extraction script without check=True to handle non-zero exits gracefully
    result = subprocess.run(
        ["venv\\Scripts\\python.exe", "test_extraction.py", sample],
        capture_output=True,
        text=True
    )
    
    # Parse out just the "FULL TEXT" section for concise display
    output_lines = result.stdout.splitlines() + result.stderr.splitlines()
    capture = False
    full_text = []
    
    for line in output_lines:
        if "-- FULL TEXT --" in line:
            capture = True
            continue
        if "-- SEGMENTS --" in line:
            break
        if capture and line.strip():
            full_text.append(line.strip())
            
    if full_text:
        print("\n".join(full_text))
    else:
        print("No text extracted. Printing raw output for debugging:")
        print(result.stdout)
        print(result.stderr)
        
    print("\n" + "="*50 + "\n")
