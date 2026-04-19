import os
import time
import numpy as np
from Levenshtein import distance as lev_distance

def plot_latency_breakdown():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Skipping plotting due to Matplotlib/Windows OS Error: {e}")
        return
        
    # Example Sample Data (you should replace this with real timing logs from your pipeline runs)
    labels = ['Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5']
    
    # Mock data representing time spent in each pipeline stage in seconds
    img_preprocessing = np.array([0.5, 0.6, 0.4, 0.7, 0.5])
    easy_ocr_layout = np.array([1.2, 1.5, 1.1, 1.3, 1.2])
    trocr_extraction = np.array([4.5, 5.0, 4.2, 6.1, 4.8])
    tinyllama_grading = np.array([3.2, 3.5, 3.0, 3.8, 3.4])

    width = 0.5
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(labels, img_preprocessing, width, label='Preprocessing (CLAHE)', color='#4c72b0')
    ax.bar(labels, easy_ocr_layout, width, bottom=img_preprocessing, label='Layout Detection (EasyOCR)', color='#dd8452')
    ax.bar(labels, trocr_extraction, width, bottom=img_preprocessing+easy_ocr_layout, label='Handwriting OCR (TrOCR)', color='#55a868')
    ax.bar(labels, tinyllama_grading, width, bottom=img_preprocessing+easy_ocr_layout+trocr_extraction, label='Feedback Gen (TinyLlama)', color='#c44e52')

    ax.set_ylabel('Execution Time (Seconds)', fontsize=12)
    ax.set_title('Pipeline Component Latency Breakdown', fontsize=14, fontweight='bold')
    
    # Grid and styling
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Put legend outside
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    output_path = 'pipeline_latency_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Done] Saved latency visualization to '{output_path}'")

def compute_cer(ground_truth_list, ocr_text_list):
    """
    Computes average Character Error Rate (CER).
    """
    total_cer = 0.0
    valid_samples = 0
    
    print("\n--- OCR Text Evaluation ---")
    
    for i, (gt, ocr) in enumerate(zip(ground_truth_list, ocr_text_list)):
        gt_clean = str(gt).strip()
        ocr_clean = str(ocr).strip()
        
        if len(gt_clean) == 0:
            continue
            
        distances = lev_distance(gt_clean, ocr_clean)
        cer = (distances / max(1, len(gt_clean))) * 100.0
        total_cer += cer
        valid_samples += 1
        
        print(f"\nSample {i+1}:")
        print(f"Ground Truth: '{gt_clean}'")
        # Safe print for Windows terminal unicode issues
        safe_ocr = ocr_clean.encode('ascii', errors='replace').decode('ascii')
        print(f"OCR Output:   '{safe_ocr}'")
        print(f"CER:          {cer:.2f}%")
        
    avg_cer = total_cer / valid_samples if valid_samples > 0 else 0
    print(f"\n=> Average Character Error Rate (CER): {avg_cer:.2f}%")
    print("   (Under 5% is excellent for handwriting, under 15% is good)")

if __name__ == "__main__":
    print("Generating Visualizations & Metrics for the Ankya AI Report...\n")
    
    # 1. Latency Chart Generation
    plot_latency_breakdown()
    
    # 2. Text Extraction Evaluation (Live run via test_extraction.py)
    from test_extraction import extract_text_from_image
    
    image_path = "examples/sample6.png"
    print(f"\n--- Running dynamic text extraction on {image_path} ---")
    
    try:
        # Run the extraction pipeline using TrOCR
        result = extract_text_from_image(image_path, use_trocr=True)
        extracted_text = result.get("full_text", "")
    except Exception as e:
        print(f"Warning: Extraction failed ({e}). Please ensure {image_path} exists.")
        extracted_text = ""
    
    # Ground truth manually transcribed from sample6.png
    ground_truth = (
        "During the spring or rainy season, frogs and toads migrate to ponds and slow-moving water bodies for reproduction. "
        "The male and female come together in water, where the female lays hundred of eggs and the male releases sperm over them. "
        "In frogs, a jelly-like covering surrounds the eggs and provides protection. "
        "Since fertization occurs outside the female's body, this process is called as external fertilization. "
        "Blood pressure reaches its maximum value when the heart pumps blood; this is called systolic pressure. "
        "In a healthy adult, the normal systolic pressure is about 120 mmHg. "
        "When the heart relaxes between beats, the blood pressure decreases; this is known as diastolic pressure, which is normally about 80 mmHg in adults. "
        "Blood pressure is measured when the body is at rest and is written as systolic pressure over diastolic pressure, often without units. "
        "For ex:- A blood pressure reading of 90/60 in a 10-month-old child indicates a systolic pressure of 90mmHg and diastolic pressure of 60mmHg."
    )
    
    compute_cer([ground_truth], [extracted_text])
    
    print("\n[Complete] You can include these metrics in your report!")
