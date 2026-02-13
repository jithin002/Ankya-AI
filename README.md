# Ankya-AI: Intelligent Character Recognition (ICR) & Grading Pipeline

## Overview

Ankya-AI is an advanced Intelligent Character Recognition (ICR) and automated grading system designed to evaluate handwritten student answers. It leverages a multi-stage pipeline combining state-of-the-art OCR technologies, semantic analysis, and Large Language Models (LLMs) to provide accurate scores and detailed feedback.

This repository features a highly optimized pipeline (`pipelines/icr_pipeline3.py`) capable of running on consumer hardware by utilizing lazy loading and efficient memory management.

## Key Features

*   **Hybrid OCR Engine**: Combines **EasyOCR** for layout detection, **TrOCR** (Transformer OCR) for high-accuracy handwriting recognition, and **Tesseract** for fallback ensemble voting.
*   **Smart Memory Management**: Implements a `ModelManager` with lazy loading and automatic garbage collection to run heavy models (OCR, LLMs, Transformers) sequentially on limited GPU memory (e.g., 4GB VRAM).
*   **Multi-Dimensional Grading**:
    *   **Keyword Matching**: Fuzzy matching for essential terms.
    *   **Semantic Similarity**: Uses `SentenceTransformer` to measure meaning against reference answers.
    *   **Grammar Analysis**: AI-based grammar error correction scoring.
    *   **Content Coverage**: Verifies coverage of key reference points.
    *   **Presentation**: Heuristic scoring based on answer length and structure.
*   **LLM Integration**: Uses **TinyLlama-1.1B-Chat** to generate human-like feedback and structured grading justifications in JSON format.

## Pipeline Architecture

The core logic resides in `pipelines/icr_pipeline3.py` and follows this workflow:

1.  **Image Preprocessing**: Deskewing, denoising, and adaptive thresholding.
2.  **Text Extraction (OCR)**:
    *   Layouts detected via EasyOCR.
    *   Text recognized line-by-line using TrOCR.
    *   Results merged and post-processed.
3.  **Deterministic Scoring**: Calculates scores for keywords, semantics, grammar, and coverage.
4.  **LLM Evaluation**: The extracted text and grading rubrics are sent to TinyLlama to generate a final recommended mark and explanation.
5.  **Result Aggregation**: Combines all scores into a final report.

## Installation

Ensure you have Python installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

*Note: You may need to install `tesseract-ocr` separately on your system.*

## Usage

The main entry point for the pipeline is the `evaluate_from_image` function in `pipelines/icr_pipeline3.py`.

### Example

```python
from pipelines.icr_pipeline3 import evaluate_from_image

# Define references
references = {
    "max_marks": 10,
    "keywords": [{"term": "Newton", "weight": 1.0}],
    "reference_long": "Newton's first law of motion states that...",
    "reference_short": ["inertia", "rest", "motion"]
}

# Run the pipeline
result = evaluate_from_image(
    image_path="path/to/student_answer.jpg",
    question="Explain Newton's First Law",
    reference=references
)

print(result)
```

## Configuration

The pipeline automatically detects if CUDA is available. You can adjust model paths and settings in the `Config / Globals` section of `icr_pipeline3.py`.

*   **Models Used**:
    *   OCR: `microsoft/trocr-small-handwritten`, `EasyOCR`
    *   Embeddings: `all-MiniLM-L6-v2`
    *   Grammar: `prithivida/grammar_error_correcter_v1`
    *   LLM: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## License

[MIT License](LICENSE)
