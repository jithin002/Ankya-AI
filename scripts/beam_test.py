# scripts/beam_test.py
"""Test TrOCR per-line recognition with different beam sizes on a single image.
"""
import os
import sys

# ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipelines.icr_pipeline3 import trocr_init, recognize_with_trocr_per_line, easyocr_blocks_from_image
import cv2

if len(sys.argv) < 2:
    print("Usage: python beam_test.py <image_path>")
    sys.exit(1)

img = sys.argv[1]
if not os.path.exists(img):
    print("Image not found:", img)
    sys.exit(1)

# init trocr
trocr_init()
img_cv = cv2.imread(img)
from pipelines.icr_pipeline3 import OCR_READER
blocks = easyocr_blocks_from_image(img_cv, OCR_READER)

for beams in [1, 4, 8, 12, 16]:
    joined, per_line = recognize_with_trocr_per_line(img, blocks, pad=6, num_beams=beams)
    print(f"--- beams={beams} ---")
    print(joined)
    print()
