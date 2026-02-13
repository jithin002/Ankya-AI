# run_trocr.py
"""import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

MODEL_NAME = "microsoft/trocr-small-handwritten"  # try this first
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def ocr_trocr(path):
    image = Image.open(path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated = model.generate(pixel_values, max_new_tokens=256, num_beams=4, do_sample=False)
    return tokenizer.decode(generated[0], skip_special_tokens=True)

if __name__ == "__main__":
    path = "examples/sample3.jpg"
    try:
        out = ocr_trocr(path)
        print("TrOCR output:\n", out)
    except Exception as e:
        print("TrOCR error:", e)
"""
"""# tests/run_trocr.py
from PIL import Image
import cv2, sys
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# load pretrained trocr small-handwritten (or trocr-base if you have memory)
model_name = "microsoft/trocr-small-handwritten"  # try this first
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)

img_path = "examples/sample3.jpg"
img = Image.open(img_path).convert("RGB")

# If your image is a full page, it's best to split into lines before HTR; 
# this simple test runs whole page to see baseline:
pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
with torch.no_grad():
    generated_ids = model.generate(pixel_values, num_beams=4, max_new_tokens=256)
pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("TrOCR prediction:\n", pred)"""


# tests/run_trocr_on_crops.py
import cv2
from PIL import Image
import torch
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# model (small handwritten)
MODEL = "microsoft/trocr-small-handwritten"
processor = TrOCRProcessor.from_pretrained(MODEL)
model = VisionEncoderDecoderModel.from_pretrained(MODEL).to(device)
model.eval()

# EasyOCR detector (use GPU if available)
reader = easyocr.Reader(['en'], gpu=(device=="cuda"))

img_path = "examples/sample4.jpg"
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise FileNotFoundError(img_path)

# detect text lines/boxes using EasyOCR
detections = reader.readtext(img_bgr, detail=1)  # list of (bbox, text, conf)
print(f"Detected {len(detections)} boxes by EasyOCR (detection only).")

results = []
for i, (bbox, _, conf) in enumerate(detections):
    # bbox: 4 points; get bounding rect with padding
    xs = [int(p[0]) for p in bbox]; ys = [int(p[1]) for p in bbox]
    x0, x1 = max(min(xs)-8,0), min(max(xs)+8, img_bgr.shape[1])
    y0, y1 = max(min(ys)-6,0), min(max(ys)+6, img_bgr.shape[0])
    crop = img_bgr[y0:y1, x0:x1]
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    # optionally resize slightly for trocr
    w,h = pil.size
    max_w = 1000
    if w > max_w:
        pil = pil.resize((max_w, int(max_w*h/w)), Image.BICUBIC)
    # TrOCR inference (beam size small for low VRAM)
    with torch.no_grad():
        pv = processor(images=pil, return_tensors="pt").pixel_values.to(device)
        ids = model.generate(pv, num_beams=2, max_new_tokens=192)
    pred = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    results.append({"box":(x0,y0,x1,y1),"pred":pred,"det_conf":conf})
    print(f"[{i}] bbox={x0,y0,x1,y1} det_conf={conf:.2f} -> TrOCR: {pred}")

# join results by reading order (sort by y)
results_sorted = sorted(results, key=lambda r: r["box"][1])
joined = " ".join(r["pred"] for r in results_sorted)
print("\n--- Final assembled text ---\n", joined)

# free GPU
torch.cuda.empty_cache()


