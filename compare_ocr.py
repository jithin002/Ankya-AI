# compare_ocr.py
import easyocr
from PIL import Image
import pytesseract
from pipelines.icr_pipeline2 import preprocess_image_for_ocr, OCR_READER
import numpy as np

IMG = "examples/sample3.jpg"

def easyocr_summary(img_path, reader, top=30):
    res = reader.readtext(img_path, detail=1)
    avg = sum([c for _,_,c in res]) / max(1, len(res))
    print(f"EasyOCR on {img_path} | avg_conf={avg:.2f} | boxes={len(res)}")
    for _,t,c in res[:top]:
        print(f"  {round(c,2)}  {t[:120]}")
    return res

print("1) EasyOCR on original")
r = easyocr.Reader(["en"], gpu=True)   # set gpu=False if no CUDA
easyocr_summary(IMG, r)

print("\n2) Tesseract raw (if installed)")
try:
    txt = pytesseract.image_to_string(Image.open(IMG), lang='eng', config="--psm 3")
    print(txt[:1000])
except Exception as e:
    print("Tesseract error:", e)

print("\n3) Preprocess -> EasyOCR")
proc = preprocess_image_for_ocr(IMG, save_tmp=True)
print("preprocessed saved at:", proc)
easyocr_summary(proc, r)
