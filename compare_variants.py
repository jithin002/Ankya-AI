# compare_variants.py
import cv2, easyocr, numpy as np, tempfile
from PIL import Image
from pipelines.icr_pipeline2 import deskew_image  # reuse your deskew if available

IMG = "examples/sample3.jpg"
reader = easyocr.Reader(["en"], gpu=True)

def save_temp(img_gray):
    bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(tmp.name, bgr)
    return tmp.name

def easy_summary(path):
    res = reader.readtext(path, detail=1)
    avg = sum([c for _,_,c in res]) / max(1,len(res))
    print(f"{path} => avg_conf={avg:.2f} boxes={len(res)}")
    for _,t,c in res[:12]:
        print(" ", round(c,2), t[:120])
    return avg, res

# Variant 0: original
print("=== ORIGINAL ===")
easy_summary(IMG)

img = cv2.imread(IMG)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Variant A: simple (old pipeline style)
def preprocess_simple(gray):
    # bilateral + deskew + adaptive threshold + small open + upscale 1.6
    den = cv2.bilateralFilter(gray, 9, 75, 75)
    deskewed, _ = deskew_image(den)
    try:
        th = cv2.adaptiveThreshold(deskewed,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,8)
    except:
        _, th = cv2.threshold(deskewed,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    cleaned = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    up = cv2.resize(cleaned, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    return up

# Variant B: no line removal (new pipeline minus line inpainting)
def preprocess_no_line_removal(gray):
    # CLAHE + median blur + deskew + adaptive threshold + closing small + upscale 1.6
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g = cv2.medianBlur(g, 3)
    deskewed, _ = deskew_image(g)
    try:
        th = cv2.adaptiveThreshold(deskewed,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,25,8)
    except:
        _, th = cv2.threshold(deskewed,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    up = cv2.resize(closed, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    return up

# Variant C: milder (less aggressive CLAHE/closing + smaller upscale)
def preprocess_milder(gray):
    g = cv2.medianBlur(gray, 3)
    deskewed, _ = deskew_image(g)
    _, th = cv2.threshold(deskewed,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    # use opening to remove speckle but avoid closing
    cleaned = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    up = cv2.resize(cleaned, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
    return up

for name, fn in [("simple", preprocess_simple), ("no_line_removal", preprocess_no_line_removal), ("milder", preprocess_milder)]:
    print(f"\n=== VARIANT: {name} ===")
    pre = fn(gray)
    tmp = save_temp(pre)
    easy_summary(tmp)
