# easyocr_tune.py
import easyocr
from pipelines.icr_pipeline2 import preprocess_image_for_ocr

IMG = "examples/sample3.jpg"
# try changing contrast_ths and adjust_contrast
reader_default = easyocr.Reader(["en"], gpu=True)
reader_tuned = easyocr.Reader(["en"], gpu=True, contrast_ths=0.05, adjust_contrast=0.7)

print("Default EasyOCR:")
res = reader_default.readtext(IMG, detail=1)
print(" avg:", sum([c for _,_,c in res])/max(1,len(res)), "boxes:", len(res))
for _,t,c in res[:20]:
    print(round(c,2), t[:120])

print("\nTuned EasyOCR (contrast_ths=0.05, adjust_contrast=0.7):")
res2 = reader_tuned.readtext(IMG, detail=1)
print(" avg:", sum([c for _,_,c in res2])/max(1,len(res2)), "boxes:", len(res2))
for _,t,c in res2[:20]:
    print(round(c,2), t[:120])

print("\nTuned EasyOCR on preprocessed image:")
proc = preprocess_image_for_ocr(IMG, save_tmp=True)
print("preprocessed path", proc)
res3 = reader_tuned.readtext(proc, detail=1)
print(" avg:", sum([c for _,_,c in res3])/max(1,len(res3)), "boxes:", len(res3))
for _,t,c in res3[:20]:
    print(round(c,2), t[:120])
