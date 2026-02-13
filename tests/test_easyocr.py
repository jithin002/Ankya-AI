import cv2, easyocr
r = easyocr.Reader(['en'], gpu=True)   # or gpu=False if CPU
img = cv2.imread("examples/sample3.jpg")
res = r.readtext(img, detail=1)
for bbox, text, conf in res[:30]:
    print(round(conf,2), text)