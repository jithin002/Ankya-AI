# tests/test_tesseract.py
import pytesseract, cv2
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # edit if needed
img = cv2.imread("examples/sample3.jpg")
print(pytesseract.image_to_string(img)[:400])
