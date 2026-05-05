import cv2, easyocr
from test_extraction import merge_nearby_horizontal_blocks

img = cv2.imread('examples/sample8.jpeg')
reader = easyocr.Reader(['en'], gpu=True)
res = reader.readtext(img, detail=1, text_threshold=0.7, low_text=0.4)
print(f'Raw EasyOCR blocks: {len(res)}')
for bbox, text, conf in sorted(res, key=lambda x: min(p[1] for p in x[0])):
    ys = [p[1] for p in bbox]
    print(f'  y={int(min(ys)):4d}  [{conf:.2f}] {text}')

blocks = []
for bbox, text, conf in res:
    xs, ys = zip(*bbox)
    x, y = min(xs), min(ys)
    w, h = max(xs)-x, max(ys)-y
    blocks.append({'text': text, 'conf': float(conf)*100, 'bbox': (x,y,w,h), 'cy': y+h/2.0})

for thresh in [20, 50, 80]:
    merged = merge_nearby_horizontal_blocks(blocks, gap_thresh=thresh)
    print(f'\ngap_thresh={thresh}: {len(merged)} blocks')
    for b in merged:
        print(f'  cy={int(b["cy"]):4d} | {b["text"]}')
