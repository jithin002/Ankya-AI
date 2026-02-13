# scripts/generate_line_crops.py
import os, csv, argparse
import cv2
from pipelines.icr_pipeline3 import ocr_image_to_blocks, normalize_text  # uses your existing pipeline
from PIL import Image

def main(image_dir, out_dir, csv_out):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    idx = 0
    img_files = [os.path.join(image_dir,f) for f in os.listdir(image_dir) if f.lower().endswith((".jpg",".png",".jpeg"))]
    for img_path in sorted(img_files):
        blocks = ocr_image_to_blocks(img_path, save_debug_image=False)  # outputs blocks with bbox(tiny)
        img_cv = cv2.imread(img_path)
        h,w = img_cv.shape[:2]
        for b in blocks:
            x,y,bbw,bbh = [int(v) for v in b['bbox']]
            # expand a little
            pad = int(0.02 * h)
            x0 = max(0, x-pad); y0 = max(0, y-pad)
            x1 = min(w, x + bbw + pad); y1 = min(h, y + bbh + pad)
            crop = img_cv[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            fname = f"line_{idx:05d}.png"
            outpath = os.path.join(out_dir, fname)
            cv2.imwrite(outpath, crop)
            suggested = normalize_text(b.get("text",""))
            rows.append([fname, suggested])
            idx += 1
    # write CSV
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename","text"])
        writer.writerows(rows)
    print("Wrote crops:", out_dir, "CSV:", csv_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out_dir", default="data/images")
    parser.add_argument("--csv_out", default="data/labels.csv")
    args = parser.parse_args()
    main(args.image_dir, args.out_dir, args.csv_out)
