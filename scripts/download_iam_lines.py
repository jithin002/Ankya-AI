# scripts/download_iam_lines.py
# Usage:
#   python scripts/download_iam_lines.py --out_dir data --subset 800
#
import os
import csv
import argparse
from datasets import load_dataset
from PIL import Image
import io

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="data", help="output dir for images and labels.csv")
parser.add_argument("--subset", type=int, default=None, help="if set, randomly sample this many examples")
parser.add_argument("--split", default="train", help="which split to use (train/validation/test) - depends on HF dataset")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
images_dir = os.path.join(args.out_dir, "images")
os.makedirs(images_dir, exist_ok=True)
csv_path = os.path.join(args.out_dir, "labels.csv")

print("Loading Teklia/IAM-line from Hugging Face...")
ds = load_dataset("Teklia/IAM-line")   # HF prepared line-level dataset
# try to pick the split requested
if args.split not in ds:
    # fallback to whole concatenated dataset if split not found
    combined = ds["train"].select(range(len(ds["train"]))) if "train" in ds else ds["train"]
else:
    combined = ds[args.split]

n = len(combined)
print(f"Dataset split '{args.split}' size: {n}")

# optionally sample
import random
indices = list(range(n))
if args.subset is not None and args.subset < n:
    random.shuffle(indices)
    indices = indices[: args.subset]

rows = []
count = 0
for i in indices:
    item = combined[i]
    # item may contain an 'image' column as image object or bytes
    # adapt depending on dataset format
    img = item.get("image", None) or item.get("img", None)
    text = item.get("text", "") or item.get("words", "") or item.get("labels", "") or item.get("sentence", "")
    if img is None:
        continue
    # HF image: either PIL.Image.Image (-> save direct) or bytes
    if hasattr(img, "to_pil"):
        pil = img.to_pil()
    elif isinstance(img, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(img)).convert("RGB")
    elif isinstance(img, Image.Image):
        pil = img
    else:
        # try converting from numpy array
        try:
            import numpy as np
            pil = Image.fromarray(img)
        except Exception:
            continue
    fname = f"iam_line_{count:06d}.png"
    fpath = os.path.join(images_dir, fname)
    pil.save(fpath, format="PNG")
    # normalize text to single-line
    text = str(text).replace("\n", " ").strip()
    rows.append([fname, text])
    count += 1
    if count % 100 == 0:
        print("saved", count)

print("Writing CSV:", csv_path)
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename","text"])
    writer.writerows(rows)

print("Done. Saved", count, "images to", images_dir)
