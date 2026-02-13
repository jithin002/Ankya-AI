#!/usr/bin/env python3
"""
analysis/plot_figures_no_human.py

Produces basic visualizations from eval_records.jsonl WITHOUT requiring human marks.

Run:
    python analysis/plot_figures_no_human.py
"""

import os
import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Try to import the visualize helper from your pipeline (optional)
try:
    from pipelines.icr_pipeline3 import visualize_blocks_on_image
    HAVE_VISUALIZE = True
except Exception:
    HAVE_VISUALIZE = False

# --- Config ---
OUT_DIR = os.path.join("analysis", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Try several likely locations for eval_records.jsonl
POSSIBLE_PATHS = [
    "eval_records.jsonl",
    os.path.join("..", "eval_records.jsonl"),  # if script run in analysis/
    os.path.join(os.path.dirname(__file__), "..", "eval_records.jsonl") if "__file__" in globals() else None
]
POSSIBLE_PATHS = [p for p in POSSIBLE_PATHS if p]

def find_records_file():
    for p in POSSIBLE_PATHS:
        if p and os.path.exists(p):
            return p
    return None

def load_records(path):
    recs = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] failed to parse line {i}: {e}")
    return recs

# --- plotting helpers (one plot per file) ---
def save_and_show(fig, fname):
    out = os.path.join(OUT_DIR, fname)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print("[SAVED]", out)
    plt.close(fig)

def plot_ocr_conf_hist(records):
    confs = [r.get("avg_ocr_conf_raw", None) for r in records]
    confs = [c for c in confs if c is not None]
    if not confs:
        print("No OCR confidence values found; skipping OCR histogram.")
        return
    fig = plt.figure(figsize=(6,4))
    plt.hist(confs, bins=20)
    plt.xlabel("Avg OCR confidence (0-100)")
    plt.ylabel("Number of answers")
    plt.title("Distribution of OCR confidence")
    save_and_show(fig, "fig_ocr_conf_hist.png")

def plot_routing_bar(records):
    routes = [r.get("route", "unknown") for r in records]
    if not routes:
        print("No routing info; skipping routing plot.")
        return
    cnt = Counter(routes)
    labels = list(cnt.keys())
    vals = [cnt[k] for k in labels]
    fig = plt.figure(figsize=(6,4))
    plt.bar(labels, vals)
    plt.xlabel("Routing decision")
    plt.ylabel("Count")
    plt.title("Routing outcomes")
    save_and_show(fig, "fig_routing.png")

def plot_component_boxplots(records):
    keys = ['keyword_pct','semantic_pct','grammar_pct','coverage_pct','presentation_pct']
    data = []
    for k in keys:
        vals = []
        for r in records:
            cs = r.get("component_scores") or {}
            v = cs.get(k)
            if v is not None:
                vals.append(float(v))
        data.append(vals)
    # ensure at least one value exists
    if not any(data):
        print("No component score data present; skipping component boxplots.")
        return
    fig = plt.figure(figsize=(8,4))
    # transform data to only present columns
    present = [d for d in data if d]
    labels = [k for k, d in zip(keys, data) if d]
    plt.boxplot(present, labels=labels)
    plt.ylabel("Percent (%)")
    plt.title("Distribution of component scores")
    save_and_show(fig, "fig_component_boxplots.png")

def plot_composite_conf_hist(records):
    comps = [r.get("composite_confidence", None) for r in records]
    comps = [c for c in comps if c is not None]
    if not comps:
        print("No composite_confidence values; skipping.")
        return
    fig = plt.figure(figsize=(6,4))
    plt.hist(comps, bins=20)
    plt.xlabel("Composite confidence (0-1)")
    plt.ylabel("Count")
    plt.title("Composite confidence distribution")
    save_and_show(fig, "fig_composite_conf_hist.png")

def plot_predicted_marks_hist(records):
    preds = [r.get("deterministic_recommended_marks", None) for r in records]
    preds = [p for p in preds if p is not None]
    if not preds:
        print("No predicted marks found; skipping predicted marks histogram.")
        return
    fig = plt.figure(figsize=(6,4))
    plt.hist(preds, bins=20)
    plt.xlabel("Predicted marks")
    plt.ylabel("Count")
    plt.title("Distribution of deterministic predicted marks")
    save_and_show(fig, "fig_predicted_marks_hist.png")

def plot_keyword_vs_semantic(records):
    xs = []
    ys = []
    for r in records:
        cs = r.get("component_scores") or {}
        k = cs.get("keyword_pct")
        s = cs.get("semantic_pct")
        if k is not None and s is not None:
            xs.append(float(k)); ys.append(float(s))
    if not xs:
        print("No keyword/semantic pairs; skipping scatter.")
        return
    fig = plt.figure(figsize=(6,6))
    plt.scatter(xs, ys, alpha=0.7)
    plt.xlabel("Keyword %")
    plt.ylabel("Semantic %")
    plt.title("Keyword vs Semantic component scores")
    plt.grid(True, linestyle=':', alpha=0.5)
    save_and_show(fig, "fig_keyword_vs_semantic.png")

def plot_length_vs_finalpct(records):
    xs = []
    ys = []
    for r in records:
        txt = r.get("student_text", "") or ""
        final = r.get("deterministic_final_pct", None)
        if final is not None:
            tok = len([t for t in txt.split() if t.strip()])
            xs.append(tok); ys.append(float(final))
    if not xs:
        print("No text length/final pct data; skipping.")
        return
    fig = plt.figure(figsize=(6,4))
    plt.scatter(xs, ys, alpha=0.6)
    plt.xlabel("Student text length (tokens)")
    plt.ylabel("Deterministic final %")
    plt.title("Answer length vs final percent score")
    save_and_show(fig, "fig_length_vs_finalpct.png")

def create_qualitative_montage(records, max_examples=3):
    # Need image_path and blocks in records to annotate. Use composite_confidence for selection.
    valid = [r for r in records if r.get("image_path") and r.get("question_map") is not None]
    if not valid:
        # fallback: try any that have image_path
        valid = [r for r in records if r.get("image_path")]
    if not valid:
        print("No records with image_path; skipping montage.")
        return
    # sort by composite_confidence
    valid_sorted = sorted(valid, key=lambda r: r.get("composite_confidence", 0.0))
    sel = []
    # choose lowest, median, highest if available
    sel.append(valid_sorted[0])
    if len(valid_sorted) >= 3:
        sel.append(valid_sorted[len(valid_sorted)//2])
        sel.append(valid_sorted[-1])
    elif len(valid_sorted) == 2:
        sel.append(valid_sorted[1])
    # annotate & save individual annotated images (if visualize available)
    ann_paths = []
    for i, r in enumerate(sel):
        img_path = r.get("image_path")
        # if annotate function is available and blocks exist, use it
        out_ann = os.path.join(OUT_DIR, f"annotated_{i}.png")
        try:
            # prefer blocks from the saved record; otherwise try to run visualize with empty blocks
            blocks = r.get("question_map")  # not the same structure; try saved blocks if present
            # best-effort: if visualize_blocks_on_image exists and record contains 'student_text' and 'component_scores'
            if HAVE_VISUALIZE and isinstance(r.get("student_text", None), str):
                # We don't always have blocks in the record. If not, just copy the image.
                try:
                    # Try to use any saved 'blocks' field if present
                    if "blocks" in r and isinstance(r["blocks"], list):
                        visualize_blocks_on_image(img_path, r["blocks"], out_ann)
                    else:
                        # fallback: open and write a copy with small overlay
                        img = Image.open(img_path).convert("RGB")
                        img.save(out_ann)
                except Exception as e:
                    print(f"[WARN] annotate failed for {img_path}: {e}. Copying instead.")
                    img = Image.open(img_path).convert("RGB"); img.save(out_ann)
            else:
                # No visualize helper: copy the image to output
                img = Image.open(img_path).convert("RGB")
                img.save(out_ann)
            ann_paths.append((out_ann, r))
        except Exception as e:
            print(f"[WARN] couldn't create annotated image for {img_path}: {e}")
    if not ann_paths:
        print("No annotated images created; skipping montage.")
        return
    # Create montage: horizontally align up to 3
    imgs = [Image.open(p[0]).resize((800,1000)) for p in ann_paths]  # unify size
    widths, heights = zip(*(im.size for im in imgs))
    total_w = sum(widths)
    max_h = max(heights)
    montage = Image.new('RGB', (total_w, max_h + 120), (255,255,255))
    x = 0
    for i, im in enumerate(imgs):
        montage.paste(im, (x, 0))
        # draw labels under each image
        draw_txt = f"Pred: {ann_paths[i][1].get('deterministic_recommended_marks', 'N/A'):.2f} | Composite: {ann_paths[i][1].get('composite_confidence', 0.0):.2f}"
        from PIL import ImageDraw, ImageFont
        d = ImageDraw.Draw(montage)
        try:
            fnt = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            fnt = ImageFont.load_default()
        d.text((x + 10, max_h + 10), draw_txt, fill=(0,0,0), font=fnt)
        x += im.size[0]
    out_mont = os.path.join(OUT_DIR, "fig_montage_examples.png")
    montage.save(out_mont)
    print("[SAVED]", out_mont)

def main():
    path = find_records_file()
    if not path:
        print("No eval_records.jsonl found in expected locations. Create one by running the pipeline with saving enabled.")
        print("Looked in:", POSSIBLE_PATHS)
        return
    print("Loading records from:", path)
    records = load_records(path)
    if not records:
        print("No records found in file.")
        return
    print(f"Loaded {len(records)} records.")

    # Run plots
    plot_ocr_conf_hist(records)
    plot_routing_bar(records)
    plot_component_boxplots(records)
    plot_composite_conf_hist(records)
    plot_predicted_marks_hist(records)
    plot_keyword_vs_semantic(records)
    plot_length_vs_finalpct(records)
    create_qualitative_montage(records)

if __name__ == "__main__":
    main()
