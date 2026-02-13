# analysis/plot_figures.py
import os
import json
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Optional: only import scipy if available (pearsonr)
try:
    from scipy.stats import pearsonr
except Exception:
    pearsonr = None

# Output folder for figures
OUT_DIR = "analysis/figs"
os.makedirs(OUT_DIR, exist_ok=True)

def load_records(path="eval_records.jsonl"):
    records = []
    if not os.path.exists(path):
        print(f"[ERROR] records file not found: {path}")
        return records
    with open(path, "r", encoding="utf-8") as fh:
        for l in fh:
            l = l.strip()
            if not l:
                continue
            try:
                records.append(json.loads(l))
            except Exception as e:
                print("Failed to parse line:", e)
    print(f"[INFO] loaded {len(records)} records from {path}")
    return records

def save_fig(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print("[SAVED]", p)

def plot_ocr_conf_hist(records):
    confs = [r.get("avg_ocr_conf_raw", 0.0) for r in records]
    if not confs:
        print("No OCR confidences to plot.")
        return
    fig = plt.figure(figsize=(6,4))
    plt.hist(confs, bins=20)
    plt.xlabel("Avg OCR confidence (0-100)")
    plt.ylabel("Count")
    plt.title("OCR confidence distribution")
    save_fig(fig, "ocr_conf_hist.png")
    plt.close(fig)

def plot_pred_vs_human(records):
    y_pred = np.array([r.get("deterministic_recommended_marks", np.nan) for r in records])
    y_true = np.array([r.get("human_marks", np.nan) for r in records])
    mask = ~np.isnan(y_true)
    y_pred = y_pred[mask]; y_true = y_true[mask]
    if len(y_true) < 2:
        print("Not enough human-labeled samples (need >=2) for Pred-vs-Human plot.")
        return
    fig = plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.0)
    if pearsonr:
        try:
            r, p = pearsonr(y_true, y_pred)
            plt.title(f"Predicted vs Human marks (Pearson R={r:.2f})")
        except Exception as e:
            plt.title("Predicted vs Human marks")
    else:
        plt.title("Predicted vs Human marks")
    plt.xlabel("Human marks")
    plt.ylabel("Predicted marks")
    save_fig(fig, "pred_vs_human.png")
    plt.close(fig)

def plot_bland_altman(records):
    y_pred = np.array([r.get("deterministic_recommended_marks", np.nan) for r in records])
    y_true = np.array([r.get("human_marks", np.nan) for r in records])
    mask = ~np.isnan(y_true)
    y_pred = y_pred[mask]; y_true = y_true[mask]
    if len(y_true) < 2:
        print("Not enough human-labeled samples (need >=2) for Bland–Altman plot.")
        return
    mean_vals = (y_pred + y_true) / 2.0
    diffs = y_pred - y_true
    md = np.mean(diffs)
    sd = np.std(diffs, ddof=1)
    fig = plt.figure(figsize=(6,4))
    plt.scatter(mean_vals, diffs, alpha=0.6)
    plt.axhline(md, color='black', linestyle='-')
    plt.axhline(md + 1.96 * sd, color='red', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='red', linestyle='--')
    plt.xlabel("Mean of (predicted, human)")
    plt.ylabel("Difference (predicted - human)")
    plt.title("Bland–Altman: agreement")
    save_fig(fig, "bland_altman.png")
    plt.close(fig)

def plot_error_vs_conf(records):
    y_pred = np.array([r.get("deterministic_recommended_marks", np.nan) for r in records])
    y_true = np.array([r.get("human_marks", np.nan) for r in records])
    conf = np.array([r.get("avg_ocr_conf_raw", np.nan) for r in records])
    mask = ~np.isnan(y_true)
    y_pred = y_pred[mask]; y_true = y_true[mask]; conf = conf[mask]
    if len(y_true) == 0:
        print("No human-labeled records for Error vs OCR confidence plot.")
        return
    err = np.abs(y_pred - y_true)
    fig = plt.figure(figsize=(6,4))
    plt.scatter(conf, err, alpha=0.6)
    # binned mean trend
    bins = np.linspace(0, 100, 11)
    inds = np.digitize(conf, bins)
    bin_centers = []
    bin_means = []
    for i in range(1, len(bins)+1):
        maskb = inds == i
        if np.any(maskb):
            bin_centers.append((bins[i-1] + (bins[i-1]+(bins[1]-bins[0])))/2)
            bin_means.append(np.nanmean(err[maskb]))
    if bin_centers and bin_means:
        plt.plot(bin_centers[:len(bin_means)], bin_means, linestyle='-', linewidth=2)
    plt.xlabel("Avg OCR confidence (0-100)")
    plt.ylabel("Absolute error (marks)")
    plt.title("Absolute grading error vs OCR confidence")
    save_fig(fig, "error_vs_conf.png")
    plt.close(fig)

def plot_routing_counts(records):
    routes = [r.get("route", "unknown") for r in records]
    if not routes:
        print("No route information.")
        return
    cnt = Counter(routes)
    labels = list(cnt.keys()); vals = [cnt[k] for k in labels]
    fig = plt.figure(figsize=(6,4))
    plt.bar(labels, vals)
    plt.xlabel("Routing decision")
    plt.ylabel("Count")
    plt.title("Routing distribution")
    save_fig(fig, "routing.png")
    plt.close(fig)

def plot_component_boxplots(records):
    keys = ['keyword_pct','semantic_pct','grammar_pct','coverage_pct','presentation_pct']
    data = []
    for k in keys:
        arr = [r.get("component_scores", {}).get(k, np.nan) for r in records]
        arr = [x for x in arr if not (x is None or (isinstance(x, float) and math.isnan(x)))]
        data.append(arr if arr else [0.0])
    fig = plt.figure(figsize=(8,4))
    plt.boxplot(data, labels=keys, showfliers=False)
    plt.ylabel("Percent (%)")
    plt.title("Component score distributions")
    save_fig(fig, "component_boxplots.png")
    plt.close(fig)

def create_qualitative_montage(records, max_examples=3):
    """
    If records contain 'image_path' and 'question_map' or 'student_text' and
    blocks data, try to produce up to max_examples annotated images using your pipeline's visualize helper.
    This is optional — the function will safely skip if prerequisites are missing.
    """
    # Attempt to import visualize function from your pipeline
    try:
        from pipelines.icr_pipeline3 import visualize_blocks_on_image
    except Exception as e:
        print("Cannot import visualize_blocks_on_image:", e)
        visualize_blocks_on_image = None

    examples = []
    # Prefer high/medium/low composite_confidence to pick diverse examples
    recs = sorted(records, key=lambda r: r.get("composite_confidence", 0.0))
    if not recs:
        print("No records for qualitative montage.")
        return

    # pick low, mid, high
    picks = []
    if len(recs) >= 1:
        picks.append(recs[0])
    if len(recs) >= 2:
        picks.append(recs[len(recs)//2])
    if len(recs) >= 3:
        picks.append(recs[-1])
    picks = picks[:max_examples]

    montage_paths = []
    for i, r in enumerate(picks):
        img_path = r.get("image_path")
        blocks = r.get("question_map")  # your pipeline stores 'question_map' mapping; blocks are in 'blocks' sometimes
        # If blocks are stored differently, allow fallback to 'blocks' key
        block_objs = r.get("blocks") or r.get("ocr_blocks") or None
        out_img = os.path.join(OUT_DIR, f"qual_example_{i}.png")
        try:
            if visualize_blocks_on_image and block_objs and img_path and os.path.exists(img_path):
                # If block_objs is a list of dicts with bbox/text/conf — use visualization
                visualize_blocks_on_image(img_path, block_objs, out_img)
            else:
                # Fallback: just copy original image (if exists) and overlay a small text strip using PIL
                from PIL import Image, ImageDraw, ImageFont
                if img_path and os.path.exists(img_path):
                    im = Image.open(img_path).convert("RGB")
                    draw = ImageDraw.Draw(im)
                    # small overlay text
                    txt = f"Pred: {r.get('deterministic_recommended_marks')}  Human: {r.get('human_marks')}\nRoute: {r.get('route')}"
                    try:
                        font = ImageFont.truetype("arial.ttf", 16)
                    except:
                        font = ImageFont.load_default()
                    draw.rectangle([0, im.height - 60, im.width, im.height], fill=(255,255,255,200))
                    draw.text((5, im.height - 55), txt, fill=(0,0,0), font=font)
                    im.save(out_img)
                else:
                    print("Skipping qualitative example, image missing or visualize helper unavailable.")
                    continue
            montage_paths.append(out_img)
        except Exception as e:
            print("Failed to create qualitative image:", e)
            continue

    # If we produced images, create a composite montage
    if montage_paths:
        try:
            from PIL import Image
            ims = [Image.open(p) for p in montage_paths]
            widths, heights = zip(*(i.size for i in ims))
            total_w = sum(widths)
            max_h = max(heights)
            montage = Image.new('RGB', (total_w, max_h), (255,255,255))
            x = 0
            for im in ims:
                montage.paste(im, (x, 0))
                x += im.width
            out = os.path.join(OUT_DIR, "qualitative_montage.png")
            montage.save(out)
            print("[SAVED]", out)
        except Exception as e:
            print("Failed to assemble montage:", e)

def main():
    records = load_records("eval_records.jsonl")

    # Basic checks
    if not records:
        print("No records — run your evaluation script and ensure eval_records.jsonl exists.")
        return

    plot_ocr_conf_hist(records)
    plot_routing_counts(records)
    plot_component_boxplots(records)
    # plots that require human marks
    plot_pred_vs_human(records)
    plot_bland_altman(records)
    plot_error_vs_conf(records)
    # optional qualitative visualization
    create_qualitative_montage(records, max_examples=3)
    print("All done. Check the 'analysis/figs' folder for generated plots.")

if __name__ == "__main__":
    main()
