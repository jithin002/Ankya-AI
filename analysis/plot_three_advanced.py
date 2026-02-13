#!/usr/bin/env python3
"""
analysis/plot_three_advanced.py

Produces three advanced visualizations from eval_records.jsonl:
  1) Spatial OCR-confidence overlay on a representative page (if block bboxes exist)
  2) Embedding projection (t-SNE or UMAP) of student answers colored by composite_confidence
  3) Sankey-like routing flow (counts -> route categories)

Save this as analysis/plot_three_advanced.py and run:
    python analysis/plot_three_advanced.py

Notes:
 - Embedding plot will use precomputed embeddings in records under 'embed' or compute them
   using sentence-transformers (if installed). TSNE (scikit-learn) preferred, UMAP fallback.
 - Spatial overlay requires records that include 'blocks' (list of dicts with 'bbox' and 'conf').
   If no blocks are found, the script will try to copy a debug image created by the pipeline
   (e.g., image_path + ".preproc_debug_trocr.png" or ".preproc_debug.png") as a fallback.
"""

import os
import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# ---------------- config ----------------
OUT_DIR = os.path.join("analysis", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

POSSIBLE_RECORD_PATHS = [
    "eval_records.jsonl",
    os.path.join("..", "eval_records.jsonl"),
    os.path.join("analysis", "eval_records.jsonl"),
    os.path.join(os.path.dirname(__file__), "..", "eval_records.jsonl") if "__file__" in globals() else None
]
POSSIBLE_RECORD_PATHS = [p for p in POSSIBLE_RECORD_PATHS if p]

# ---------------- helpers ----------------
def find_records_file():
    for p in POSSIBLE_RECORD_PATHS:
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

def save_fig(fig, fname):
    out = os.path.join(OUT_DIR, fname)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[SAVED]", out)

# ---------------- 1) Spatial overlay ----------------
def plot_spatial_overlay(records, outname="fig_ocr_conf_overlay.png"):
    """
    Draw semi-transparent rectangles on a representative scanned image
    where alpha ~ confidence for each OCR bbox.
    Requires record with 'image_path' and 'blocks' (list with 'bbox' and 'conf').
    Fallback: copy pipeline debug image if present.
    """
    # find a record that has both image_path and blocks
    rec = next((r for r in records if r.get("image_path") and (r.get("blocks") or r.get("ocr_blocks") or r.get("per_line"))), None)

    # fallback: try record with image_path only (we'll search for debug images)
    if rec is None:
        rec = next((r for r in records if r.get("image_path")), None)
        if rec is None:
            print("[SPATIAL] No records with image_path found; skipping spatial overlay.")
            return

    img_path = rec.get("image_path")
    if not img_path or not os.path.exists(img_path):
        print(f"[SPATIAL] Image not found at {img_path}; skipping spatial overlay.")
        return

    blocks = rec.get("blocks") or rec.get("ocr_blocks") or rec.get("per_line") or []
    # If no structured blocks, check for debug images produced by pipeline and copy them
    if not blocks:
        debug_paths = [
            img_path + ".preproc_debug_trocr.png",
            img_path + ".preproc_debug.png",
            img_path + ".preproc_debug_trocr.jpg",
            img_path + ".preproc_debug.jpg"
        ]
        found_debug = next((p for p in debug_paths if os.path.exists(p)), None)
        if found_debug:
            # copy debug image to out folder and return
            dst = os.path.join(OUT_DIR, outname)
            try:
                Image.open(found_debug).save(dst)
                print(f"[SPATIAL] No blocks found; copied pipeline debug image: {found_debug} -> {dst}")
            except Exception as e:
                print(f"[SPATIAL] Failed to copy debug image: {e}")
            return
        else:
            print("[SPATIAL] No blocks and no pipeline debug image found; skipping spatial overlay.")
            return

    try:
        img = Image.open(img_path).convert("RGBA")
    except Exception as e:
        print("[SPATIAL] Failed to open image:", e)
        return

    overlay = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)

    # standardize block formats and draw rectangles
    for b in blocks:
        # support multiple block shapes
        bbox = None
        conf = None
        if isinstance(b, dict):
            bbox = b.get("bbox") or b.get("box") or b.get("bbox_px")
            conf = b.get("conf") or b.get("confidence") or b.get("conf_score")
        elif isinstance(b, (list, tuple)) and len(b) >= 3:
            # maybe EasyOCR triple: (bbox_polygon, text, conf)
            maybe_bbox = b[0]
            conf = b[2] if len(b) > 2 else None
            if isinstance(maybe_bbox, (list, tuple)) and len(maybe_bbox) >= 4 and isinstance(maybe_bbox[0], (list, tuple)):
                xs = [int(p[0]) for p in maybe_bbox]; ys = [int(p[1]) for p in maybe_bbox]
                bbox = (min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
        if bbox is None:
            continue
        # normalize bbox to x,y,w,h
        if isinstance(bbox[0], (list, tuple)):
            xs = [int(p[0]) for p in bbox]; ys = [int(p[1]) for p in bbox]
            x = min(xs); y = min(ys); w = max(xs)-x; h = max(ys)-y
        else:
            # assume [x,y,w,h] or (x,y,w,h)
            x,y,w,h = [int(v) for v in bbox[:4]]
        try:
            conf_val = float(conf) if conf is not None else 50.0
        except:
            conf_val = 50.0
        # map conf (0..100) -> alpha (30..220) inverted so low conf = bright overlay
        # we want low-conf areas to stand out (red), so alpha increases as conf decreases
        alpha = int(np.clip(220 - (conf_val * 1.6), 30, 230))
        # color: red with computed alpha
        draw.rectangle([x, y, x + w, y + h], fill=(255, 0, 0, alpha), outline=(180,0,0,200))
    composed = Image.alpha_composite(img, overlay)

    # save composed image
    outpath = os.path.join(OUT_DIR, outname)
    try:
        composed.convert("RGB").save(outpath, quality=95)
        print("[SPATIAL] Saved overlay image to:", outpath)
    except Exception as e:
        print("[SPATIAL] Failed to save overlay image:", e)

# ---------------- 2) Embedding projection (t-SNE/UMAP) ----------------
def plot_embedding_projection(records, outname="fig_embedding_tsne.png", use_precomputed=True):
    """
    Create a 2D projection (t-SNE or UMAP) of embeddings for student_text.
    Color points by composite_confidence (0..1). Saves image to OUT_DIR.
    """
    # gather texts and possible embeddings
    texts = [r.get("student_text","") for r in records]
    composite_conf = np.array([r.get("composite_confidence", 0.0) for r in records], dtype=float)

    # Try to use precomputed embeddings if present
    embs = None
    if use_precomputed and any('embed' in r or 'embedding' in r for r in records):
        emb_list = []
        for r in records:
            if 'embed' in r:
                emb_list.append(r['embed'])
            elif 'embedding' in r:
                emb_list.append(r['embedding'])
            else:
                emb_list.append(None)
        # keep only those with embeddings
        valid_idx = [i for i, e in enumerate(emb_list) if e is not None]
        if len(valid_idx) >= 2:
            embs = np.array([emb_list[i] for i in valid_idx], dtype=float)
            confs = composite_conf[valid_idx]
            labels = [os.path.basename(records[i].get("image_path","")) for i in valid_idx]
        else:
            embs = None

    # If no precomputed embeddings, try to compute with sentence-transformers (optional)
    if embs is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("[EMBED] Computing embeddings with sentence-transformers (all-MiniLM-L6-v2).")
            model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            embs = model.encode(texts, show_progress_bar=True)
            confs = composite_conf
            labels = [os.path.basename(r.get("image_path","")) for r in records]
        except Exception as e:
            print("[EMBED] sentence-transformers not available or failed:", e)
            print("[EMBED] Skipping embedding projection.")
            return

    if embs is None or len(embs) < 2:
        print("[EMBED] Not enough embeddings to compute projection (need >=2).")
        return

    # Try t-SNE (scikit-learn), else UMAP
    proj = None
    try:
        from sklearn.manifold import TSNE
        print("[EMBED] Running t-SNE...")
        proj = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto').fit_transform(embs)
    except Exception as e:
        print("[EMBED] t-SNE not available or failed:", e)
        try:
            import umap
            print("[EMBED] Running UMAP...")
            proj = umap.UMAP(n_components=2, random_state=42).fit_transform(embs)
        except Exception as e2:
            print("[EMBED] UMAP not available or failed:", e2)
            print("[EMBED] Skipping embedding projection.")
            return

    # Plot
    fig = plt.figure(figsize=(7,6))
    sc = plt.scatter(proj[:,0], proj[:,1], c=confs, cmap='viridis', s=40, alpha=0.85)
    cbar = plt.colorbar(sc)
    cbar.set_label('composite_confidence (0-1)')
    plt.title("2D projection of student-answer embeddings")
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    # annotate a few extreme points (min, median, max conf)
    try:
        idx_min = int(np.argmin(confs)); idx_med = int(np.argsort(confs)[len(confs)//2]); idx_max = int(np.argmax(confs))
        for idx in [idx_min, idx_med, idx_max]:
            plt.annotate(labels[idx], (proj[idx,0], proj[idx,1]), fontsize=7, alpha=0.8)
    except Exception:
        pass
    save_fig(fig, outname)

# ---------------- 3) Sankey-like routing flow ----------------
def plot_sankey_routing(records, outname="fig_routing_sankey.png"):
    """
    Plot a Sankey-like flow from dataset -> route categories (auto_accept/partial_review/teacher_review).
    Uses matplotlib.sankey if available; otherwise falls back to a simple bar chart.
    """
    routes = [r.get("route","unknown") for r in records]
    if not routes:
        print("[SANKEY] No 'route' fields found; skipping.")
        return
    cnt = Counter(routes)
    labels = list(cnt.keys())
    vals = [cnt[k] for k in labels]
    total = sum(vals)

    # Try matplotlib.sankey
    try:
        from matplotlib.sankey import Sankey
        fig = plt.figure(figsize=(7,4))
        sank = Sankey(ax=fig.add_subplot(1,1,1), unit=None)
        # flows: input positive, outputs negative; create flow list: [total, -c1, -c2, -c3, ...]
        flows = [float(total)] + [-float(v) for v in vals]
        labels_sank = ["dataset"] + labels
        sank.add(flows=flows, labels=labels_sank, orientations=[0] + [1]*len(vals), trunklength=1.0)
        sank.finish()
        plt.title("Sankey: dataset -> routing categories")
        save_fig(fig, outname)
        return
    except Exception as e:
        print("[SANKEY] matplotlib.sankey not usable:", e)
        # fallback to stacked horizontal bar showing counts
        fig = plt.figure(figsize=(7,4))
        plt.bar(labels, vals, color=plt.cm.tab10(range(len(vals))))
        plt.ylabel("Count")
        plt.title("Routing outcomes (counts)")
        for i,v in enumerate(vals):
            plt.text(i, v + max(vals)*0.01, str(v), ha='center', va='bottom')
        save_fig(fig, outname)
        return

# ---------------- main ----------------
def main():
    rec_file = find_records_file()
    if not rec_file:
        print("No eval_records.jsonl found in expected locations. Looked in:")
        for p in POSSIBLE_RECORD_PATHS:
            print("  ", p)
        print("Run your pipeline with saving enabled to create eval_records.jsonl (one JSON per line).")
        return

    print("Loading records from:", rec_file)
    records = load_records(rec_file)
    if not records:
        print("No records found in the file:", rec_file)
        return
    print(f"Loaded {len(records)} records.")

    # 1) Spatial overlay (one representative)
    print("\n[STEP 1] Spatial overlay (if blocks exist or debug images present)")
    plot_spatial_overlay(records)

    # 2) Embedding projection
    print("\n[STEP 2] Embedding projection (t-SNE/UMAP)")
    plot_embedding_projection(records)

    # 3) Sankey-like routing flow
    print("\n[STEP 3] Sankey-like routing flow")
    plot_sankey_routing(records)

if __name__ == "__main__":
    main()
