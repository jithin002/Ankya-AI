# train/train_trocr.py  -- LoRA + Albumentations + Lazy Dataset Edition
#
# KEY FIX: Uses a PyTorch Dataset (__getitem__) so images are loaded
# one-at-a-time during training, preventing the ArrowMemoryError.
#
# Usage:
#   python train/train_trocr.py \
#       --images_dir data/images \
#       --labels_csv data/labels.csv \
#       --output_dir fine_tuned_trocr_small \
#       --pretrained microsoft/trocr-small-handwritten \
#       --epochs 15 --fp16

import os
import csv
import sys
import random
import argparse
from multiprocessing import freeze_support

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
import evaluate

# ── LoRA / PEFT ────────────────────────────────────────────────────────────────
from peft import LoraConfig, get_peft_model, TaskType

# ── Albumentations ─────────────────────────────────────────────────────────────
try:
    import albumentations as A
    HAS_AUG = True
except ImportError:
    HAS_AUG = False
    print("[WARN] albumentations not installed — training without augmentation.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI ARGS
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--images_dir", required=True)
parser.add_argument("--labels_csv",  required=True)
parser.add_argument("--pretrained",  default="microsoft/trocr-small-handwritten")
parser.add_argument("--output_dir",  default="fine_tuned_trocr_small")
parser.add_argument("--epochs",      type=int,   default=15)
parser.add_argument("--per_device_train_batch_size", type=int,   default=4)
parser.add_argument("--gradient_accumulation_steps", type=int,   default=4)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--fp16",        action="store_true")
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--lora_r",      type=int,   default=16)
parser.add_argument("--lora_alpha",  type=int,   default=32)
parser.add_argument("--lora_dropout",type=float, default=0.05)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION (applied lazily per sample at training time)
# ══════════════════════════════════════════════════════════════════════════════
if HAS_AUG:
    aug = A.Compose([
        A.Rotate(limit=3, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.4),               # updated API — no var_limit kwarg
        A.MotionBlur(blur_limit=3, p=0.3),
        A.CLAHE(p=0.3),
    ])
else:
    aug = None


def apply_aug(pil_img: Image.Image) -> Image.Image:
    if aug is None:
        return pil_img
    arr = np.array(pil_img)
    return Image.fromarray(aug(image=arr)["image"])


# ══════════════════════════════════════════════════════════════════════════════
# LOAD BASE MODEL + LORA WRAP
# ══════════════════════════════════════════════════════════════════════════════
print(f"Loading base model: {args.pretrained}")
base_model = VisionEncoderDecoderModel.from_pretrained(args.pretrained)

lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=["query", "value"],
    bias="none",
    inference_mode=False,
)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()

processor = ViTImageProcessor.from_pretrained(args.pretrained)
tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

max_target_length = 128
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

if not getattr(model.config, "decoder_start_token_id", None):
    model.config.decoder_start_token_id = (
        tokenizer.bos_token_id or tokenizer.eos_token_id or 0
    )
model.config.pad_token_id = pad_id
model.config.eos_token_id = tokenizer.eos_token_id


# ══════════════════════════════════════════════════════════════════════════════
# LOAD CSV ROWS
# ══════════════════════════════════════════════════════════════════════════════
rows = []
with open(args.labels_csv, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        img_path = os.path.join(args.images_dir, r["filename"])
        text = r["text"].strip()
        if os.path.exists(img_path) and text:
            rows.append({"image": img_path, "text": text})

print(f"Loaded {len(rows)} examples.")
random.shuffle(rows)
split_idx  = int(0.95 * len(rows))
train_rows = rows[:split_idx]
val_rows   = rows[split_idx:]
print(f"Train: {len(train_rows)}  |  Val: {len(val_rows)}")


# ══════════════════════════════════════════════════════════════════════════════
# LAZY PYTORCH DATASET — images loaded ONE AT A TIME, never all at once
# ══════════════════════════════════════════════════════════════════════════════
class TrOCRDataset(TorchDataset):
    def __init__(self, rows, is_train=False):
        self.rows     = rows
        self.is_train = is_train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        # Load image
        img = Image.open(row["image"]).convert("RGB").resize((384, 384), Image.BICUBIC)
        if self.is_train:
            img = apply_aug(img)
        # Process image → pixel_values tensor
        pixel_values = processor(images=img, return_tensors="pt").pixel_values[0]
        # Tokenize label
        label_ids = tokenizer(
            row["text"],
            max_length=max_target_length,
            truncation=True,
            padding=False,
        ).input_ids
        return {"pixel_values": pixel_values, "labels": label_ids}


train_ds = TrOCRDataset(train_rows, is_train=True)
val_ds   = TrOCRDataset(val_rows,   is_train=False)


# ══════════════════════════════════════════════════════════════════════════════
# COLLATE — pads token sequences to longest in batch (not whole dataset!)
# ══════════════════════════════════════════════════════════════════════════════
def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])  # (B, C, H, W)

    # Pad label sequences
    max_len = max(len(x["labels"]) for x in batch)
    labels = []
    for x in batch:
        ids = x["labels"]
        pad_len = max_len - len(ids)
        ids_padded = ids + [-100] * pad_len      # -100 = ignored by cross-entropy
        labels.append(ids_padded)
    labels = torch.tensor(labels, dtype=torch.long)

    return {"pixel_values": pixel_values, "labels": labels}


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


def compute_metrics(eval_pred):
    preds, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(preds, tuple):
        preds = preds[0]
    if isinstance(preds, np.ndarray) and preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    # Replace -100 padding values with pad_id in both preds and labels before decoding
    preds_clean = np.where(preds == -100, pad_id, preds)
    decoded_preds  = tokenizer.batch_decode(preds_clean, skip_special_tokens=True)
    
    labels_clean   = np.where(labels == -100, pad_id, labels)
    decoded_labels = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)

    n = min(len(decoded_preds), len(decoded_labels))
    decoded_preds, decoded_labels = decoded_preds[:n], decoded_labels[:n]

    cer_val = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    wer_val = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    cer_val = cer_val.get("cer", cer_val) if isinstance(cer_val, dict) else cer_val
    wer_val = wer_val.get("wer", wer_val) if isinstance(wer_val, dict) else wer_val
    return {"cer": cer_val, "wer": wer_val}


# ══════════════════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=max(1, args.per_device_train_batch_size),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=bool(args.fp16),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        dataloader_num_workers=0,       # Windows safe
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        predict_with_generate=True,
        generation_max_length=max_target_length,
        generation_num_beams=4,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()

    # Save the LoRA adapter + processor + tokenizer
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved to: {args.output_dir}")
    print("NOTE: To use in inference, load the BASE model then call `model.load_adapter(output_dir)`")


if __name__ == "__main__":
    freeze_support()
    main()