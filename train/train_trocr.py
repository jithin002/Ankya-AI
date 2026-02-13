# train/train_trocr.py
import os
import json
from multiprocessing import freeze_support
from datasets import load_dataset, Dataset, DatasetDict, Features, ClassLabel, Value, Array3D
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
import torch
from PIL import Image
import numpy as np
import argparse
import evaluate

# ---------------- args ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--images_dir", required=True)
parser.add_argument("--labels_csv", required=True)   # CSV produced earlier
parser.add_argument("--pretrained", default="microsoft/trocr-base-handwritten")
parser.add_argument("--output_dir", default="fine_tuned_trocr")
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--per_device_train_batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--unfreeze_encoder", action="store_true", help="If set, do not freeze the encoder during training")
args = parser.parse_args()

# ---------------- init models/processors ----------------
model = VisionEncoderDecoderModel.from_pretrained(args.pretrained)
try:
    model.gradient_checkpointing_enable()
    print("Enabled gradient checkpointing")
except Exception as e:
    print("Could not enable gradient checkpointing:", e)

# 2) freeze encoder parameters (train only decoder & lm head)
if not args.unfreeze_encoder:
    for name, param in model.encoder.named_parameters():
        param.requires_grad = False
    print("Froze encoder parameters; training only decoder/heads.")
else:
    print("Encoder left trainable (unfrozen) as requested")

# 3) optionally print trainable param count
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable} / {total}")


processor = ViTImageProcessor.from_pretrained(args.pretrained)
tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

# Hyperparams
max_target_length = 128
# ensure tokenizer has pad token; fallback to eos if missing
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # try to set a safe pad token (rare case)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

# ---------------- Ensure model config tokens are set ----------------
if getattr(model.config, "decoder_start_token_id", None) is None:
    if getattr(tokenizer, "bos_token_id", None) is not None:
        model.config.decoder_start_token_id = tokenizer.bos_token_id
    elif getattr(tokenizer, "cls_token_id", None) is not None:
        model.config.decoder_start_token_id = tokenizer.cls_token_id
    elif getattr(tokenizer, "eos_token_id", None) is not None:
        model.config.decoder_start_token_id = tokenizer.eos_token_id
    else:
        model.config.decoder_start_token_id = 0  # last resort

if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
    model.config.pad_token_id = tokenizer.pad_token_id
if getattr(model.config, "eos_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
    model.config.eos_token_id = tokenizer.eos_token_id

# keep vocab_size consistent if possible
try:
    model.config.vocab_size = model.decoder.config.vocab_size
except Exception:
    pass

print("Model config tokens:",
      "decoder_start_token_id=", model.config.decoder_start_token_id,
      "pad_token_id=", model.config.pad_token_id,
      "eos_token_id=", model.config.eos_token_id)

# ---------------- load dataset CSV ----------------
import csv
rows = []
with open(args.labels_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        img_path = os.path.join(args.images_dir, r['filename'])
        text = r['text']
        if not os.path.exists(img_path):
            print("missing image", img_path)
            continue
        if not text or len(text.strip()) == 0:
            continue
        rows.append({"image": img_path, "text": text})
print("Loaded examples:", len(rows))

# simple split
np.random.seed(args.seed)
np.random.shuffle(rows)
split = int(0.95 * len(rows))
train_rows = rows[:split]
val_rows = rows[split:]

from datasets import Dataset
train_ds = Dataset.from_list(train_rows)
val_ds = Dataset.from_list(val_rows)
dataset = DatasetDict({"train": train_ds, "validation": val_ds})

# ---------------- Preprocess / map function ----------------
def load_and_resize(path, size=(224,224)):
    img = Image.open(path).convert("RGB")
    img = img.resize(size, resample=Image.BICUBIC)
    return img

def map_fn(batch):
    images = [load_and_resize(p, size=(224,224)) for p in batch["image"]]
    # processor returns tensors with shape (batch, channels, H, W)
    pixel_values = processor(images=images, return_tensors="pt").pixel_values  # torch tensor
    # convert each element to numpy array for efficient stacking in collate
    pixel_numpy_list = [pv.numpy() for pv in pixel_values]
    # keep labels as raw text (strings) so collate can call tokenizer(...)
    labels = batch["text"]
    return {"pixel_values": pixel_numpy_list, "labels": labels}


dataset = dataset.map(map_fn, batched=True, remove_columns=["image","text"])

# ---------------- Collate function (efficient) ----------------
def collate_fn(batch):
    """
    batch: list of {"pixel_values": numpy-array (C,H,W), "labels": list[int]}
    Returns dict with torch tensors ready for Trainer.
    """
    # stack pixel_values into single numpy array then convert to torch
    pv_list = [x["pixel_values"] for x in batch]
    try:
        pv_arr = np.stack(pv_list, axis=0)  # shape: (B, C, H, W)
    except Exception:
        # fallback: convert each to np.array then stack
        pv_arr = np.stack([np.array(x) for x in pv_list], axis=0)
    pixel_values = torch.from_numpy(pv_arr).float()

    # labels: use tokenizer to pad to longest in batch
    labels_list = [x["labels"] for x in batch]
    tokenized = tokenizer(labels_list, padding="longest", truncation=True, max_length=max_target_length, return_tensors="pt")
    labels = tokenized["input_ids"]
    # replace pad token id with -100 (ignored by loss)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else model.config.pad_token_id
    if pad_id is None:
        pad_id = -100
    labels[labels == pad_id] = -100

    return {"pixel_values": pixel_values, "labels": labels}

# set model config for generation (already set above) - keep consistent
model.config.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else model.config.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else model.config.pad_token_id

# ---------------- Trainer args ----------------
training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,                  # where checkpoints & logs go
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=max(1, args.per_device_train_batch_size),
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    fp16=bool(args.fp16),                        # mixed precision
    num_train_epochs=args.epochs,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    logging_steps=200,
    save_strategy="epoch",
    eval_strategy="epoch",
    eval_accumulation_steps=4,
    dataloader_num_workers=0,                    # safe on Windows
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="none",                            # disable HF logging if you don't want it
    predict_with_generate=True,
    generation_max_length=max_target_length,
    generation_num_beams=4,
)

# metrics
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_metrics(eval_pred):
    # eval_pred is an EvalPrediction with .predictions and .label_ids
    preds = eval_pred.predictions
    labels = eval_pred.label_ids

    # If generate() was used, predictions can be a tuple (generated_ids, ...)
    if isinstance(preds, tuple):
        preds = preds[0]

    # If logits were returned, take argmax (safe fallback)
    try:
        if isinstance(preds, np.ndarray) and preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)
    except Exception:
        pass

    # decode preds
    try:
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    except Exception:
        decoded_preds = [str(x) for x in preds]

    # decode labels (replace -100 with pad_token_id)
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(labels, np.ndarray):
        labels_clean = np.where(labels == -100, pad_token_id, labels)
        decoded_labels = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)
    else:
        decoded_labels = [str(x) for x in labels]

    # sanity: lengths must match
    if len(decoded_preds) != len(decoded_labels):
        # best-effort: truncate/pad to match
        minlen = min(len(decoded_preds), len(decoded_labels))
        decoded_preds = decoded_preds[:minlen]
        decoded_labels = decoded_labels[:minlen]

    cer_res = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    wer_res = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    # return scalar values for trainer logging
    cer_val = cer_res.get("cer", cer_res) if isinstance(cer_res, dict) else cer_res
    wer_val = wer_res.get("wer", wer_res) if isinstance(wer_res, dict) else wer_res
    return {"cer": cer_val, "wer": wer_val}

def main():
    # quick sanity print
    print("Train examples:", len(dataset["train"]), "Validation examples:", len(dataset["validation"]))
    # small test batch from dataset
    sample = dataset["train"][0]
    print("Sample pixel_values shape (per-sample):", np.array(sample["pixel_values"]).shape if "pixel_values" in sample else None)
    print("Sample label length:", len(sample["labels"]) if "labels" in sample else None)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn,
        tokenizer=tokenizer,           # pass the text tokenizer here (not the image processor)
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print("Saved model to", args.output_dir)

    # Save processor and tokenizer so the checkpoint directory is self-contained
    try:
        processor.save_pretrained(args.output_dir)
        print("Saved image processor to", args.output_dir)
    except Exception as e:
        print("Failed to save processor:", e)
    try:
        tokenizer.save_pretrained(args.output_dir)
        print("Saved tokenizer to", args.output_dir)
    except Exception as e:
        print("Failed to save tokenizer:", e)

if __name__ == "__main__":
    # on Windows, ensures child processes can be spawned safely
    freeze_support()
    main()