#!/usr/bin/env python3
"""
Re:Learn — DeBERTa-v3 Partial Training CLI (@[Quote] Strategy)
Author: Senior ML & DL Training Engineer

Implements the Partial Training Strategy for Model 1 (DeBERTa-v3-base):
- 17 target classes from data/taxonomy.json
- Concatenated features: Question Text + Expected Physics Summary + Student Final Answer + Student Working Steps
- Freezes embeddings (100%) and bottom 6 encoder layers (0-5)
- Fine-tunes top 6 encoder layers (6-11) and classification head
- Focal Loss with balanced class weights for severe class distribution skew
- Primary evaluation metric: Macro-F1 and confusion matrix
- Checkpoint saving and metrics logging to checkpoints/deberta/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_deberta")

# Check required libraries
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    logger.error("PyTorch is required for train_deberta.py. Please install torch.")
    sys.exit(1)

try:
    from transformers import (
        AutoTokenizer,
        DataCollatorWithPadding,
        get_linear_schedule_with_warmup,
    )
except ImportError:
    logger.error("Transformers is required for train_deberta.py. Please install transformers.")
    sys.exit(1)

try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
except ImportError:
    logger.error("scikit-learn is required for evaluation metrics. Please install scikit-learn.")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # Fallback if tqdm not installed

from backend.models.classifier import (
    DebertaMisconceptionClassifier,
    compute_balanced_class_weights,
    extract_record_label,
    format_misconception_input,
    load_taxonomy_labels,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


class MisconceptionTorchDataset(Dataset):
    """
    PyTorch Dataset wrapping formatted physics question-answer pairs and their
    ground-truth misconception labels.
    """
    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        label2id: Dict[str, int],
        max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.items = []

        for r in records:
            text = format_misconception_input(r)
            label_str = extract_record_label(r)
            label_idx = self.label2id.get(label_str, self.label2id.get("UNCERTAIN-GUESS", 0))
            self.items.append((text, label_idx))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text, label = self.items[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Dynamic padding in collator
            return_token_type_ids=False,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "label": label,
        }


def collate_fn_with_padding(tokenizer: AutoTokenizer):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [item["label"] for item in batch]
        features = [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch]
        batch_collated = data_collator(features)
        batch_collated["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch_collated

    return collate


def evaluate(
    model: DebertaMisconceptionClassifier,
    data_loader: DataLoader,
    device: torch.device,
    classes: List[str],
) -> Dict[str, Any]:
    """
    Run evaluation loop over data_loader and compute Macro-F1, Weighted-F1, Accuracy,
    loss, per-class metrics, and confusion matrix across canonical taxonomy classes.
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    num_samples = len(all_targets)
    if num_samples == 0:
        return {
            "loss": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "accuracy": 0.0,
            "classification_report_str": "Empty evaluation dataset (0 samples).",
            "classification_report_dict": {},
            "confusion_matrix": [[0] * len(classes) for _ in range(len(classes))],
            "predictions": [],
            "targets": [],
        }

    avg_loss = total_loss / num_samples

    y_true_str = [classes[idx] for idx in all_targets]
    y_pred_str = [classes[idx] for idx in all_preds]

    # Compute macro F1 across all canonical classes
    macro_f1 = float(f1_score(y_true_str, y_pred_str, labels=classes, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_str, y_pred_str, labels=classes, average="weighted", zero_division=0))
    accuracy = float(accuracy_score(y_true_str, y_pred_str))

    report_str = classification_report(
        y_true_str,
        y_pred_str,
        labels=classes,
        target_names=classes,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true_str,
        y_pred_str,
        labels=classes,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_str, y_pred_str, labels=classes)

    return {
        "loss": avg_loss,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "classification_report_str": report_str,
        "classification_report_dict": report_dict,
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds,
        "targets": all_targets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train & Evaluate Re:Learn Model 1 DeBERTa-v3 with Partial Training Strategy (@[Quote])",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-data", type=str, default="data/splits/train.json", help="Path to train split JSON")
    parser.add_argument("--val-data", type=str, default="data/splits/val.json", help="Path to val split JSON")
    parser.add_argument("--test-data", type=str, default="data/splits/test.json", help="Path to test split JSON")
    parser.add_argument("--taxonomy", type=str, default="data/taxonomy.json", help="Path to taxonomy JSON")
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v3-base", help="Hugging Face model checkpoint")
    parser.add_argument("--output-dir", type=str, default="checkpoints/deberta", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for trainable layers")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup proportion of total steps")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length for tokenization")
    parser.add_argument("--freeze-layers", type=int, default=6, help="Bottom encoder layers to freeze (0 to freeze_layers-1)")
    parser.add_argument("--loss", type=str, choices=["focal", "ce"], default="focal", help="Loss function")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma parameter for Focal Loss")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable inverse-frequency class weights")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Steps before updating weights")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', or 'auto')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit train samples for fast smoke testing")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit val samples for fast smoke testing")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Limit test samples for fast smoke testing")
    parser.add_argument("--no-eval-test", action="store_true", help="Skip final evaluation on test split")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  RE:LEARN MODEL 1 — DEBERTA-V3 PARTIAL TRAINING (@[QUOTE] STRATEGY)")
    print("=" * 80)

    # 1. Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Target compute device: {device} (CUDA available: {torch.cuda.is_available()})")

    # 2. Load Taxonomy Classes
    taxonomy_classes = load_taxonomy_labels(args.taxonomy)
    num_classes = len(taxonomy_classes)
    label2id = {cls: idx for idx, cls in enumerate(taxonomy_classes)}
    id2label = {idx: cls for idx, cls in enumerate(taxonomy_classes)}
    logger.info(f"Loaded {num_classes} canonical taxonomy classes from {args.taxonomy}")

    # 3. Load Splits
    logger.info(f"Loading training data from {args.train_data}...")
    train_records = load_dataset(args.train_data)
    if args.max_train_samples:
        train_records = train_records[: args.max_train_samples]

    if len(train_records) == 0:
        logger.error("Training dataset is empty. Provide at least 1 training sample.")
        return 1

    logger.info(f"Loading validation data from {args.val_data}...")
    val_records = load_dataset(args.val_data)
    if args.max_val_samples:
        val_records = val_records[: args.max_val_samples]

    test_records = []
    if not args.no_eval_test and args.test_data and Path(args.test_data).exists():
        logger.info(f"Loading test data from {args.test_data}...")
        test_records = load_dataset(args.test_data)
        if args.max_test_samples:
            test_records = test_records[: args.max_test_samples]

    logger.info(f"Dataset summary: Train={len(train_records)} | Val={len(val_records)} | Test={len(test_records)}")

    # 4. Compute Class Weights if enabled
    class_weights = None
    if not args.no_class_weights:
        train_label_indices = [
            label2id.get(extract_record_label(r), label2id.get("UNCERTAIN-GUESS", 0))
            for r in train_records
        ]
        class_weights = compute_balanced_class_weights(train_label_indices, num_classes=num_classes, device=device)
        logger.info(f"Computed inverse-frequency balanced class weights for {num_classes} classes.")

    # 5. Initialize Tokenizer & Model
    logger.info(f"Loading tokenizer for: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info(f"Instantiating DeBERTa-v3 Misconception Classifier with Partial Training...")
    logger.info(f"  * Strategy: Freeze embeddings + Freeze bottom {args.freeze_layers} encoder layers (0-{args.freeze_layers-1})")
    logger.info(f"  * Fine-tune: Top {12 - args.freeze_layers} encoder layers ({args.freeze_layers}-11) + pooler + classifier head")
    logger.info(f"  * Loss function: {args.loss.upper()} (gamma={args.focal_gamma if args.loss=='focal' else 'N/A'})")

    model = DebertaMisconceptionClassifier(
        model_name=args.model_name,
        classes=taxonomy_classes,
        freeze_embeddings=True,
        freeze_layers_count=args.freeze_layers,
        loss_type=args.loss,
        focal_gamma=args.focal_gamma,
        class_weights=class_weights,
    )
    model.to(device)

    # 6. Build Datasets and DataLoaders
    collate = collate_fn_with_padding(tokenizer)
    train_dataset = MisconceptionTorchDataset(train_records, tokenizer, label2id, max_length=args.max_length)
    val_dataset = MisconceptionTorchDataset(val_records, tokenizer, label2id, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,  # Safe on Windows
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    # 7. Optimizer with Layer-wise Weight Decay Exclusion
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    trainable_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in trainable_named_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in trainable_named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

    total_training_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    logger.info(f"Training schedule: {args.epochs} epochs | {total_training_steps} total steps | {warmup_steps} warmup steps")

    # 8. Training Loop
    best_val_macro_f1 = -1.0
    best_checkpoint_dir = output_dir / "best"
    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_history: List[Dict[str, Any]] = []

    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"] / args.gradient_accumulation_steps
            loss.backward()

            step_loss += loss.item() * args.gradient_accumulation_steps
            epoch_loss += loss.item() * args.gradient_accumulation_steps

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                curr_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": f"{step_loss:.4f}", "lr": f"{curr_lr:.2e}"})
                step_loss = 0.0

        avg_train_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch} finished. Average Train Loss: {avg_train_loss:.4f}")

        # Validation at end of epoch
        logger.info(f"Running validation for Epoch {epoch}...")
        val_metrics = evaluate(model, val_loader, device=device, classes=taxonomy_classes)

        val_macro_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]
        logger.info(
            f"Epoch {epoch} Validation: Macro-F1 = {val_macro_f1:.4f} | "
            f"Accuracy = {val_acc:.4f} | Loss = {val_metrics['loss']:.4f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "val_weighted_f1": round(val_metrics["weighted_f1"], 4),
            "val_accuracy": round(val_acc, 4),
        }
        training_history.append(epoch_record)

        # Track Best Checkpoint based on Validation Macro-F1
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            logger.info(f"New best validation Macro-F1 ({best_val_macro_f1:.4f})! Saving checkpoint...")
            model.save_pretrained(best_checkpoint_dir)
            tokenizer.save_pretrained(best_checkpoint_dir)
            with open(best_checkpoint_dir / "best_val_metrics.json", "w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=2)

    total_time = time.time() - start_train_time
    logger.info(f"Training completed in {total_time:.2f} seconds. Best Val Macro-F1: {best_val_macro_f1:.4f}")

    # 9. Final Evaluation on Held-Out Test Split using Best Checkpoint
    test_metrics = None
    if test_records:
        logger.info("Evaluating Best Model Checkpoint on Held-Out Test Set...")
        test_dataset = MisconceptionTorchDataset(test_records, tokenizer, label2id, max_length=args.max_length)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )

        # Load best model weights for test evaluation
        best_model = DebertaMisconceptionClassifier.from_pretrained(best_checkpoint_dir, device=device)
        test_metrics = evaluate(best_model, test_loader, device=device, classes=taxonomy_classes)

        print("\n" + "=" * 80)
        print("  HELD-OUT TEST SET EVALUATION RESULTS (DEBERTA-V3 PARTIAL TRAINING)")
        print("=" * 80)
        print(f"  Macro-F1 (Primary Metric): {test_metrics['macro_f1']:.4f}")
        print(f"  Weighted-F1:              {test_metrics['weighted_f1']:.4f}")
        print(f"  Accuracy:                 {test_metrics['accuracy']:.4f}")
        print(f"  Test Loss:                {test_metrics['loss']:.4f}")
        print("\nPer-Class Classification Report (Test Set):")
        print(test_metrics["classification_report_str"])

        # Save test metrics and confusion matrix
        with open(output_dir / "test_eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "macro_f1": round(test_metrics["macro_f1"], 4),
                "weighted_f1": round(test_metrics["weighted_f1"], 4),
                "accuracy": round(test_metrics["accuracy"], 4),
                "loss": round(test_metrics["loss"], 4),
                "per_class": test_metrics["classification_report_dict"],
            }, f, indent=2)

        with open(output_dir / "test_confusion_matrix.json", "w", encoding="utf-8") as f:
            json.dump({
                "classes": taxonomy_classes,
                "confusion_matrix": test_metrics["confusion_matrix"],
            }, f, indent=2)

    # 10. Save Full Training History
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_name": args.model_name,
            "strategy": "partial_training_@[quote]",
            "freeze_layers": args.freeze_layers,
            "loss_function": args.loss,
            "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
            "epochs": args.epochs,
            "best_val_macro_f1": round(best_val_macro_f1, 4),
            "total_time_seconds": round(total_time, 2),
            "history": training_history,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  [SUCCESS] DeBERTa-v3 training completed successfully!")
    print(f"  Best checkpoint stored at: {best_checkpoint_dir}")
    print(f"  Metrics and logs stored at: {output_dir}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
