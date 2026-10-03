#!/usr/bin/env python3
"""
Re:Learn — Model Evaluation CLI
Author: Senior ML & DL Training Engineer

Evaluates a saved Model 1 checkpoint (Baseline or DeBERTa-v3) on any dataset split
or custom test file. Reports Macro-F1, Weighted-F1, Accuracy, Per-Class metrics,
and Confusion Matrix.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

from backend.models.classifier import (
    BaselineMisconceptionClassifier,
    extract_record_label,
    format_misconception_input,
    load_taxonomy_labels,
    SKLEARN_AVAILABLE,
    TORCH_AVAILABLE,
    TRANSFORMERS_AVAILABLE,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")


def load_dataset(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}, got {type(data)}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Re:Learn Model 1 (Baseline or DeBERTa-v3) on a Dataset Split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["baseline", "deberta"],
        default="baseline",
        help="Type of model to evaluate ('baseline' or 'deberta')",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to saved model checkpoint file (.joblib for baseline, directory for deberta)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/splits/test.json",
        help="Path to dataset split JSON to evaluate on",
    )
    parser.add_argument(
        "--taxonomy",
        type=str,
        default="data/taxonomy.json",
        help="Path to taxonomy JSON",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DeBERTa evaluation")
    parser.add_argument("--max-length", type=int, default=256, help="Max token sequence length for DeBERTa")
    parser.add_argument("--device", type=str, default="auto", help="Compute device for DeBERTa ('cuda', 'cpu', 'auto')")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save metrics JSON")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit evaluation samples for fast testing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 80)
    print(f"  RE:LEARN MODEL EVALUATION ({args.model_type.upper()})")
    print("=" * 80)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at: {checkpoint_path}")
        return 1

    # 1. Load Taxonomy
    taxonomy_classes = load_taxonomy_labels(args.taxonomy)
    logger.info(f"Loaded {len(taxonomy_classes)} taxonomy classes from {args.taxonomy}")

    # 2. Load Dataset
    records = load_dataset(args.data)
    if args.max_samples:
        records = records[: args.max_samples]
    logger.info(f"Loaded {len(records)} evaluation records from {args.data}")

    # 3. Format Features
    texts = [format_misconception_input(r) for r in records]
    labels = [extract_record_label(r) for r in records]

    # 4. Evaluate based on model type
    if args.model_type == "baseline":
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn is required to evaluate baseline model. Please install scikit-learn joblib.")
            return 1
        logger.info(f"Loading baseline checkpoint from: {checkpoint_path}")
        classifier = BaselineMisconceptionClassifier.load(checkpoint_path)
        eval_results = classifier.evaluate(texts, labels)

    elif args.model_type == "deberta":
        if not (TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE):
            logger.error("PyTorch and Transformers are required to evaluate DeBERTa model. Please install torch transformers.")
            return 1

        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer
        from backend.models.classifier import DebertaMisconceptionClassifier
        from train_deberta import MisconceptionTorchDataset, collate_fn_with_padding, evaluate as evaluate_deberta

        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

        logger.info(f"Loading DeBERTa model from: {checkpoint_path} on device {device}")
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        model = DebertaMisconceptionClassifier.from_pretrained(str(checkpoint_path), device=device)

        label2id = {cls: idx for idx, cls in enumerate(taxonomy_classes)}
        dataset = MisconceptionTorchDataset(records, tokenizer, label2id, max_length=args.max_length)
        collate = collate_fn_with_padding(tokenizer)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

        eval_results = evaluate_deberta(model, data_loader, device=device, classes=taxonomy_classes)

    else:
        logger.error(f"Unknown model type: {args.model_type}")
        return 1

    # 5. Print Results
    print("\n" + "=" * 50)
    print("  EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Macro-F1 (Primary Metric): {eval_results['macro_f1']:.4f}")
    print(f"  Weighted-F1:              {eval_results['weighted_f1']:.4f}")
    print(f"  Accuracy:                 {eval_results['accuracy']:.4f}")
    if "loss" in eval_results:
        print(f"  Loss:                     {eval_results['loss']:.4f}")

    print("\nPer-Class Classification Report:")
    print(eval_results["classification_report_str"])

    # 6. Save output if requested
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "model_type": args.model_type,
            "checkpoint": str(checkpoint_path),
            "data": str(args.data),
            "samples": len(records),
            "macro_f1": round(eval_results["macro_f1"], 4),
            "weighted_f1": round(eval_results["weighted_f1"], 4),
            "accuracy": round(eval_results["accuracy"], 4),
            "per_class": eval_results["classification_report_dict"],
            "confusion_matrix": eval_results["confusion_matrix"],
            "classes": taxonomy_classes,
        }
        if "loss" in eval_results:
            save_dict["loss"] = round(eval_results["loss"], 4)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_dict, f, indent=2)
        logger.info(f"Evaluation report successfully saved to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
