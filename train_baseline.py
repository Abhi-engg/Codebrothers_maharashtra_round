#!/usr/bin/env python3
"""
Re:Learn — Baseline Classifier Training CLI
Author: Senior ML & DL Training Engineer

Trains and evaluates Model 1 Fast Baseline (TF-IDF + Calibrated Logistic Regression)
on the Re:Learn physics misconception dataset splits.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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
    MisconceptionBaselineClassifier,
    extract_record_label,
    format_misconception_input,
    load_taxonomy_labels,
    SKLEARN_AVAILABLE,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_baseline")


def load_dataset(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}, got {type(data)}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train & Evaluate Re:Learn Model 1 Fast Baseline (TF-IDF + Calibrated Logistic Regression)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-data", type=str, default="data/splits/train.json", help="Path to train split JSON")
    parser.add_argument("--val-data", type=str, default="data/splits/val.json", help="Path to val split JSON")
    parser.add_argument("--test-data", type=str, default="data/splits/test.json", help="Path to test split JSON")
    parser.add_argument("--taxonomy", type=str, default="data/taxonomy.json", help="Path to taxonomy JSON")
    parser.add_argument("--output-dir", type=str, default="checkpoints/baseline", help="Directory to save checkpoint")
    parser.add_argument("--max-features", type=int, default=25000, help="Max TF-IDF vocabulary features")
    parser.add_argument("--ngram-max", type=int, default=3, help="Max n-gram size for TF-IDF (1, ngram_max)")
    parser.add_argument("--no-calibrate", action="store_true", help="Disable CalibratedClassifierCV probability calibration")
    parser.add_argument("--no-eval-test", action="store_true", help="Skip evaluation on held-out test split")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit train samples for fast debugging")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit val samples for fast debugging")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Limit test samples for fast debugging")
    return parser.parse_args()


def main() -> int:
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn and joblib are required for train_baseline.py. Please install: pip install scikit-learn joblib")
        return 1

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  RE:LEARN MODEL 1 — BASELINE TRAINING & EVALUATION (TF-IDF + LOGISTIC REGRESSION)")
    print("=" * 80)

    # 1. Load Taxonomy
    taxonomy_classes = load_taxonomy_labels(args.taxonomy)
    logger.info(f"Loaded {len(taxonomy_classes)} canonical taxonomy classes from {args.taxonomy}")

    # 2. Load Splits
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

    # 3. Format Features
    logger.info("Formatting text inputs: Question + Expected Physics + Student Answer + Working Steps...")
    train_texts = [format_misconception_input(r) for r in train_records]
    train_labels = [extract_record_label(r) for r in train_records]

    val_texts = [format_misconception_input(r) for r in val_records]
    val_labels = [extract_record_label(r) for r in val_records]

    logger.info(f"Train set: {len(train_texts)} records | Val set: {len(val_texts)} records")

    # 4. Instantiate & Train Model
    calibrate = not args.no_calibrate
    classifier = MisconceptionBaselineClassifier(
        classes=taxonomy_classes,
        ngram_range=(1, args.ngram_max),
        max_features=args.max_features,
        sublinear_tf=True,
        calibrate=calibrate,
        random_state=args.seed,
    )

    start_time = time.time()
    classifier.fit(train_texts, train_labels)
    fit_duration = time.time() - start_time
    logger.info(f"Training completed in {fit_duration:.2f} seconds.")

    # 5. Evaluate on Validation Set
    logger.info("Evaluating on Validation Set...")
    val_results = classifier.evaluate(val_texts, val_labels)

    print("\n" + "-" * 50)
    print("  VALIDATION SET EVALUATION RESULTS")
    print("-" * 50)
    print(f"  Macro-F1 (Primary Metric): {val_results['macro_f1']:.4f}")
    print(f"  Weighted-F1:              {val_results['weighted_f1']:.4f}")
    print(f"  Accuracy:                 {val_results['accuracy']:.4f}")
    print("\nPer-Class Classification Report (Validation):")
    print(val_results["classification_report_str"])

    # 6. Evaluate on Test Set
    test_results = None
    if test_records:
        logger.info("Evaluating on Held-Out Test Set...")
        test_texts = [format_misconception_input(r) for r in test_records]
        test_labels = [extract_record_label(r) for r in test_records]
        test_results = classifier.evaluate(test_texts, test_labels)

        print("\n" + "=" * 50)
        print("  HELD-OUT TEST SET EVALUATION RESULTS")
        print("=" * 50)
        print(f"  Macro-F1 (Primary Metric): {test_results['macro_f1']:.4f}")
        print(f"  Weighted-F1:              {test_results['weighted_f1']:.4f}")
        print(f"  Accuracy:                 {test_results['accuracy']:.4f}")
        print("\nPer-Class Classification Report (Test):")
        print(test_results["classification_report_str"])

    # 7. Save Model Checkpoint
    checkpoint_file = output_dir / "model.joblib"
    classifier.save(checkpoint_file)

    # 8. Save Metrics Summary JSON
    metrics_summary = {
        "model_type": "baseline_tfidf_logistic_regression",
        "training_time_seconds": round(fit_duration, 2),
        "calibrated": calibrate,
        "max_features": args.max_features,
        "ngram_range": [1, args.ngram_max],
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "validation_metrics": {
            "macro_f1": round(val_results["macro_f1"], 4),
            "weighted_f1": round(val_results["weighted_f1"], 4),
            "accuracy": round(val_results["accuracy"], 4),
            "per_class": val_results["classification_report_dict"],
        },
    }
    if test_results:
        metrics_summary["test_samples"] = len(test_records)
        metrics_summary["test_metrics"] = {
            "macro_f1": round(test_results["macro_f1"], 4),
            "weighted_f1": round(test_results["weighted_f1"], 4),
            "accuracy": round(test_results["accuracy"], 4),
            "per_class": test_results["classification_report_dict"],
        }
        # Save confusion matrix
        with open(output_dir / "test_confusion_matrix.json", "w", encoding="utf-8") as f:
            json.dump({
                "classes": taxonomy_classes,
                "confusion_matrix": test_results["confusion_matrix"],
            }, f, indent=2)

    with open(output_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info(f"Evaluation metrics saved to: {output_dir / 'eval_metrics.json'}")
    print(f"\n[SUCCESS] Baseline training completed! Checkpoint stored at: {checkpoint_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
