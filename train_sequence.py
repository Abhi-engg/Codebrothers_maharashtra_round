"""
Training and Evaluation CLI for Re:Learn Model 2 (Sequence Pattern Analyzer)
Trains both RuleBasedSequenceAnalyzer (baseline evaluation) and GRUSequenceAnalyzer (neural training).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from backend.models.sequence_model import (
    RuleBasedSequenceAnalyzer,
    GRUSequenceAnalyzer,
    SEQUENCE_PATTERN_CLASSES,
    load_taxonomy_classes,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("relearn.train_sequence")


class StudentSessionDataset(Dataset):
    """PyTorch Dataset for multi-step student sessions."""

    def __init__(self, sessions: List[Dict[str, Any]], analyzer: GRUSequenceAnalyzer, max_len: int = 8) -> None:
        self.sessions = sessions
        self.analyzer = analyzer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        session = self.sessions[idx]
        diag_idx, conf, pos, mask = self.analyzer.encode_session(session, max_len=self.max_len)

        pattern_str = session.get("pattern_type", "TRANSIENT_SLIP")
        pattern_target = self.analyzer.pattern2idx.get(pattern_str, 1)

        misc_str = session.get("persistent_misconception_id")
        if misc_str and misc_str in self.analyzer.diag2idx:
            misc_target = self.analyzer.diag2idx[misc_str]
        else:
            misc_target = self.analyzer.none_misc_idx

        return {
            "diag_indices": diag_idx,
            "confidences": conf,
            "step_positions": pos,
            "mask": mask,
            "pattern_target": torch.tensor(pattern_target, dtype=torch.long),
            "misc_target": torch.tensor(misc_target, dtype=torch.long),
        }


def stratified_session_split(
    sessions: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stratified train/val/test split based on pattern_type."""
    random.seed(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for s in sessions:
        p = s.get("pattern_type", "TRANSIENT_SLIP")
        buckets.setdefault(p, []).append(s)

    train, val, test = [], [], []
    for p, items in buckets.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def evaluate_rule_based(
    analyzer: RuleBasedSequenceAnalyzer,
    test_sessions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate deterministic rule-based baseline on test split."""
    y_true = [s["pattern_type"] for s in test_sessions]
    y_pred = [analyzer.analyze_session(s)["pattern_type"] for s in test_sessions]

    macro_f1 = f1_score(y_true, y_pred, labels=SEQUENCE_PATTERN_CLASSES, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, labels=SEQUENCE_PATTERN_CLASSES, output_dict=True, zero_division=0)

    logger.info("Rule-Based Baseline: Test Accuracy = %.4f | Macro-F1 = %.4f", acc, macro_f1)
    return {"accuracy": round(float(acc), 4), "macro_f1": round(float(macro_f1), 4), "report": rep}


def evaluate_gru(
    analyzer: GRUSequenceAnalyzer,
    data_loader: DataLoader,
) -> Tuple[float, float, float, Dict[str, Any], List[List[int]]]:
    """Evaluate GRU model on validation or test split."""
    analyzer.model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            diag_idx = batch["diag_indices"].to(analyzer.device)
            conf = batch["confidences"].to(analyzer.device)
            pos = batch["step_positions"].to(analyzer.device)
            mask = batch["mask"].to(analyzer.device)
            targets = batch["pattern_target"].to(analyzer.device)

            pattern_logits, _ = analyzer.model(diag_idx, conf, pos, mask)
            loss = criterion(pattern_logits, targets)
            total_loss += loss.item() * len(targets)

            preds = torch.argmax(pattern_logits, dim=-1).cpu().tolist()
            y_pred.extend([analyzer.idx2pattern[p] for p in preds])
            y_true.extend([analyzer.idx2pattern[t] for t in targets.cpu().tolist()])

    avg_loss = total_loss / max(1, len(data_loader.dataset))
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=SEQUENCE_PATTERN_CLASSES, average="macro", zero_division=0)
    rep = classification_report(y_true, y_pred, labels=SEQUENCE_PATTERN_CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=SEQUENCE_PATTERN_CLASSES).tolist()

    return avg_loss, acc, macro_f1, rep, cm


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate Re:Learn Model 2 Sequence Pattern Analyzer")
    parser.add_argument("--data", type=str, default="data/sequence_dataset.json", help="Path to sequence dataset")
    parser.add_argument("--taxonomy", type=str, default="data/taxonomy.json", help="Path to taxonomy JSON")
    parser.add_argument("--output-dir", type=str, default="checkpoints/sequence", help="Output directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    logger.info("Model 2 Training Device: %s (CUDA Available: %s)", device_str, torch.cuda.is_available())

    # Load dataset
    with open(args.data, "r", encoding="utf-8") as f:
        sessions = json.load(f)
    logger.info("Loaded %d sequence sessions from %s", len(sessions), args.data)

    # Split
    train_sess, val_sess, test_sess = stratified_session_split(sessions, seed=args.seed)
    logger.info("Stratified Splits: Train=%d | Val=%d | Test=%d", len(train_sess), len(val_sess), len(test_sess))

    # Evaluate Rule-Based Baseline first
    logger.info("Evaluating Rule-Based Sequence Analyzer Baseline...")
    rule_analyzer = RuleBasedSequenceAnalyzer(load_taxonomy_classes(args.taxonomy))
    rule_metrics = evaluate_rule_based(rule_analyzer, test_sess)

    # Initialize GRU Neural Analyzer
    logger.info("Initializing GRUSequenceAnalyzer (2-layer Bidirectional GRU + Attention Pooling)...")
    gru_analyzer = GRUSequenceAnalyzer(
        taxonomy_classes=load_taxonomy_classes(args.taxonomy),
        embedding_dim=32,
        hidden_dim=64,
        num_layers=2,
        device=device_str,
    )

    train_ds = StudentSessionDataset(train_sess, gru_analyzer)
    val_ds = StudentSessionDataset(val_sess, gru_analyzer)
    test_ds = StudentSessionDataset(test_sess, gru_analyzer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    criterion_pattern = nn.CrossEntropyLoss()
    criterion_misc = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(gru_analyzer.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    out_dir = Path(args.output_dir)
    best_dir = out_dir / "best"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    history = []

    logger.info("Starting GRU training for %d epochs...", args.epochs)
    for epoch in range(1, args.epochs + 1):
        gru_analyzer.model.train()
        train_loss = 0.0

        for batch in train_loader:
            diag_idx = batch["diag_indices"].to(gru_analyzer.device)
            conf = batch["confidences"].to(gru_analyzer.device)
            pos = batch["step_positions"].to(gru_analyzer.device)
            mask = batch["mask"].to(gru_analyzer.device)
            target_pattern = batch["pattern_target"].to(gru_analyzer.device)
            target_misc = batch["misc_target"].to(gru_analyzer.device)

            optimizer.zero_grad()
            pattern_logits, misc_logits = gru_analyzer.model(diag_idx, conf, pos, mask)

            loss_pattern = criterion_pattern(pattern_logits, target_pattern)
            loss_misc = criterion_misc(misc_logits, target_misc)
            loss = loss_pattern + 0.5 * loss_misc

            loss.backward()
            nn.utils.clip_grad_norm_(gru_analyzer.model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(target_pattern)

        scheduler.step()
        avg_train_loss = train_loss / len(train_ds)

        val_loss, val_acc, val_f1, _, _ = evaluate_gru(gru_analyzer, val_loader)
        logger.info(
            "Epoch %02d/%02d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f | Val Macro-F1: %.4f",
            epoch, args.epochs, avg_train_loss, val_loss, val_acc, val_f1,
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            gru_analyzer.save(best_dir)
            logger.info("Saved new best checkpoint to %s (Val Macro-F1: %.4f)", best_dir, val_f1)

    # Final Evaluation on Held-Out Test Split
    logger.info("Loading best GRU checkpoint for final evaluation on test set...")
    best_analyzer = GRUSequenceAnalyzer.load(best_dir, device=device_str)
    test_loss, test_acc, test_f1, test_rep, test_cm = evaluate_gru(best_analyzer, test_loader)

    print("\n" + "=" * 80)
    print("  HELD-OUT TEST SET EVALUATION RESULTS (MODEL 2 — GRU SEQUENCE ANALYZER)")
    print("=" * 80)
    print(f"  Macro-F1 (Primary Metric): {test_f1:.4f}")
    print(f"  Accuracy:                 {test_acc:.4f}")
    print(f"  Test Loss:                {test_loss:.4f}\n")

    print("Classification Report (Pattern Types):")
    print(f"{'Pattern Class':<28} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 68)
    for p in SEQUENCE_PATTERN_CLASSES:
        m = test_rep.get(p, {})
        print(f"{p:<28} {m.get('precision', 0.0):<10.4f} {m.get('recall', 0.0):<10.4f} {m.get('f1-score', 0.0):<10.4f} {int(m.get('support', 0)):<8}")

    # Save artifacts
    results = {
        "model": "GRUSequenceAnalyzer",
        "test_macro_f1": round(float(test_f1), 4),
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
        "rule_based_baseline": rule_metrics,
        "classification_report": test_rep,
    }
    with open(out_dir / "test_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    cm_data = {"labels": SEQUENCE_PATTERN_CLASSES, "matrix": test_cm}
    with open(out_dir / "test_confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(cm_data, f, indent=2)

    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info("Saved all evaluation metrics and confusion matrix to %s", out_dir)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    ret = main()
    os._exit(ret)
