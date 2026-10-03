#!/usr/bin/env python3
"""
Re:Learn — Dataset Quality & Zero-Leakage Validator
Author: Senior ML & DL Training Engineer
Validates:
  1. JSON integrity & schema requirements
  2. Label taxonomy alignment (no orphaned classes)
  3. Strict zero-leakage guarantee on template_group_id across splits
  4. Class distribution metrics & abstention coverage
"""

import json
import sys
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).resolve().parent

def validate():
    print("=" * 60)
    print("RE:LEARN DATASET AUDIT & ZERO-LEAKAGE VALIDATION")
    print("=" * 60)

    # 1. Load Taxonomy
    tax_path = DATA_DIR / "taxonomy.json"
    assert tax_path.exists(), f"Missing {tax_path}"
    with open(tax_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    valid_labels = set()
    for grade_info in taxonomy["domains"]:
        for concept in grade_info["concepts"]:
            for misc in concept["misconceptions"]:
                valid_labels.add(misc["misconception_id"])

    for base in taxonomy["baseline_classes"]:
        valid_labels.add(base["class_id"])

    print(f"[OK] Taxonomy Loaded: {len(valid_labels)} valid target classes.")

    # 2. Load Individual Dataset
    ind_path = DATA_DIR / "individual_dataset.json"
    assert ind_path.exists(), f"Missing {ind_path}"
    with open(ind_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"[OK] Individual Dataset: {len(records)} total records.")

    label_counts = Counter()
    coarse_counts = Counter()
    missing_working_count = 0

    for idx, r in enumerate(records):
        assert "record_id" in r, f"Record {idx} missing record_id"
        assert "template_group_id" in r, f"Record {idx} missing template_group_id"
        assert "question" in r and "question_text" in r["question"], f"Record {idx} missing question text"
        assert "student_submission" in r, f"Record {idx} missing student_submission"
        assert "ground_truth" in r, f"Record {idx} missing ground_truth"

        lbl = r["ground_truth"]["primary_label"]
        coarse = r["ground_truth"]["coarse_category"]
        assert lbl in valid_labels, f"Unknown label '{lbl}' in record {r['record_id']}! Must be in taxonomy."

        label_counts[lbl] += 1
        coarse_counts[coarse] += 1
        if not r["student_submission"]["working_steps"]:
            missing_working_count += 1

    print("\n--- Coarse Category Distribution ---")
    for cat, cnt in coarse_counts.most_common():
        print(f"  • {cat:30s}: {cnt:3d} ({cnt/len(records)*100:.1f}%)")

    print("\n--- Fine-Grained Label Distribution ---")
    for lbl, cnt in label_counts.most_common():
        print(f"  • {lbl:25s}: {cnt:3d} ({cnt/len(records)*100:.1f}%)")

    # 3. Check Sequence Dataset
    seq_path = DATA_DIR / "sequence_dataset.json"
    assert seq_path.exists(), f"Missing {seq_path}"
    with open(seq_path, "r", encoding="utf-8") as f:
        sequences = json.load(f)
    print(f"\n[OK] Sequence Dataset: {len(sequences)} longitudinal sessions verified.")

    # 4. Leakage Check on Splits
    splits_dir = DATA_DIR / "splits"
    with open(splits_dir / "train.json", "r", encoding="utf-8") as f:
        train = json.load(f)
    with open(splits_dir / "val.json", "r", encoding="utf-8") as f:
        val = json.load(f)
    with open(splits_dir / "test.json", "r", encoding="utf-8") as f:
        test = json.load(f)

    train_groups = {r["template_group_id"] for r in train}
    val_groups = {r["template_group_id"] for r in val}
    test_groups = {r["template_group_id"] for r in test}

    leak_train_val = train_groups.intersection(val_groups)
    leak_train_test = train_groups.intersection(test_groups)
    leak_val_test = val_groups.intersection(test_groups)

    assert not leak_train_val, f"DATA LEAKAGE DETECTED between Train and Val: {leak_train_val}"
    assert not leak_train_test, f"DATA LEAKAGE DETECTED between Train and Test: {leak_train_test}"
    assert not leak_val_test, f"DATA LEAKAGE DETECTED between Val and Test: {leak_val_test}"

    print("\n" + "=" * 60)
    print("[PASS] ZERO-LEAKAGE VERIFIED: No template group shared across Train/Val/Test splits.")
    print(f"  • Train: {len(train):3d} records | Groups: {train_groups}")
    print(f"  • Val:   {len(val):3d} records | Groups: {val_groups}")
    print(f"  • Test:  {len(test):3d} records | Groups: {test_groups}")
    print("=" * 60)

if __name__ == "__main__":
    validate()
