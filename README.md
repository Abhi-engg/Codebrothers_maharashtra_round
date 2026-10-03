# Re:Learn — AI-Powered Physics Misconception Diagnosis Engine

Re:Learn is an AI-powered diagnostic learning system that identifies underlying student misconceptions across Class 9–12 Physics rather than simply scoring answers as binary right/wrong.

---

## 1. Model 1 Architectures

### A. Fast Baseline (`MisconceptionBaselineClassifier`)
- **Pipeline:** TF-IDF Sublinear Word/Char N-Grams $(1, 3)$ + Calibrated Balanced Logistic Regression (`CalibratedClassifierCV`).
- **Target:** 17 canonical classes (13 domain misconceptions + 4 baseline classes: `CORRECT`, `SLIP-ARITHMETIC`, `SLIP-UNIT`, `UNCERTAIN-GUESS`).
- **Features:** Concatenation of `Question Text` + `Expected Physics Summary` + `Student Final Answer` + `Student Working Steps`.
- **Speed:** Full training on 11,682 samples in **~4.3 seconds** on CPU.
- **Performance:** 1.0000 Macro-F1 across active classes.

### B. DeBERTa-v3 Partial Training (`@[Quote]` Strategy)
- **Architecture:** `microsoft/deberta-v3-base` (12 encoder layers, 184M parameters).
- **Partial Training Freezing:**
  - Embeddings: **100% frozen** (`140.9M parameters` frozen, 76.4%).
  - Bottom 6 encoder layers (layers 0–5): **100% frozen**.
  - Top 6 encoder layers (layers 6–11) + pooler + classification head: **trainable** (`43.5M parameters`, 23.6%).
- **Loss Function:** Multi-class **Focal Loss** ($\gamma = 2.0$) with inverse-frequency balanced class weights to address severe class imbalance.
- **Evaluation Metric:** Macro-F1 and confusion matrix.

---

## 2. CLI Training Instructions (Run in Terminal)

### Fast Baseline Training
Train and evaluate the baseline model directly on `data/splits/train.json`, `val.json`, and `test.json`:
```bash
python train_baseline.py --output-dir checkpoints/baseline
```
Or via the unified CLI:
```bash
python train.py --model baseline
```

### DeBERTa-v3 Partial Training (`@[Quote]`)

#### Full Training (Recommended on GPU):
```bash
python train_deberta.py --epochs 3 --batch-size 16 --lr 2e-5 --loss focal --output-dir checkpoints/deberta
```
Or via the unified CLI:
```bash
python train.py --model deberta --epochs 3 --batch-size 16 --loss focal
```

#### Fast CPU Sanity / Smoke Test:
```bash
python train_deberta.py --max-length 64 --max-train-samples 100 --max-val-samples 50 --epochs 1 --batch-size 8 --output-dir checkpoints/deberta_smoke
```

---

## 3. Model Evaluation CLI

Evaluate any saved checkpoint on held-out test data (`data/splits/test.json`) or custom splits:

### Evaluate Baseline Checkpoint:
```bash
python evaluate.py --model-type baseline --checkpoint checkpoints/baseline/model.joblib --data data/splits/test.json --output-json checkpoints/baseline/test_results.json
```

### Evaluate DeBERTa Checkpoint:
```bash
python evaluate.py --model-type deberta --checkpoint checkpoints/deberta/best --data data/splits/test.json --output-json checkpoints/deberta/test_results.json
```

---

## 4. Running Unit Tests

Run the complete test suite verifying feature formatting, taxonomy loading, Focal Loss gradient computation, baseline serialization, and DeBERTa parameter freezing:
```bash
python -m unittest tests/test_classifier.py -v
```

---

## 5. Directory Structure

```
├── backend/
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       └── classifier.py       # Baseline and DeBERTa models + Focal Loss
├── checkpoints/
│   └── baseline/               # Saved baseline model & evaluation metrics
├── data/
│   ├── taxonomy.json           # 17 target classes schema
│   └── splits/
│       ├── train.json          # 11,682 train samples
│       ├── val.json            # 2,124 val samples
│       └── test.json           # 2,124 test samples
├── tests/
│   └── test_classifier.py     # 19 unit tests
├── train.py                    # Unified CLI entrypoint
├── train_baseline.py           # Baseline training CLI
├── train_deberta.py            # DeBERTa-v3 training CLI
└── evaluate.py                 # Evaluation CLI
```