# Re:Learn — Senior ML/DL Full Review: PS Alignment, Multimodal Architecture & Dataset Audit

> **Reviewed by:** Senior ML & DL Training Engineer
> **Status:** CRITICAL ISSUES FOUND — Must fix before Phase 2 (Model Training) begins.

---

## Section 1: Problem Statement Re-Read (From Image)

The official PS (uploaded) states precisely:

> *"Develop Re:Learn, an AI-powered learning system that trains a model to analyze a learner's answers, working, or code and identify the underlying misconception rather than simply marking the response as incorrect."*

> **Key Feature 1 — Misconception Dataset:** Build a dataset of correct responses, incorrect responses, and their underlying misconceptions.

> **Key Feature 2 — Misconception Model:** Train and evaluate a model to classify or infer the misconception behind a learner's response.

> **Key Feature 3 — Misconception Differentiation:** Distinguish between different misconceptions that produce similar mistakes.

> **Key Feature 4 — Adaptive Intervention:** Generate or select an intervention based on the model's diagnosis.

> **Key Feature 5 — Resolution Assessment:** Determine whether the learner's underlying misconception has been resolved after intervention.

> **Key Feature 6 — Learner Model:** Track recurring misconceptions and demonstrated understanding across attempts.

> **Key Feature 7 — Model Evaluation:** Evaluate diagnosis accuracy and performance on responses or misconceptions not seen during training.

---

## Section 2: Architecture Review — What We Got Right

| PS Requirement | Our Architecture | Status |
| :--- | :--- | :--- |
| Misconception Dataset | `taxonomy.json` + `individual_dataset.json` + `sequence_dataset.json` | ✅ Strong |
| Misconception Model | DeBERTa / TF-IDF Classifier (Model 1) | ✅ Planned |
| Misconception Differentiation | Identical-wrong-answer pairs with different labels | ✅ Present in synthetic data |
| Adaptive Intervention | Qwen2.5 + RAG + Structured Whiteboard Tool Contract | ✅ Designed |
| Resolution Assessment | Multi-attempt state machine (`Resolved/Still Present/Uncertain`) | ✅ Designed |
| Learner Model | SQLite learner profile store across sessions | ✅ Designed |
| Model Evaluation | Macro-F1, Confusion Matrix, held-out unseen split | ✅ Planned |

---

## Section 3: Multimodal Architecture Review

### What the PS Calls "Multimodal"

The PS says: *"analyze a learner's answers, working, or code"*. This means the model must handle:

1. **Final answer** (MCQ option OR typed numerical result).
2. **Student working** (typed math steps OR handwritten scanned paper steps).
3. Optionally: drawn diagrams (ray diagrams, circuit sketches).

### What Our Current Architecture Actually Implements

```
[Student Submission] → [Input Mode Classifier]
     ├─ typed_text      → Direct text into DeBERTa tokenizer (READY)
     ├─ mcq_selection   → Question + choice text concatenation (READY)
     ├─ ocr_handwritten → PaddleOCR text extraction → DeBERTa (STUB ONLY)
     └─ canvas_drawn    → Vision parser → feature vector (NOT IMPLEMENTED)

[DeBERTa / TF-IDF Classifier] → Misconception Label + Confidence
[GRU Sequence Analyzer] → Pattern Label across Q1→Q2→Q3
[Abstention Logic] → If confidence < 0.45 → UNCERTAIN
[Qwen2.5 Intervention Generator] → Explanation + Whiteboard Commands
[Reassessment Transfer Question] → Resolution State Machine
```

### Critical Architecture Gap Identified

> The PS says *"multimodal"* — but the core ML diagnosis task **only needs text modality for the CBSE/Class 10 Physics domain**. The "multimodal" label refers to **input diversity** (MCQ + typed text + OCR scan), NOT a vision transformer or multi-encoder fusion architecture.

**Senior Engineer Verdict:** Do NOT chase ViT/CLIP for diagnosis. The multimodal design is correct — use PaddleOCR to extract text from images and pipe it through the same DeBERTa text classifier. Do not add architectural complexity that is not justified by data availability.

---

## Section 4: Dataset Audit — Critical Issues

> **CRITICAL** means: if not fixed, the trained model will be unreliable or mislead evaluators.
> **WARNING** means: real accuracy risk but model can still function.
> **NOTE** means: minor quality improvements.

---

### 🔴 CRITICAL ISSUE #1: 40 ScienceQA Records Have Trivial Placeholder Working Steps

**Audit Finding:**
```
ScienceQA records total: 40
ScienceQA records with trivial placeholder working steps: 40
```

Every single ScienceQA record has its `working_steps` set to:
```
"Selected distractor option: 'X' based on naive physical intuition."
```

**Why This Is a Critical Training Bug:**
- The model **must learn from the divergence between expected physics steps and the student's actual working**.
- If 20% of the dataset has fake one-sentence working steps, the model learns the **placeholder string as a misconception signal**, not genuine physics errors.
- During evaluation, a model trained on these will hallucinate wrong diagnoses.

**Fix:** For all ScienceQA records, replace the working steps with **topic-appropriate naive physical reasoning** generated from the question's known distractor pattern. This must come from the actual ScienceQA `solution` and `hint` fields.

---

### 🔴 CRITICAL ISSUE #2: Test Set Has Only 8 of 16 Labels — Half the Taxonomy Is Untestable

**Audit Finding:**
```
Labels MISSING from test set:
  MISC-G09-MOT-02, MISC-G10-OPT-03, MISC-G11-DYN-01,
  MISC-G12-EST-01, MISC-G11-KIN-01, MISC-G10-ELE-01,
  MISC-G09-GRV-01, MISC-G12-EMI-01
```

This means that when we run evaluation on the held-out test set:
- The model **cannot be evaluated on Grades 11 and 12 misconceptions at all**.
- The Macro-F1 metric will be computed on only 8 classes, hiding complete failure on the other 8.
- PS Requirement 7 (*"evaluate on misconceptions not seen during training"*) is currently **impossible to satisfy**.

**Root Cause:** The group-based K-Fold split assigned Grade 9–11 templates exclusively to Train, and all test templates happen to be Grade 10 Optics and Electricity.

**Fix:** Redesign the split so **every taxonomy label has at least 2 records in the test set**. This requires either generating more records per label or using **stratified group split** by label distribution.

---

### 🟡 WARNING #3: Severe Class Imbalance — `MISC-G09-MOT-02` is 23% of Dataset

**Audit Finding:**
```
MISC-G09-MOT-02 (Impetus Fallacy):  46 records  (23.0%)
MISC-G10-OPT-03 (Universal Inversion):  1 record  (0.5%)
MISC-G10-ELE-01 (Current Attenuation):  1 record  (0.5%)
```

A ratio of **46:1** between the most and least represented misconception class means:
- Any standard cross-entropy classifier will predict `MISC-G09-MOT-02` by default.
- Rare classes like `MISC-G10-OPT-03` and `MISC-G10-ELE-01` will effectively never be predicted.
- Accuracy will look falsely high because the majority class dominates.

**Fix Required:** Must use **Focal Loss** (already identified in Phase 2 plan) AND generate at minimum 8 additional records for the 3 under-represented classes (`MISC-G09-MOT-01`, `MISC-G10-OPT-03`, `MISC-G10-ELE-01`). Target: no class below 5 records.

---

### 🟡 WARNING #4: `MISC-G09-MOT-01` Defined in Taxonomy but Has ZERO Records

**Audit Finding:**
```
MISC-G09-MOT-01 (Distance vs Displacement Conflation): 0 records
```

A label that exists in the taxonomy with no training examples means:
- The classifier cannot learn its signal at all.
- If a student exhibits this misconception, the model will map it to the nearest neighbor class.

**Fix:** Add at least 5 records for `MISC-G09-MOT-01` with genuine student working examples (e.g., "round trip with displacement = 0 but student calculates average velocity using total distance").

---

### 🟡 WARNING #5: Sequence Dataset Lacks Diversity — Only 3 Topic Domains, Max 3 Steps

**Audit Finding:**
```
Steps per session: {3: 8, 2: 24}
Sequence topics: {
  'Light: Spherical Mirrors': 16,
  'Motion and Force': 8,
  'Electricity: Parallel Resistors': 8
}
```

- **24 out of 32 sessions have only 2 steps.** A GRU sequence model needs minimum 3 steps to learn meaningful temporal patterns. Two-step sequences are statistically indistinguishable from random.
- The sequence model should see Grade 11/12 topics but currently has **zero** sequences for Kinematics, Dynamics, or Electromagnetism.
- There is **no `RESOLVING_TRAJECTORY` pattern type** in the data (where a student starts wrong and gets correct after intervention). This means the reassessment engine has no training signal for successful learning.

**Fix:** Generate at least 15 sessions with 4–5 steps each, covering Grade 11/12 topics, and add minimum 5 `RESOLVING_TRAJECTORY` sessions.

---

### 🔵 NOTE #6: Grade 12 Coverage Is Weakest (Only 18 records = 9% of dataset)

```
Grade 9:  66 records
Grade 10: 87 records
Grade 11: 29 records
Grade 12: 18 records  ← Critically thin
```

Grade 12 has important, well-documented misconceptions (`MISC-G12-EST-01`, `MISC-G12-EMI-01`) that underpin PS-critical differentiation tasks. The coverage drop from Grade 11 to 12 is steep and will affect generalization to senior students.

---

## Section 5: Action Plan Before Phase 2 (Model Training)

These **must be resolved before any classifier is trained**. Training on a broken dataset produces a broken model that will mislead evaluators:

| Priority | Issue | Required Fix | Owner |
| :--- | :--- | :--- | :--- |
| 🔴 P0 | ScienceQA 40 records have fake working steps | Replace with authentic topic-reasoning narratives | Dataset Engineer |
| 🔴 P0 | Test set missing 8 of 16 labels | Redesign splits — stratified by label + grade | ML Engineer |
| 🟡 P1 | `MISC-G09-MOT-01` has zero records | Add minimum 5 records to `build_dataset.py` | Dataset Engineer |
| 🟡 P1 | `MISC-G10-OPT-03` and `MISC-G10-ELE-01` have only 1 record each | Augment to minimum 8 records per class | Dataset Engineer |
| 🟡 P1 | Sequence sessions only 2 steps (24/32 sessions) | Rebuild to minimum 3–5 steps per session | Dataset Engineer |
| 🟡 P1 | No `RESOLVING_TRAJECTORY` sessions | Add 5+ resolving sessions for reassessment training | Dataset Engineer |
| 🔵 P2 | Grade 12 only 18 records | Add 10 more Grade 12 physics records | Dataset Engineer |

---

## Section 6: What Is Correct and Should NOT Change

- ✅ **Taxonomy structure** is well-grounded in NCERT and PER research.
- ✅ **3-file data architecture** (taxonomy + individual + sequence) is correct for this task.
- ✅ **Group-based split principle** is the right leakage-prevention strategy.
- ✅ **Abstention class (`UNCERTAIN-GUESS`)** with 19 records is well-implemented.
- ✅ **Multimodal input design** (text + OCR → same text classifier) is architecturally sound.
- ✅ **Evidence combiner + confidence threshold** is the right approach before dispatching LLM.
- ✅ **Structured Whiteboard Tool Contract** (normalized coordinates) is the right safety boundary.
- ✅ **The 4 real-world sources** (CBSE, PER-FCI, PER-CSEM, ScienceQA) are the correct benchmarks — the problem is quality of integration, not the source choice.
