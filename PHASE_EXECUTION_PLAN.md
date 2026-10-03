# Re:Learn — Phase-by-Phase Detailed Execution Plan

This document outlines the granular, end-to-end execution roadmap for developing **Re:Learn** (Class 10 Physics Adaptive Misconception Diagnosis & Targeted Intervention System). Each phase includes explicit subtasks, technical deliverables, data schemas, API contracts, verification tests, and edge-case handling.

---

## Roadmap Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       RE:LEARN PHASE IMPLEMENTATION PIPELINE                │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
  ┌────────────────────────────────────▼────────────────────────────────────┐
  │ PHASE 1: TAXONOMY & DATASET ENGINEERING                                 │
  │ • Define Class 10 Physics Taxonomy (Light & Electricity)                 │
  │ • Construct Individual Dataset (150+ labeled responses with working)   │
  │ • Construct Sequence Dataset (Ordered multi-question attempt logs)      │
  │ • Create Held-out Train/Val/Test Splits                                 │
  └────────────────────────────────────┬────────────────────────────────────┘
                                       │
  ┌────────────────────────────────────▼────────────────────────────────────┐
  │ PHASE 2: CORE ML MODELS & DIAGNOSIS LOGIC                               │
  │ • Model 1: Individual Classifier (TF-IDF + Calibrated Logistic/DeBERTa) │
  │ • Model 2: Sequence Pattern Analyzer (Transition Heuristics & GRU Net) │
  │ • Combiner Engine & Abstention Policy (Handling low confidence / unsure)│
  └────────────────────────────────────┬────────────────────────────────────┘
                                       │
  ┌────────────────────────────────────▼────────────────────────────────────┐
  │ PHASE 3: MULTIMODAL INGESTION & STRUCTURED WHITEBOARD ENGINE            │
  │ • Ingestion Pipeline: Text, MCQ & OCR equation/step extractor           │
  │ • Whiteboard Tool Contract: Normalized coordinates [0.0, 1.0]           │
  │ • Frontend Canvas / SVG Vector Renderer (draw_line, draw_arrow, etc.)   │
  └────────────────────────────────────┬────────────────────────────────────┘
                                       │
  ┌────────────────────────────────────▼────────────────────────────────────┐
  │ PHASE 4: TARGETED INTERVENTION & REASSESSMENT STATE MACHINE             │
  │ • Pedagogical Intervention Engine (RAG + Qwen2.5 Physics Remediation)   │
  │ • Reassessment Transfer Question Bank                                   │
  │ • Multi-Attempt Mastery Logic (Resolved / Still Present / Uncertain)   │
  │ • Persistent Learner Profile Database (SQLite/SQLAlchemy)              │
  └────────────────────────────────────┬────────────────────────────────────┘
                                       │
  ┌────────────────────────────────────▼────────────────────────────────────┐
  │ PHASE 5: FULL-STACK DEMO APPLICATION & INTEGRATION                      │
  │ • FastAPI Backend Microservice with REST endpoints                      │
  │ • Interactive Single Page Application (SPA)                             │
  │ • Real-time Quiz $\to$ Diagnosis $\to$ Whiteboard $\to$ Reassessment Flow│
  └────────────────────────────────────┬────────────────────────────────────┘
                                       │
  ┌────────────────────────────────────▼────────────────────────────────────┐
  │ PHASE 6: RIGOROUS EVALUATION, BENCHMARKING & DEMO HARDENING             │
  │ • Classification Evaluation (Precision, Recall, Macro-F1 per class)    │
  │ • Unseen Response Generalization & Identical-Answer Disambiguation Test │
  │ • End-to-End Demo Scenarios and Presentation Hardening                  │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Taxonomy & Dataset Engineering

### 1.1 Objective
Establish an authoritative, curriculum-aligned (NCERT Class 10 Physics) taxonomy of common conceptual misconceptions, and synthesize realistic student responses containing explicit numerical steps, reasoning paths, arithmetic slips, and multi-question sequences.

### 1.2 Subtasks & Deliverables
- [ ] **1.2.1 Taxonomy Definition (`data/taxonomy.json`):**
  - **Domain 1: Light — Reflection & Refraction**
    - `MISC-LGT-01`: Inversion of Cartesian Sign Convention (assuming focal length $f > 0$ for concave mirror or $u > 0$).
    - `MISC-LGT-02`: Lens vs. Mirror Formula Confusion (using $\frac{1}{v} - \frac{1}{u} = \frac{1}{f}$ for mirrors).
    - `MISC-LGT-03`: Persistent Real Image Assumption (believing concave mirrors *always* form real images, even when object is between Pole $P$ and Focus $F$).
    - `MISC-LGT-04`: Magnification Sign Error (using $m = -\frac{v}{u}$ for lenses or $m = +\frac{v}{u}$ for mirrors).
    - `MISC-LGT-05`: Refraction Direction Inversion (bending away from normal when entering optically denser medium).
  - **Domain 2: Electricity & Circuits**
    - `MISC-ELE-01`: Current Attenuation Fallacy (believing electric current gets "used up" across consecutive series resistors).
    - `MISC-ELE-02`: Voltage vs Current Conflation (assuming equal current splits equally in parallel branches regardless of resistance).
    - `MISC-ELE-03`: Equivalent Resistance Inverse Inversion (calculating $\frac{1}{R_p} = \frac{1}{R_1} + \frac{1}{R_2}$ and forgetting to take the reciprocal).
  - **Non-Misconception Baseline Classes:**
    - `CORRECT`: Correct solution with valid physical reasoning.
    - `SLIP-ARITH`: Correct physical setup and formula, but simple arithmetic calculation slip.
    - `SLIP-UNIT`: Correct physics, but failed unit conversion (e.g., cm to m, or mA to A).
    - `UNCERTAIN-GUESS`: Bare answer given with no working or contradictory guesses; insufficient evidence.

- [ ] **1.2.2 Individual Response Dataset (`data/individual_dataset.json`):**
  - Build 150+ granular labeled examples with fields:
    ```json
    {
      "id": "RESP-001",
      "question_id": "PHY-LGT-001",
      "chapter": "Light - Reflection",
      "question_text": "An object is placed 20 cm in front of a concave mirror of focal length 15 cm. Find the image distance v.",
      "correct_answer": "-60 cm",
      "correct_steps": "1/v = 1/f - 1/u = 1/(-15) - 1/(-20) = -1/15 + 1/20 = -1/60 => v = -60 cm",
      "student_answer": "+60 cm",
      "student_working": "1/v = 1/15 - 1/20 = 1/60 => v = 60 cm",
      "input_type": "text_with_steps",
      "misconception_id": "MISC-LGT-01",
      "misconception_name": "Inversion of Cartesian Sign Convention",
      "error_category": "conceptual",
      "reviewer_confidence": 1.0
    }
    ```
  - Include identical wrong answers arising from different roots:
    * *Example:* Student A gets $v = 8.57\text{ cm}$ because they used the lens formula instead of mirror formula (`MISC-LGT-02`).
    * *Example:* Student B gets $v = 8.57\text{ cm}$ because they subtracted reciprocals directly without finding common denominator (`SLIP-ARITH`).

- [ ] **1.2.3 Sequence Attempt Dataset (`data/sequence_dataset.json`):**
  - Synthesize 30+ ordered 3-to-5 question attempt sequences:
    * Session ID, student pseudonym, ordered question sequence.
    * Tracks whether errors are isolated (e.g. slips) or systematic (e.g. repeating sign errors across concave and convex lenses).

- [ ] **1.2.4 Data Splits:**
  - Create held-out train, validation, and test splits (80/10/10) partitioned by **question templates** (not random duplication) to measure true out-of-distribution generalization.

### 1.3 Acceptance Criteria
* `data/taxonomy.json`, `data/individual_dataset.json`, and `data/sequence_dataset.json` validate against JSON schema.
* Minimum 8 conceptual misconception classes + 4 baseline classes.
* Zero data leakage between question templates in train and test sets.

---

## Phase 2: Core Machine Learning Models & Diagnosis Logic

### 2.1 Objective
Implement **Model 1** (Individual Misconception Classifier), **Model 2** (Sequence Pattern Analyzer), and the **Abstention & Evidence Combiner** engine.

### 2.2 Subtasks & Deliverables
- [ ] **2.2.1 Model 1: Individual Classifier (`backend/models/classifier.py`):**
  - **Feature Extraction:** Concatenate `[Question Text] + [Correct Solution] + [Student Answer] + [Student Working]`.
  - **Classifier Pipeline:**
    - Baseline: TF-IDF n-grams $(1, 3)$ + Calibrated Logistic Regression / LinearSVC with probability calibration (`CalibratedClassifierCV`).
    - Transformer adapter: Support for `DeBERTa-v3-base` fine-tuning pipeline.
  - **Output:** Predicted misconception ID, class probability distribution, and confidence score.

- [ ] **2.2.2 Model 2: Sequence Pattern Analyzer (`backend/models/sequence_model.py`):**
  - **Rule-based Transition Baseline:**
    - Frequency of same error class across consecutive attempts.
    - Concept graph adjacency (e.g., if student fails both focal length sign and magnification sign, flag `Systematic Sign Convention Gap`).
  - **Sequence Model (GRU):**
    - Lightweight 2-layer GRU accepting fixed-dimension step embeddings $[e_1, e_2, \dots, e_t]$ to predict trajectory-level misconception patterns.
  - **Output:** Sequence-level pattern label, evidence summary list, pattern confidence score.

- [ ] **2.2.3 Evidence Combiner & Abstention Logic (`backend/models/combiner.py`):**
  - **Confidence Thresholds:**
    * $\text{Confidence} \ge 0.75$: Direct diagnosis accepted.
    * $0.45 \le \text{Confidence} < 0.75$: Ambiguous $\to$ flag top 2 candidates and generate a 1-step diagnostic probe question.
    * $\text{Confidence} < 0.45$ or empty working: Abstain $\to$ return `UNCERTAIN` and prompt user for working steps.

### 2.3 Acceptance Criteria
* Unit tests verify that Model 1 outputs valid probability distributions summing to $1.0$.
* Test case where student submits just a number with no working correctly triggers `UNCERTAIN` (abstention).
* Sequence model correctly identifies repeating error patterns across $Q_1 \to Q_2 \to Q_3$.

---

## Phase 3: Multimodal Ingestion & Structured Whiteboard Engine

### 3.1 Objective
Process diverse student input modalities (typed, MCQ, handwritten OCR) into normalized text, and build the backend-validated structured whiteboard rendering engine for visual physics explanations.

### 3.2 Subtasks & Deliverables
- [ ] **3.2.1 Multimodal Ingestion Service (`backend/ingestion/ocr_parser.py`):**
  - Ingestion endpoint accepting raw text, structured JSON, or base64 image data.
  - Mathematical OCR normalization: converts visual fractions (e.g. `1/v + 1/u = 1/f`), superscripts, and negative signs into standardized math strings.
  - Confidence scoring on OCR extraction; flags low-clarity images for user re-entry.

- [ ] **3.2.2 Structured Whiteboard API Contract (`backend/intervention/whiteboard_schema.py`):**
  - Defines strictly typed drawing primitives in normalized coordinates $[0.0, 1.0]$:
    * `draw_line(x1, y1, x2, y2, color, width, dash)`
    * `draw_arrow(x1, y1, x2, y2, label, color)`
    * `draw_shape(type, x, y, width, height, style)` — types: `concave_mirror`, `convex_lens`, `resistor`, `battery`, `ray_beam`
    * `draw_text(x, y, text, size, color, math_flag)`
    * `highlight_region(x, y, width, height, label)`
  - Schema validator ensuring no coordinate exceeds $[0.0, 1.0]$ and rejecting any arbitrary code.

- [ ] **3.2.3 Interactive Whiteboard Renderer (`frontend/js/whiteboard.js`):**
  - HTML5 Canvas & SVG dual renderer.
  - Translates normalized coordinates $[x, y]$ to canvas pixel space $[x \cdot W, y \cdot H]$.
  - Step-by-step animated stroke drawing sequence to visually illustrate the optical ray trace or circuit loop.

### 3.3 Acceptance Criteria
* Whiteboard schema rejects invalid JSON payloads or coordinates $< 0.0$ or $> 1.0$.
* Frontend renders clean ray diagrams (Principal axis, Concave mirror curve, Focal point $F$, Center $C$, and Ray arrows) without external heavy libraries.

---

## Phase 4: Targeted Intervention & Reassessment State Machine

### 4.1 Objective
Deliver pedagogically sound interventions tied specifically to the diagnosed misconception and implement the multi-attempt mastery verification logic.

### 4.2 Subtasks & Deliverables
- [ ] **4.2.1 Targeted Intervention Generator (`backend/intervention/generator.py`):**
  - Curated Intervention Bank mapped 1-to-1 to each misconception class in `taxonomy.json`.
  - LLM Prompt Adapter (Qwen2.5-Instruct / local LLM format):
    * Ingests: Diagnosed Misconception ID, Student Error, Correct Physics Principle, NCERT reference.
    * Generates: 3-sentence targeted explanation + structured whiteboard command sequence.

- [ ] **4.2.2 Transfer Reassessment Engine (`backend/reassessment/tracker.py`):**
  - Curated bank of isomorphic transfer questions testing the exact same physical concept in a modified setting (e.g., swapping values, changing mirror from concave to convex, or asking for object position given image position).

- [ ] **4.2.3 Multi-Attempt Mastery State Machine:**
  - Evaluates student response on transfer question:
    * **State 1: `Likely Resolved`** — Student succeeds on transfer question with valid physical reasoning.
    * **State 2: `Still Present`** — Student repeats the exact same misconception or falls into a connected trap.
    * **State 3: `Uncertain`** — Student gets the correct numerical answer through an unclear method or incomplete steps.
  - Enforces the golden rule: *One lucky guess never grants complete mastery.*

- [ ] **4.2.4 Learner Profile Database (`backend/database/db.py`):**
  - Lightweight SQLite database storing:
    * `students`: Student ID, username, overall proficiency.
    * `attempts`: Timestamp, question ID, student answer, extracted steps, Model 1 diagnosis, Model 2 sequence pattern, confidence.
    * `misconception_records`: Misconception ID, occurrence count, status (`active`, `improving`, `resolved`).

### 4.3 Acceptance Criteria
* Intervention generator generates relevant explanations specifically addressing the student's mistake.
* Reassessment logic changes student status to `Likely Resolved` only after verifying transfer question reasoning.

---

## Phase 5: Full-Stack Demo Application & API Integration

### 5.1 Objective
Connect all backend services and build a cohesive, responsive web UI for live demonstration and user interaction.

### 5.2 Subtasks & Deliverables
- [ ] **5.2.1 FastAPI Backend Endpoints (`backend/main.py`):**
  - `GET  /api/quiz/questions` — List available Class 10 Physics questions.
  - `POST /api/quiz/submit-answer` — Submit answer, working text, or image.
  - `POST /api/diagnose` — Run individual classifier + sequence analyzer + combiner.
  - `POST /api/intervention` — Retrieve targeted remediation + whiteboard commands.
  - `POST /api/reassessment/submit` — Submit transfer question answer and update state machine.
  - `GET  /api/learner/{student_id}/profile` — Fetch live student mastery status and history.

- [ ] **5.2.2 Interactive Web Frontend (`frontend/index.html`, `app.js`, `style.css`):**
  - Clean, responsive dashboard with distinct views:
    1. **Quiz & Working Input Panel:** Question prompt, formula hints, LaTeX rendering, typed step editor, and file upload zone for handwritten solutions.
    2. **Real-Time Diagnosis Card:** Reveals identified misconception, confidence gauge, evidence extracted from student steps, and comparison with common traps.
    3. **Interactive Visual Whiteboard:** Animated optical ray tracing or circuit diagram illustrating why the student's path deviated.
    4. **Reassessment Modal:** Follow-up transfer question to test resolution.
    5. **Learner Mastery Progress Dashboard:** Visual progress bars for Light & Electricity concepts.

### 5.3 Acceptance Criteria
* End-to-end flow runs seamlessly in a standard web browser without server errors.
* Latency for diagnosis + whiteboard payload generation under 500ms on local CPU.

---

## Phase 6: Rigorous Evaluation, Benchmarking & Demo Hardening

### 6.1 Objective
Produce quantitative evaluation metrics on held-out test data, verify system edge cases, and prepare a rock-solid live demonstration script.

### 6.2 Subtasks & Deliverables
- [ ] **6.2.1 Model Evaluation Suite (`evaluation/evaluate_models.py`):**
  - Runs inference on held-out test splits.
  - Computes and prints:
    * Per-class Precision, Recall, and F1-score.
    * Macro and Weighted F1-scores.
    * Confusion Matrix highlighting boundary cases (e.g. `MISC-LGT-01` vs `SLIP-ARITH`).
    * Abstention rate on noisy/working-free inputs.

- [ ] **6.2.2 Deterministic Edge-Case Scenarios (`evaluation/test_scenarios.py`):**
  - **Scenario 1 (Identical Wrong Answers, Different Roots):**
    Two inputs with answer `+10 cm` — one diagnosed as sign error, one as formula confusion.
  - **Scenario 2 (Sequence Pattern Detection):**
    Student fails $Q_1$, $Q_2$, and $Q_3$ with varying numerical errors; Model 2 detects the unifying pattern *"Conflates converging with diverging optics"*.
  - **Scenario 3 (Abstention on Empty Steps):**
    Student types wrong number with empty working; system outputs `UNCERTAIN` and asks for steps.
  - **Scenario 4 (Multi-Attempt Reassessment):**
    Student succeeds on transfer question; state updates to `Likely Resolved`.

- [ ] **6.2.3 Demo Script & Documentation:**
  - Update `README.md` with 1-click startup instructions (`pip install -r requirements.txt`, `uvicorn backend.main:app`).

### 6.3 Acceptance Criteria
* Model 1 achieves $> 80\%$ accuracy and $> 0.75$ Macro-F1 on held-out test set.
* All 4 edge-case test scenarios pass deterministically.

---

## Complete Project Directory Blueprint

```
d:\Codebrothers_maharashtra_round\
├── README.md                            # Quickstart & project summary
├── RELEARN_MASTER_SPECIFICATION.md      # High-level architecture & requirements
├── PHASE_EXECUTION_PLAN.md              # This granular execution roadmap
├── requirements.txt                     # Python dependencies (scikit-learn, fastapi, etc.)
├── data/
│   ├── taxonomy.json                    # Class 10 Physics misconception taxonomy
│   ├── individual_dataset.json          # 150+ labeled individual student responses
│   └── sequence_dataset.json            # Multi-question sequential student logs
├── backend/
│   ├── main.py                          # FastAPI application entrypoint
│   ├── config.py                        # System settings & confidence thresholds
│   ├── models/
│   │   ├── classifier.py                # Model 1: Individual misconception classifier
│   │   ├── sequence_model.py            # Model 2: Sequence pattern & GRU analyzer
│   │   └── combiner.py                  # Evidence synthesizer & abstention logic
│   ├── ingestion/
│   │   ├── ocr_parser.py                # Mathematical OCR & step normalizer
│   │   └── input_validator.py           # Submission schema validation
│   ├── intervention/
│   │   ├── generator.py                 # Targeted explanation & RAG engine
│   │   └── whiteboard_schema.py         # Whiteboard tool API & coordinate validator
│   ├── reassessment/
│   │   └── tracker.py                   # Transfer questions & mastery state machine
│   └── database/
│       └── db.py                        # SQLite database & student profile models
├── frontend/
│   ├── index.html                       # Single-page interface
│   ├── css/
│   │   └── style.css                    # UI styles & dark/light theme
│   └── js/
│       ├── app.js                       # Frontend state & API orchestration
│       ├── whiteboard.js                # HTML5 canvas vector diagram renderer
│       └── profile.js                   # Student progress & mastery tracking
└── evaluation/
    ├── evaluate_models.py               # Evaluation metrics (P, R, F1, confusion matrix)
    └── test_scenarios.py                # Deterministic end-to-end edge case suite
```
