# Re:Learn — Master System Architecture & Implementation Blueprint

---

## 1. Problem Statement Extraction & Core Motivation

### 1.1 The Fundamental Flaw in Current EdTech
Modern educational technology platforms evaluate student submissions primarily via binary scoring:
$$\text{Status} \in \{\text{Correct}, \text{Incorrect}\}$$

When a student gives a wrong answer:
* Standard platforms provide generic solution text or ask the student to re-watch a lecture video.
* They fail to identify **why** the student made that specific mistake.
* They cannot distinguish between different mental models or misconceptions that happen to yield the exact same numerical result.
* They conflate simple arithmetic or unit slips with deep conceptual misconceptions.
* They treat a single correct follow-up answer as proof of total concept mastery.

### 1.2 Problem Statement Requirements (Mapped from PS & Official Appendix)
1. **Targeted Domain:** Class 10 Physics (e.g., *Light – Reflection & Refraction*, *Electricity*, or *Motion*).
2. **Multimodal Student Evidence:** Process typed text, selected MCQ options, written numerical working steps, and uploaded photos/diagrams (e.g., ray diagrams or circuit setups).
3. **Misconception Diagnosis:** Predict the exact underlying misconception from the taxonomy, not just a binary error tag.
4. **Differentiating Similar Errors:** Explicitly differentiate multiple distinct misconceptions that produce identical wrong numerical answers.
5. **Calibrated Confidence & Abstention:** When evidence is missing or ambiguous, the system must abstain, express calibrated uncertainty, or trigger an active diagnostic follow-up question instead of hallucinating certainty.
6. **Targeted Pedagogical Intervention:** Deliver personalized explanations and structured visual whiteboard drawings addressing the diagnosed misconception.
7. **Robust Reassessment & Mastery Verification:** Test the student using a novel transfer question in an altered context; verify whether the misconception is *Resolved*, *Still Present*, or *Uncertain* (avoiding the fallacy that one correct guess equals mastery).
8. **Learner Progression Model:** Maintain an auditable history of student attempts, recurring traps, and demonstrated understanding over time.

---

## 2. What Makes Our Solution Innovative & Unique

| Feature | Conventional Systems / Naive LLM Wrappers | **Re:Learn Innovative Architecture** |
| :--- | :--- | :--- |
| **Model Separation** | Single monolithic LLM prompt that diagnoses, explains, and grades in one pass (unstable, hallucination-prone). | **Decoupled Modular Pipeline:** OCR extracts features $\to$ DeBERTa-v3 diagnoses individual intent $\to$ GRU tracks temporal trajectory $\to$ Qwen generates guided intervention. |
| **Handling Ambiguity** | Forced guessing; model always picks an answer even without evidence. | **Explicit Abstention Policy:** Calibrated confidence scoring with intermediate 1-step diagnostic probes. |
| **Visual Whiteboard** | Either static images or raw JavaScript/Python generation that risks execution vulnerabilities. | **Safe Typed Tool Contract:** Normalized coordinates ($0.0 \to 1.0$) validated on backend before rendering on HTML5 Canvas. |
| **Multi-Question Context** | Treats each question in isolation as a stateless event. | **Sequence Pattern Analysis (Model 2):** Disentangles one-off arithmetic slips from persistent cognitive traps across attempts. |
| **Mastery Verification** | Assumes 1 correct answer on a follow-up means the student has mastered the concept. | **Multi-Attempt Reassessment Logic:** Evaluates conceptual transfer to confirm true resolution. |

---

## 3. High-Level System Architecture & Component Flow

```
                      [ STUDENT SUBMISSION ]
         (MCQ Choice / Typed Working / Uploaded Photo / Diagram)
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │       STAGE 1: MULTIMODAL INGESTION          │
         │   • Text & MCQ: Direct JSON normalization    │
         │   • Images: PaddleOCR equation/step parser   │
         │   • Diagrams: Vision feature parser          │
         └──────────────────────┬───────────────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
  ┌─────────────────────────────┐┌─────────────────────────────┐
  │  MODEL 1: INDIVIDUAL DIAGNOSIS││   MODEL 2: SEQUENCE ANALYZER│
  │ • DeBERTa-v3-base text model││ • Rules baseline + GRU net  │
  │ • Question: "What error in  ││ • Question: "What cognitive │
  │   this step explains result?││   trap recurs over time?"   │
  │ • Output: Misconception ID  ││ • Output: Sequence pattern  │
  │   + calibrated confidence   ││   + trajectory confidence   │
  └──────────────┬──────────────┘└──────────────┬──────────────┘
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
         ┌──────────────────────────────────────────────┐
         │   STAGE 2: EVIDENCE COMBINER & ABSTENTION    │
         │ • Reconciles single-step and sequence priors │
         │ • Confidence < Threshold ➔ Diagnostic Probe  │
         └──────────────────────┬───────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │        STAGE 3: TARGETED INTERVENTION        │
         │ • Qwen2.5-Instruct with RAG on Physics text  │
         │ • Emits Structured Whiteboard Instructions   │
         │   (draw_line, draw_arrow, draw_shape)        │
         └──────────────────────┬───────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │       STAGE 4: REASSESSMENT & LEARNER STORE  │
         │ • New transfer question in fresh context     │
         │ • Mastery: [Resolved | Still Present | Unsure]│
         │ • Update persistent learner profile in DB    │
         └──────────────────────────────────────────────┘
```

---

## 4. Deep-Dive Model Descriptions

### 4.1 Input Ingestion & Multimodal Parser
* **Core Philosophy:** *A single text model cannot directly read images, handwriting, and diagrams. Extract text and visual structures first; pass normalized evidence to the classifier.*
* **Components:**
  * **Text / MCQ Handler:** Extracts selected option, question context, and typed calculation steps.
  * **Handwritten OCR Engine (PaddleOCR):** Scans handwritten steps, identifies mathematical expressions, sign placements (e.g., negative signs in lens equations), and fractions.
  * **Diagram Feature Parser:** Extracts visual primitives (e.g., ray angles, arrow directions, virtual dotted lines, object vs image positions).
* **Payload Output:**
  ```json
  {
    "question_id": "PHY-LGT-004",
    "question_text": "An object is 20 cm in front of a concave mirror of focal length 15 cm. Find image distance.",
    "input_mode": "handwritten_image",
    "extracted_steps": "1/v - 1/u = 1/f => 1/v = 1/(-15) + 1/(-20)",
    "final_answer": "-8.57 cm",
    "extraction_confidence": 0.94
  }
  ```

---

### 4.2 Model 1: Multimodal Misconception Diagnosis (Individual Classifier)
* **Purpose:** Predict the most probable misconception explaining *this specific response*.
* **Base Architecture:** Fine-tuned `DeBERTa-v3-base` (or lightweight `DistilBERT` / Scikit-learn TF-IDF + Calibrated Classifier as high-speed baseline).
* **Input Representation:**
  $$\text{Input} = [\text{CLS}] \, Q_{\text{text}} \, [\text{SEP}] \, A_{\text{correct}} \, [\text{SEP}] \, S_{\text{response}} \, [\text{SEP}] \, W_{\text{extracted\_steps}} \, [\text{SEP}]$$
* **Output:** Softmax distribution over taxonomy classes + Calibrated Confidence Score:
  $$\hat{y} = \arg\max P(\text{Misconception}_k \mid \text{Input})$$
* **Distinguishing Edge Case (Identical Wrong Answers):**
  * *Question:* "An object is placed at $2F$ of a convex lens. Where is image formed?"
  * *Answer:* "At $2F$, virtual and upright."
  * *Diagnosis:* Distinguishes whether the student confused convex lens with concave mirror, or misunderstood the fundamental definition of real vs virtual inversion.

---

### 4.3 Model 2: Sequence-Based Pattern Analyzer
* **Purpose:** Inspect student trajectory across an ordered set of quiz questions ($Q_1 \to Q_2 \to Q_3 \to \dots$) to detect recurring or connected misconceptions that a single question cannot reveal.
* **Architecture:**
  * **Baseline (Phase 1):** Deterministic transition rules and frequency heuristics over past response features.
  * **Neural Model (Phase 2):** Lightweight 2-layer **Gated Recurrent Unit (GRU)** network reading sequential embedding vectors of student attempt states.
* **Input:** Ordered sequence of attempt vectors $X = [x_1, x_2, \dots, x_t]$, where each $x_i$ encodes:
  $$x_i = [\text{TopicEmbedding}, \text{ErrorType}, \text{Model1\_Prediction}, \text{Confidence}, \text{TimeTaken}]$$
* **Capabilities:**
  * Detects persistent cognitive traps (e.g., student consistently reverses Cartesian sign conventions for all converging optics).
  * Distinguishes isolated arithmetic slips from structural knowledge gaps.

---

### 4.4 Evidence Combiner & Abstention Logic
* **Purpose:** Synthesizes predictions from Model 1 and Model 2; prevents forced errors when confidence is low.
* **Decision Rules:**
  1. **High Confidence ($\ge 0.75$):** Accept diagnosis; dispatch targeted intervention.
  2. **Ambiguous Diagnosis ($0.45 \le \text{Confidence} < 0.75$ between 2 candidate labels):** Trigger an active **Diagnostic Follow-Up Probe** (a targeted 1-question check designed specifically to split the candidate hypotheses).
  3. **Low Confidence / Missing Working ($< 0.45$):** Mark as `Uncertain / Insufficient Evidence`. Prompt the student to show intermediate steps.

---

### 4.5 Intervention Engine & Structured Whiteboard
* **Purpose:** Generate targeted pedagogical remediation grounded in curriculum content.
* **LLM Model:** Pretrained `Qwen2.5-3B-Instruct` or `Qwen2.5-7B-Instruct` with RAG retrieval from NCERT Class 10 Physics content.
* **Safe Whiteboard API:** The LLM does **not** write raw executable code. It outputs structured JSON tool calls in normalized coordinates $(x, y) \in [0.0, 1.0]$:
  ```json
  {
    "explanation": "Remember the Cartesian sign convention: distances measured in the direction of incident light are positive, while distances against incident light are negative. For a concave mirror, focal length f is always negative.",
    "whiteboard_commands": [
      {"tool": "draw_axes", "params": {"origin": [0.5, 0.5], "scale": "cartesian"}},
      {"tool": "draw_shape", "params": {"type": "concave_mirror", "x": 0.5, "y": 0.2, "height": 0.6}},
      {"tool": "draw_arrow", "params": {"start": [0.5, 0.5], "end": [0.2, 0.5], "label": "f = -15cm", "color": "#E53E3E"}},
      {"tool": "draw_text", "params": {"x": 0.2, "y": 0.45, "text": "F Focus (Left of Pole)", "size": 14}}
    ]
  }
  ```

---

### 4.6 Reassessment Engine & Learner State Model
* **Purpose:** Validates that the misconception has been truly addressed.
* **Transfer Logic:** Presents an isomorphic question with different numerical values, altered context, or reverse reasoning (e.g., given image distance, find object position).
* **Mastery Classification State Machine:**
  * **Likely Resolved:** Correct answer and valid reasoning on transfer question.
  * **Still Present:** Repeats the same or connected misconception on transfer question.
  * **Uncertain:** Correct numerical answer but conflicting or omitted working.

---

## 5. Detailed Implementation Phases

### Phase 1: Dataset & Misconception Taxonomy Construction
* **Target Domain:** Class 10 Physics — *Light: Reflection & Refraction* and *Electricity & Circuits*.
* **Taxonomy Schema:**
  * `MISC-LGT-01`: Reversal of Cartesian sign convention for focal length.
  * `MISC-LGT-02`: Conflating mirror formula ($1/v + 1/u = 1/f$) with lens formula ($1/v - 1/u = 1/f$).
  * `MISC-LGT-03`: Believing concave mirrors always produce real/inverted images regardless of object position inside $F$.
  * `MISC-LGT-04`: Assuming magnification $m = -v/u$ for lenses instead of $+v/u$.
  * `MISC-ELE-01`: Believing electric current is consumed along resistors in a series loop.
  * `MISC-ELE-02`: Conflating potential difference with current flow.
  * `SLIP-ARITH`: Pure arithmetic calculation slip.
  * `SLIP-UNIT`: Unit conversion oversight (cm vs m, mA vs A).
  * `UNCERTAIN`: Insufficient working / arbitrary guess.
* **Datasets to Author:**
  1. `individual_dataset.json` (150+ granular labeled examples with student reasoning variations).
  2. `sequence_dataset.json` (Multi-step student quiz sessions testing recurring error transitions).

### Phase 2: Core Machine Learning Engines
* **Model 1 Pipeline:**
  * Scikit-Learn TF-IDF + Logistic Regression/SVM baseline with calibrated probability scores.
  * Integration harness for transformer-based inference (`DeBERTa-v3-base`).
* **Model 2 Pipeline:**
  * Rule-based sequence transition analyzer with sliding window evaluation.
  * PyTorch GRU architecture scaffold for training on sequential attempt vectors.
* **Abstention & Combiner Service:**
  * Threshold evaluator combining Model 1 probability and Model 2 pattern weights.

### Phase 3: Multimodal Ingestion & Whiteboard Renderer
* **Multimodal Parser:**
  * OCR mock/live parser handling handwritten mathematical equations and sign detection.
* **Structured Whiteboard Engine:**
  * HTML5 Canvas / SVG vector renderer parsing `draw_line`, `draw_shape`, `draw_arrow`, `draw_text`, and `undo`.
  * Coordinate transformer mapping $[0.0, 1.0] \to [W, H]$.

### Phase 4: Full-Stack Web Application & API Integration
* **Backend:** FastAPI service with endpoints:
  * `POST /api/quiz/submit-answer`
  * `POST /api/diagnose/individual`
  * `POST /api/diagnose/sequence`
  * `POST /api/intervention/generate`
  * `POST /api/reassessment/evaluate`
  * `GET /api/learner/profile/{id}`
* **Frontend:** Interactive Single Page Application (SPA):
  * Dynamic Quiz Interface (MCQ, working input box, and handwritten solution upload).
  * Live Misconception Diagnosis Card (showing identified misconception, confidence meter, and rationale).
  * Interactive Vector Whiteboard Canvas demonstrating the optical ray/circuit diagram step-by-step.
  * Reassessment Follow-Up Card & Mastery State Badge.

### Phase 5: Rigorous Evaluation & Benchmark Harness
* **Metrics:**
  * Per-class Precision, Recall, and Macro-F1 across all taxonomy labels.
  * Confusion matrix highlighting separation of identical wrong answers.
  * Abstention rate on ungrounded/working-free responses.
  * Pre- vs Post-Intervention learning gain tracking.

---

## 6. Project Directory Structure

```
Codebrothers_maharashtra_round/
├── RELEARN_MASTER_SPECIFICATION.md      # This master document
├── data/
│   ├── taxonomy.json                     # Formal physics misconception taxonomy
│   ├── individual_dataset.json           # Labeled single-response dataset
│   └── sequence_dataset.json             # Labeled multi-question sequence logs
├── backend/
│   ├── main.py                           # FastAPI application entrypoint
│   ├── config.py                         # System configuration & thresholds
│   ├── models/
│   │   ├── classifier.py                 # Model 1: Individual misconception diagnosis
│   │   ├── sequence_model.py             # Model 2: Sequence pattern & GRU analyzer
│   │   └── combiner.py                   # Evidence synthesizer & abstention logic
│   ├── ingestion/
│   │   ├── ocr_parser.py                 # PaddleOCR / handwriting math extractor
│   │   └── diagram_parser.py             # Diagram primitive extractor
│   ├── intervention/
│   │   ├── generator.py                  # RAG + LLM targeted explanation engine
│   │   └── whiteboard_schema.py          # Whiteboard tool API & coordinate validator
│   ├── reassessment/
│   │   └── tracker.py                    # Knowledge state machine & transfer question bank
│   └── database/
│       └── db.py                         # SQLite storage for student learner profiles
├── frontend/
│   ├── index.html                        # Modern, responsive UI layout
│   ├── css/
│   │   └── style.css                     # Custom design styles
│   └── js/
│       ├── app.js                        # Quiz flow, API orchestration & state manager
│       ├── whiteboard.js                 # HTML5 canvas vector renderer
│       └── student_profile.js            # Learning progress & misconception history view
└── evaluation/
    ├── evaluate_models.py                # Precision, Recall, F1 & Confusion Matrix suite
    └── test_scenarios.py                 # Deterministic end-to-end test cases
```
