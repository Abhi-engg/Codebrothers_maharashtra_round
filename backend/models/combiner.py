"""
Re:Learn Evidence Combiner & Calibrated Abstention Engine
Synthesizes predictions from:
  1. Model 1 (Individual Misconception Classifier: DeBERTa / Baseline)
  2. Model 2 (Sequence Pattern Analyzer: GRU / Rule-Based)

Applies the 3-tier Pedagogical Decision Policy:
  - High Confidence (>= 0.75) + Persistent Misconception -> TRIGGER_INTERVENTION
  - Ambiguous (0.45 <= conf < 0.75) -> DIAGNOSTIC_PROBE
  - Low Confidence (< 0.45) or Missing Working -> ABSTAIN_UNCERTAIN
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("relearn.combiner")

HIGH_CONFIDENCE_THRESHOLD = 0.75
AMBIGUOUS_CONFIDENCE_THRESHOLD = 0.45


@dataclass
class DiagnosticDecision:
    """Standardized output emitted by the Re:Learn Combiner Engine."""
    final_diagnosis: str
    final_confidence: float
    decision_action: str  # 'TRIGGER_INTERVENTION' | 'DIAGNOSTIC_PROBE' | 'ABSTAIN_UNCERTAIN' | 'CONFIRM_MASTERY' | 'NO_ACTION_NEEDED'
    pattern_type: str
    rationale: str
    model1_diagnosis: str
    model1_confidence: float
    model2_pattern: str
    model2_confidence: float
    probe_question_prompt: Optional[str] = None
    whiteboard_remediation_needed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvidenceCombiner:
    """
    Synthesizes single-step evidence with longitudinal trajectory history.
    """

    def __init__(
        self,
        high_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
        ambiguous_threshold: float = AMBIGUOUS_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.high_threshold = high_threshold
        self.ambiguous_threshold = ambiguous_threshold

    def synthesize(
        self,
        model1_result: Dict[str, Any],
        model2_result: Optional[Dict[str, Any]] = None,
        student_submission: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticDecision:
        """
        Produce a unified pedagogical diagnosis decision.
        """
        # Extract Model 1
        m1_label = model1_result.get("primary_label", "UNCERTAIN-GUESS")
        m1_conf = float(model1_result.get("confidence", 0.5))

        # Extract Model 2
        if model2_result:
            m2_pattern = model2_result.get("pattern_type", "TRANSIENT_SLIP")
            m2_conf = float(model2_result.get("confidence", 0.7))
            m2_misc = model2_result.get("persistent_misconception_id")
        else:
            m2_pattern = "TRANSIENT_SLIP"
            m2_conf = 0.5
            m2_misc = None

        # Check for missing working steps
        working = ""
        if student_submission:
            working = (student_submission.get("working_steps") or "").strip()

        # Rule 1: Abstention on Missing Working / Pure Guesswork
        if not working and m1_label != "CORRECT":
            return DiagnosticDecision(
                final_diagnosis="UNCERTAIN-GUESS",
                final_confidence=0.30,
                decision_action="ABSTAIN_UNCERTAIN",
                pattern_type=m2_pattern,
                rationale="Student provided no intermediate working steps. Unable to ground conceptual diagnosis.",
                model1_diagnosis=m1_label,
                model1_confidence=m1_conf,
                model2_pattern=m2_pattern,
                model2_confidence=m2_conf,
                probe_question_prompt="Please show your intermediate calculation steps or diagram to help us diagnose your approach.",
                whiteboard_remediation_needed=False,
            )

        # Rule 2: Low Confidence Abstention (< ambiguous threshold)
        if m1_conf < self.ambiguous_threshold:
            return DiagnosticDecision(
                final_diagnosis="UNCERTAIN-GUESS",
                final_confidence=m1_conf,
                decision_action="ABSTAIN_UNCERTAIN",
                pattern_type=m2_pattern,
                rationale=f"Model 1 diagnosis confidence ({m1_conf:.2f}) is below reliable threshold ({self.ambiguous_threshold}).",
                model1_diagnosis=m1_label,
                model1_confidence=m1_conf,
                model2_pattern=m2_pattern,
                model2_confidence=m2_conf,
                probe_question_prompt="Could you explain your reasoning in more detail?",
                whiteboard_remediation_needed=False,
            )

        # Rule 3: Correct response handling
        if m1_label == "CORRECT":
            if m2_pattern == "RESOLVING_TRAJECTORY":
                return DiagnosticDecision(
                    final_diagnosis="CORRECT",
                    final_confidence=max(m1_conf, m2_conf),
                    decision_action="CONFIRM_MASTERY",
                    pattern_type=m2_pattern,
                    rationale="Student successfully answered following prior misconception, indicating conceptual resolution.",
                    model1_diagnosis=m1_label,
                    model1_confidence=m1_conf,
                    model2_pattern=m2_pattern,
                    model2_confidence=m2_conf,
                    whiteboard_remediation_needed=False,
                )
            return DiagnosticDecision(
                final_diagnosis="CORRECT",
                final_confidence=m1_conf,
                decision_action="NO_ACTION_NEEDED",
                pattern_type=m2_pattern,
                rationale="Response is mathematically and conceptually correct.",
                model1_diagnosis=m1_label,
                model1_confidence=m1_conf,
                model2_pattern=m2_pattern,
                model2_confidence=m2_conf,
                whiteboard_remediation_needed=False,
            )

        # Rule 4: Execution Slip Handling (Arithmetic or Unit Error)
        if m1_label in ("SLIP-ARITHMETIC", "SLIP-UNIT"):
            return DiagnosticDecision(
                final_diagnosis=m1_label,
                final_confidence=m1_conf,
                decision_action="NO_ACTION_NEEDED",
                pattern_type=m2_pattern,
                rationale=f"Isolated computational slip ({m1_label}). No conceptual misconception detected.",
                model1_diagnosis=m1_label,
                model1_confidence=m1_conf,
                model2_pattern=m2_pattern,
                model2_confidence=m2_conf,
                whiteboard_remediation_needed=False,
            )

        # Rule 5: Ambiguous Conceptual Diagnosis (0.45 <= conf < 0.75)
        if self.ambiguous_threshold <= m1_conf < self.high_threshold:
            return DiagnosticDecision(
                final_diagnosis=m1_label,
                final_confidence=m1_conf,
                decision_action="DIAGNOSTIC_PROBE",
                pattern_type=m2_pattern,
                rationale=(
                    f"Candidate misconception {m1_label} detected with moderate confidence ({m1_conf:.2f}). "
                    f"Triggering diagnostic probe to confirm hypothesis."
                ),
                model1_diagnosis=m1_label,
                model1_confidence=m1_conf,
                model2_pattern=m2_pattern,
                model2_confidence=m2_conf,
                probe_question_prompt=f"Quick check: How did you choose the sign convention for this step?",
                whiteboard_remediation_needed=False,
            )

        # Rule 6: High Confidence Conceptual Misconception (>= 0.75)
        # Reconcile with longitudinal prior
        combined_confidence = m1_conf
        if m2_pattern == "PERSISTENT_MISCONCEPTION" and (m2_misc == m1_label or m2_misc is None):
            combined_confidence = min(0.99, m1_conf * 0.7 + m2_conf * 0.3)
            rationale_text = (
                f"Definitive diagnosis: {m1_label} (Confidence: {combined_confidence:.2f}). "
                f"Longitudinal history confirms this cognitive trap is recurring."
            )
        elif m2_pattern == "TRANSIENT_SLIP":
            combined_confidence = m1_conf * 0.90
            rationale_text = f"Misconception {m1_label} identified on this step, though student history previously showed isolated slips."
        else:
            rationale_text = f"Misconception {m1_label} identified with high confidence ({combined_confidence:.2f})."

        return DiagnosticDecision(
            final_diagnosis=m1_label,
            final_confidence=round(combined_confidence, 4),
            decision_action="TRIGGER_INTERVENTION",
            pattern_type=m2_pattern,
            rationale=rationale_text,
            model1_diagnosis=m1_label,
            model1_confidence=m1_conf,
            model2_pattern=m2_pattern,
            model2_confidence=m2_conf,
            whiteboard_remediation_needed=True,
        )
