"""
Unit tests for Re:Learn Evidence Combiner & Abstention Logic
Tests the 3-tier pedagogical decision policy and sequence fusion.
"""

import unittest
from backend.models.combiner import (
    EvidenceCombiner,
    DiagnosticDecision,
    HIGH_CONFIDENCE_THRESHOLD,
    AMBIGUOUS_CONFIDENCE_THRESHOLD,
)


class TestEvidenceCombiner(unittest.TestCase):
    def setUp(self) -> None:
        self.combiner = EvidenceCombiner()

    def test_missing_working_abstention(self) -> None:
        m1 = {"primary_label": "MISC-G10-OPT-01", "confidence": 0.95}
        sub = {"working_steps": ""}
        decision = self.combiner.synthesize(m1, student_submission=sub)
        self.assertEqual(decision.decision_action, "ABSTAIN_UNCERTAIN")
        self.assertEqual(decision.final_diagnosis, "UNCERTAIN-GUESS")
        self.assertFalse(decision.whiteboard_remediation_needed)

    def test_low_confidence_abstention(self) -> None:
        m1 = {"primary_label": "MISC-G10-OPT-01", "confidence": 0.35}
        sub = {"working_steps": "1/v - 1/u = 1/f"}
        decision = self.combiner.synthesize(m1, student_submission=sub)
        self.assertEqual(decision.decision_action, "ABSTAIN_UNCERTAIN")
        self.assertEqual(decision.final_diagnosis, "UNCERTAIN-GUESS")

    def test_ambiguous_confidence_diagnostic_probe(self) -> None:
        m1 = {"primary_label": "MISC-G10-OPT-01", "confidence": 0.60}
        sub = {"working_steps": "1/v = 1/(-15) + 1/(-20)"}
        decision = self.combiner.synthesize(m1, student_submission=sub)
        self.assertEqual(decision.decision_action, "DIAGNOSTIC_PROBE")
        self.assertIsNotNone(decision.probe_question_prompt)
        self.assertFalse(decision.whiteboard_remediation_needed)

    def test_high_confidence_persistent_intervention(self) -> None:
        m1 = {"primary_label": "MISC-G10-OPT-01", "confidence": 0.90}
        m2 = {
            "pattern_type": "PERSISTENT_MISCONCEPTION",
            "persistent_misconception_id": "MISC-G10-OPT-01",
            "confidence": 0.95,
        }
        sub = {"working_steps": "1/v = 1/(-15) + 1/(-20)"}
        decision = self.combiner.synthesize(m1, m2, student_submission=sub)
        self.assertEqual(decision.decision_action, "TRIGGER_INTERVENTION")
        self.assertEqual(decision.final_diagnosis, "MISC-G10-OPT-01")
        self.assertTrue(decision.whiteboard_remediation_needed)
        self.assertGreaterEqual(decision.final_confidence, 0.90)

    def test_correct_resolving_trajectory_mastery(self) -> None:
        m1 = {"primary_label": "CORRECT", "confidence": 0.98}
        m2 = {"pattern_type": "RESOLVING_TRAJECTORY", "confidence": 0.95}
        sub = {"working_steps": "1/v = 1/(-15) - 1/(-20)"}
        decision = self.combiner.synthesize(m1, m2, student_submission=sub)
        self.assertEqual(decision.decision_action, "CONFIRM_MASTERY")
        self.assertEqual(decision.final_diagnosis, "CORRECT")

    def test_arithmetic_slip_no_action(self) -> None:
        m1 = {"primary_label": "SLIP-ARITHMETIC", "confidence": 0.92}
        sub = {"working_steps": "2 + 2 = 5"}
        decision = self.combiner.synthesize(m1, student_submission=sub)
        self.assertEqual(decision.decision_action, "NO_ACTION_NEEDED")
        self.assertFalse(decision.whiteboard_remediation_needed)


if __name__ == "__main__":
    unittest.main()
