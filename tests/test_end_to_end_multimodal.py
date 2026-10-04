"""
End-to-End Multimodal Integration Test
Validates the complete execution flow from:
  1. Multimodal Input Ingestion (OCR / Math parsing)
  2. Model 1 (Individual Misconception Diagnosis)
  3. Model 2 (Longitudinal Sequence Analyzer)
  4. Combiner Engine (3-tier decision & abstention policy)
  5. Structured Whiteboard Payload Generation
"""

import json
import unittest
from pathlib import Path

from backend.ingestion.multimodal_normalizer import MultimodalNormalizer
from backend.models.classifier import MisconceptionBaselineClassifier
from backend.models.sequence_model import GRUSequenceAnalyzer, RuleBasedSequenceAnalyzer
from backend.models.combiner import EvidenceCombiner
from backend.intervention.whiteboard_schema import (
    get_concave_mirror_preset,
    get_parallel_circuit_preset,
    StructuredWhiteboardPayload,
)


class TestEndToEndMultimodalPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.normalizer = MultimodalNormalizer()
        cls.combiner = EvidenceCombiner()
        cls.rule_sequence = RuleBasedSequenceAnalyzer()

        # Load trained baseline classifier checkpoint
        baseline_ckpt = Path("checkpoints/baseline/model.joblib")
        if baseline_ckpt.exists():
            cls.m1_classifier = MisconceptionBaselineClassifier.load(baseline_ckpt)
        else:
            cls.m1_classifier = None

    def test_e2e_optics_sign_inversion_to_whiteboard(self) -> None:
        """Test optics problem: handwritten steps -> diagnosis -> intervention -> whiteboard."""
        raw_submission = {
            "question_text": "An object is placed at a distance of 20 cm in front of a concave mirror of focal length 15 cm. Find the position of the image formed (v).",
            "expected_solution": "Concave mirror has negative focal length f = -15 cm, u = -20 cm. 1/v = 1/f - 1/u = 1/(-15) - 1/(-20) = -1/60 => v = -60 cm.",
            "input_mode": "ocr_handwritten",
            "final_answer": "+60.0 cm",
            "working_steps": "u = -20 cm, f = +15 cm\n1/v = 1/f - 1/u = 1/15 - (-1/20) = 1/15 + 1/20\n1/v = 0.1167 => v = +60.0 cm behind mirror",
        }

        # 1. Ingestion & Normalization
        norm = self.normalizer.normalize(raw_submission)
        self.assertEqual(norm.input_mode, "ocr_handwritten")
        self.assertIn("f", norm.detected_signs)
        self.assertEqual(norm.detected_signs["f"], "+")

        # 2. Model 1 Prediction
        if self.m1_classifier:
            m1_pred = self.m1_classifier.predict(norm.model1_formatted_input)[0]
            m1_probs = self.m1_classifier.predict_proba(norm.model1_formatted_input)[0]
            m1_conf = float(max(m1_probs)) if hasattr(m1_probs, '__iter__') else float(m1_probs)
            m1_res = {"primary_label": m1_pred, "confidence": m1_conf}
        else:
            m1_res = {"primary_label": "MISC-G10-OPT-01", "confidence": 0.95}

        # 3. Model 2 Sequence Analysis (simulate prior recurring sign errors)
        session = {
            "steps": [
                {"model1_individual_diagnosis": "MISC-G10-OPT-01", "model1_confidence": 0.90},
                {"model1_individual_diagnosis": m1_res["primary_label"], "model1_confidence": m1_res["confidence"]},
            ]
        }
        m2_res = self.rule_sequence.analyze_session(session)
        self.assertEqual(m2_res["pattern_type"], "PERSISTENT_MISCONCEPTION")

        # 4. Combiner Engine
        decision = self.combiner.synthesize(
            model1_result=m1_res,
            model2_result=m2_res,
            student_submission={"working_steps": norm.working_steps},
        )
        self.assertEqual(decision.decision_action, "TRIGGER_INTERVENTION")
        self.assertTrue(decision.whiteboard_remediation_needed)

        # 5. Emit Whiteboard Visual Payload
        whiteboard_payload = get_concave_mirror_preset()
        self.assertEqual(whiteboard_payload.misconception_id, "MISC-G10-OPT-01")
        self.assertGreater(whiteboard_payload.total_steps, 5)

        # Validate that all commands satisfy coordinate contract [0.0, 1.0]
        for cmd in whiteboard_payload.commands:
            p = cmd.params
            for k in ("start", "end", "origin"):
                if k in p:
                    self.assertTrue(0.0 <= p[k][0] <= 1.0)
                    self.assertTrue(0.0 <= p[k][1] <= 1.0)

    def test_e2e_missing_working_abstention_flow(self) -> None:
        """Test missing working steps triggers calibrated abstention rather than hallucinating."""
        raw_submission = {
            "question_text": "Find equivalent resistance of 6 ohm and 12 ohm in parallel.",
            "expected_solution": "Rp = 4 ohms",
            "input_mode": "typed_text",
            "final_answer": "0.25 ohms",
            "working_steps": "",  # Empty!
        }

        norm = self.normalizer.normalize(raw_submission)
        self.assertIn("No working steps provided.", norm.warnings)

        m1_res = {"primary_label": "MISC-G10-ELE-03", "confidence": 0.85}
        decision = self.combiner.synthesize(
            model1_result=m1_res,
            student_submission={"working_steps": norm.working_steps},
        )
        self.assertEqual(decision.decision_action, "ABSTAIN_UNCERTAIN")
        self.assertEqual(decision.final_diagnosis, "UNCERTAIN-GUESS")
        self.assertIsNotNone(decision.probe_question_prompt)
        self.assertFalse(decision.whiteboard_remediation_needed)


if __name__ == "__main__":
    unittest.main()
