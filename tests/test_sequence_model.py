"""
Unit tests for Re:Learn Model 2 (Sequence Pattern Analyzer)
Tests RuleBasedSequenceAnalyzer and GRUSequenceAnalyzer.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from backend.models.sequence_model import (
    RuleBasedSequenceAnalyzer,
    GRUSequenceAnalyzer,
    SEQUENCE_PATTERN_CLASSES,
    load_taxonomy_classes,
)
from train_sequence import stratified_session_split, StudentSessionDataset


class TestRuleBasedSequenceAnalyzer(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = RuleBasedSequenceAnalyzer()

    def test_empty_session_fallback(self) -> None:
        session = {"steps": []}
        res = self.analyzer.analyze_session(session)
        self.assertEqual(res["pattern_type"], "TRANSIENT_SLIP")
        self.assertIsNone(res["persistent_misconception_id"])

    def test_persistent_misconception_detection(self) -> None:
        session = {
            "steps": [
                {"step_index": 1, "model1_individual_diagnosis": "MISC-G10-OPT-01", "model1_confidence": 0.95},
                {"step_index": 2, "model1_individual_diagnosis": "MISC-G10-OPT-01", "model1_confidence": 0.92},
                {"step_index": 3, "model1_individual_diagnosis": "SLIP-ARITHMETIC", "model1_confidence": 0.85},
            ]
        }
        res = self.analyzer.analyze_session(session)
        self.assertEqual(res["pattern_type"], "PERSISTENT_MISCONCEPTION")
        self.assertEqual(res["persistent_misconception_id"], "MISC-G10-OPT-01")
        self.assertGreater(res["confidence"], 0.70)
        self.assertEqual(res["recommendation"], "TRIGGER_TARGETED_PEDAGOGICAL_INTERVENTION")

    def test_resolving_trajectory_detection(self) -> None:
        session = {
            "steps": [
                {"step_index": 1, "model1_individual_diagnosis": "MISC-G09-MOT-02", "model1_confidence": 0.95},
                {"step_index": 2, "model1_individual_diagnosis": "UNCERTAIN-GUESS", "model1_confidence": 0.50},
                {"step_index": 3, "model1_individual_diagnosis": "CORRECT", "model1_confidence": 0.98},
                {"step_index": 4, "model1_individual_diagnosis": "CORRECT", "model1_confidence": 0.99},
            ]
        }
        res = self.analyzer.analyze_session(session)
        self.assertEqual(res["pattern_type"], "RESOLVING_TRAJECTORY")
        self.assertEqual(res["persistent_misconception_id"], "MISC-G09-MOT-02")
        self.assertEqual(res["recommendation"], "CONFIRM_MASTERY_WITH_ADVANCED_TRANSFER")

    def test_transient_slip_detection(self) -> None:
        session = {
            "steps": [
                {"step_index": 1, "model1_individual_diagnosis": "CORRECT", "model1_confidence": 0.95},
                {"step_index": 2, "model1_individual_diagnosis": "SLIP-ARITHMETIC", "model1_confidence": 0.88},
                {"step_index": 3, "model1_individual_diagnosis": "CORRECT", "model1_confidence": 0.96},
            ]
        }
        res = self.analyzer.analyze_session(session)
        self.assertEqual(res["pattern_type"], "TRANSIENT_SLIP")
        self.assertIsNone(res["persistent_misconception_id"])
        self.assertEqual(res["recommendation"], "ENCOURAGE_CAREFUL_CALCULATION_NO_REMEDIATION")


class TestGRUSequenceAnalyzer(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.analyzer = GRUSequenceAnalyzer(device="cpu")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_encode_session_shapes(self) -> None:
        session = {
            "steps": [
                {"model1_individual_diagnosis": "MISC-G10-OPT-01", "model1_confidence": 0.9},
                {"model1_individual_diagnosis": "CORRECT", "model1_confidence": 0.95},
            ]
        }
        diag_idx, conf, pos, mask = self.analyzer.encode_session(session, max_len=6)
        self.assertEqual(diag_idx.shape, (6,))
        self.assertEqual(conf.shape, (6,))
        self.assertEqual(pos.shape, (6,))
        self.assertEqual(mask.shape, (6,))
        self.assertTrue(mask[0].item())
        self.assertTrue(mask[1].item())
        self.assertFalse(mask[2].item())

    def test_predict_session_schema(self) -> None:
        session = {
            "steps": [
                {"model1_individual_diagnosis": "MISC-G10-OPT-01", "model1_confidence": 0.9},
                {"model1_individual_diagnosis": "MISC-G10-OPT-01", "model1_confidence": 0.85},
            ]
        }
        pred = self.analyzer.predict_session(session)
        self.assertIn(pred["pattern_type"], SEQUENCE_PATTERN_CLASSES)
        self.assertIn("confidence", pred)
        self.assertIn("pattern_probabilities", pred)
        self.assertIn("recommendation", pred)
        self.assertEqual(len(pred["pattern_probabilities"]), 3)

    def test_save_and_load_roundtrip(self) -> None:
        save_path = Path(self.tmp_dir) / "gru_checkpoint"
        self.analyzer.save(save_path)

        loaded = GRUSequenceAnalyzer.load(save_path, device="cpu")
        session = {
            "steps": [
                {"model1_individual_diagnosis": "MISC-G09-MOT-02", "model1_confidence": 0.9},
                {"model1_individual_diagnosis": "CORRECT", "model1_confidence": 0.9},
            ]
        }
        pred1 = self.analyzer.predict_session(session)
        pred2 = loaded.predict_session(session)
        self.assertEqual(pred1["pattern_type"], pred2["pattern_type"])
        self.assertAlmostEqual(pred1["confidence"], pred2["confidence"], places=3)


class TestStratifiedSplit(unittest.TestCase):
    def test_stratified_split_counts(self) -> None:
        sessions = [
            {"session_id": f"s{i}", "pattern_type": "PERSISTENT_MISCONCEPTION"} for i in range(10)
        ] + [
            {"session_id": f"s{i+10}", "pattern_type": "TRANSIENT_SLIP"} for i in range(10)
        ] + [
            {"session_id": f"s{i+20}", "pattern_type": "RESOLVING_TRAJECTORY"} for i in range(10)
        ]
        train, val, test = stratified_session_split(sessions, train_ratio=0.7, val_ratio=0.15)
        self.assertEqual(len(train), 21)
        self.assertEqual(len(val), 3)
        self.assertEqual(len(test), 6)


if __name__ == "__main__":
    unittest.main()
