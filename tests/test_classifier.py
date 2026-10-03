"""
Re:Learn — Misconception Classifier Test Suite
Author: Senior ML & DL Training Engineer

Runs thorough unit and functional verification tests across:
1. Feature formatting & input concatenation
2. Taxonomy schema parsing (17 target classes)
3. Multi-class Focal Loss and class weighting
4. Baseline classifier training, calibration, inference, and serialization
5. DeBERTa-v3 Partial Training (@[Quote]) layer freezing and gradient flow
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

from backend.models.classifier import (
    DebertaMisconceptionClassifier,
    FocalLoss,
    MisconceptionBaselineClassifier,
    compute_balanced_class_weights,
    extract_record_label,
    format_misconception_input,
    load_taxonomy_labels,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = REPO_ROOT / "data" / "taxonomy.json"


class TestFeatureFormatting(unittest.TestCase):
    def test_format_from_record(self):
        record = {
            "question": {
                "question_text": "Calculate the equivalent resistance of two 6 ohm resistors in parallel.",
                "expected_physics_summary": "1/Rp = 1/6 + 1/6 = 2/6 = 1/3 => Rp = 3 ohms.",
            },
            "student_submission": {
                "final_answer": "12 ohms",
                "working_steps": "Rp = 6 + 6 = 12 ohms",
            },
        }
        text = format_misconception_input(record)
        self.assertIn("Calculate the equivalent resistance", text)
        self.assertIn("1/Rp = 1/6 + 1/6", text)
        self.assertIn("12 ohms", text)
        self.assertIn("Rp = 6 + 6 = 12 ohms", text)

    def test_format_from_kwargs(self):
        text = format_misconception_input(
            question_text="Q text",
            expected_summary="Exp summary",
            final_answer="Final ans",
            working_steps="Work steps",
        )
        self.assertIn("Question: Q text", text)
        self.assertIn("Expected Physics Summary: Exp summary", text)
        self.assertIn("Student Final Answer: Final ans", text)
        self.assertIn("Student Working Steps: Work steps", text)

    def test_format_with_empty_fields(self):
        text = format_misconception_input({})
        self.assertIn("Question: ", text)
        self.assertIn("Student Working Steps: ", text)

    def test_format_with_none_fields(self):
        # Must not raise AttributeError when question or student_submission is None
        text = format_misconception_input({"question": None, "student_submission": None})
        self.assertIn("Question: ", text)
        self.assertIn("Student Working Steps: ", text)

    def test_extract_record_label(self):
        self.assertEqual(extract_record_label({"ground_truth": {"primary_label": "MISC-1"}}), "MISC-1")
        self.assertEqual(extract_record_label({"ground_truth": None}), "UNCERTAIN-GUESS")
        self.assertEqual(extract_record_label({}), "UNCERTAIN-GUESS")
        self.assertEqual(extract_record_label(None), "UNCERTAIN-GUESS")


class TestTaxonomyLoading(unittest.TestCase):
    def test_load_taxonomy_classes(self):
        classes = load_taxonomy_labels(TAXONOMY_PATH)
        self.assertEqual(len(classes), 17, f"Expected 17 target classes, got {len(classes)}: {classes}")

        # Check key misconception IDs
        self.assertIn("MISC-G09-MOT-01", classes)
        self.assertIn("MISC-G10-OPT-01", classes)
        self.assertIn("MISC-G10-ELE-01", classes)
        self.assertIn("MISC-G12-EST-01", classes)

        # Check baseline classes
        self.assertIn("CORRECT", classes)
        self.assertIn("SLIP-ARITHMETIC", classes)
        self.assertIn("SLIP-UNIT", classes)
        self.assertIn("UNCERTAIN-GUESS", classes)


class TestFocalLoss(unittest.TestCase):
    def test_focal_loss_computation(self):
        loss_fn = FocalLoss(gamma=2.0, reduction="mean")
        logits = torch.randn(4, 17, requires_grad=True)
        targets = torch.tensor([0, 3, 10, 16], dtype=torch.long)

        loss = loss_fn(logits, targets)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)

        # Test backward pass
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())

    def test_focal_loss_with_weights(self):
        weights = torch.ones(17) * 2.0
        weights[0] = 5.0
        loss_fn = FocalLoss(gamma=2.0, weight=weights, reduction="mean")

        logits = torch.randn(2, 17)
        targets = torch.tensor([0, 1], dtype=torch.long)
        loss = loss_fn(logits, targets)
        self.assertGreater(loss.item(), 0.0)

    def test_balanced_class_weights(self):
        labels = [0, 0, 0, 1, 1, 2]
        weights = compute_balanced_class_weights(labels, num_classes=3)
        self.assertEqual(weights.shape[0], 3)
        # Class 2 is rarest, should have highest weight
        self.assertGreater(weights[2].item(), weights[0].item())
        # Mean weight should be approximately 1.0
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=4)

    def test_focal_loss_float_gamma_and_stability(self):
        # Non-integer gamma (e.g. 1.5) must not cause NaN due to float clamping
        loss_fn = FocalLoss(gamma=1.5, reduction="mean")
        logits = torch.randn(8, 17, requires_grad=True)
        targets = torch.randint(0, 17, (8,), dtype=torch.long)
        loss = loss_fn(logits, targets)
        self.assertFalse(torch.isnan(loss).any())
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertFalse(torch.isnan(logits.grad).any())

    def test_balanced_class_weights_edge_cases(self):
        # Empty labels
        w_empty = compute_balanced_class_weights([], num_classes=5)
        self.assertEqual(w_empty.shape[0], 5)
        self.assertTrue((w_empty == 1.0).all())

        # Out-of-bounds labels should not crash
        w_oob = compute_balanced_class_weights([0, 1, 99], num_classes=3)
        self.assertEqual(w_oob.shape[0], 3)
        self.assertAlmostEqual(w_oob.mean().item(), 1.0, places=4)


class TestBaselineClassifier(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.classes = ["CORRECT", "MISC-OPT", "UNCERTAIN-GUESS"]
        self.train_texts = [
            "Object at 20cm in front of concave mirror. Correct steps: 1/v = -1/60.",
            "Object at 20cm in front of concave mirror. Correct steps: 1/v = -1/60.",
            "Object at 20cm with concave mirror. Wrong sign used: 1/v = 1/60.",
            "Object at 20cm with concave mirror. Wrong sign used: 1/v = 1/60.",
            "Bare answer 60cm with no working steps provided.",
            "Bare answer 60cm with no working steps provided.",
        ]
        self.train_labels = [
            "CORRECT",
            "CORRECT",
            "MISC-OPT",
            "MISC-OPT",
            "UNCERTAIN-GUESS",
            "UNCERTAIN-GUESS",
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_baseline_fit_predict_and_evaluate(self):
        clf = MisconceptionBaselineClassifier(
            classes=self.classes,
            ngram_range=(1, 2),
            max_features=500,
            calibrate=True,
            calibration_cv=2,
            random_state=42,
        )
        clf.fit(self.train_texts, self.train_labels)

        # Predictions
        preds = clf.predict(self.train_texts)
        self.assertEqual(len(preds), len(self.train_texts))

        # Probabilities
        probs = clf.predict_proba(self.train_texts)
        self.assertEqual(probs.shape, (len(self.train_texts), len(self.classes)))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

        # Evaluation
        eval_dict = clf.evaluate(self.train_texts, self.train_labels)
        self.assertIn("macro_f1", eval_dict)
        self.assertIn("accuracy", eval_dict)
        self.assertIn("confusion_matrix", eval_dict)

        # Checkpoint serialization
        save_file = Path(self.temp_dir) / "baseline.joblib"
        clf.save(save_file)
        self.assertTrue(save_file.exists())

        # Checkpoint loading
        loaded_clf = MisconceptionBaselineClassifier.load(save_file)
        loaded_preds = loaded_clf.predict(self.train_texts)
        self.assertEqual(preds, loaded_preds)

    def test_baseline_single_string_prediction(self):
        clf = MisconceptionBaselineClassifier(classes=self.classes, calibration_cv=2)
        clf.fit(self.train_texts, self.train_labels)

        # Passing a single raw string must not crash or vectorize by character
        pred_single = clf.predict("Object at 20cm in front of concave mirror.")
        self.assertIsInstance(pred_single, list)
        self.assertEqual(len(pred_single), 1)

        prob_single = clf.predict_proba("Object at 20cm in front of concave mirror.")
        self.assertEqual(prob_single.shape, (1, len(self.classes)))

    def test_baseline_macro_f1_with_subset_classes(self):
        # When evaluating on a subset where not all classes appear,
        # macro_f1 must match the classification report's macro average across all self.classes
        clf = MisconceptionBaselineClassifier(classes=self.classes, calibration_cv=2)
        clf.fit(self.train_texts, self.train_labels)

        # Only 1 class present in this eval set
        subset_texts = [self.train_texts[0], self.train_texts[1]]
        subset_labels = ["CORRECT", "CORRECT"]

        eval_res = clf.evaluate(subset_texts, subset_labels)
        report_macro_avg = eval_res["classification_report_dict"]["macro avg"]["f1-score"]
        self.assertAlmostEqual(eval_res["macro_f1"], report_macro_avg, places=4)

    def test_baseline_evaluate_empty_dataset(self):
        clf = MisconceptionBaselineClassifier(classes=self.classes, calibration_cv=2)
        clf.fit(self.train_texts, self.train_labels)
        empty_res = clf.evaluate([], [])
        self.assertEqual(empty_res["macro_f1"], 0.0)
        self.assertEqual(empty_res["accuracy"], 0.0)

    def test_baseline_fit_with_unknown_labels(self):
        clf = MisconceptionBaselineClassifier(classes=self.classes, calibration_cv=2)
        # Includes an unknown label not in self.classes
        texts = self.train_texts + ["Random unknown text"]
        labels = self.train_labels + ["UNKNOWN_ANOMALOUS_LABEL"]
        # Must fit cleanly without KeyError
        clf.fit(texts, labels)
        self.assertTrue(clf.is_fitted)


class TestDebertaPartialTrainingStrategy(unittest.TestCase):
    def test_layer_freezing_and_parameter_status(self):
        # Test using mock config to avoid heavy download
        config = AutoConfig.from_pretrained("microsoft/deberta-v3-base", num_labels=17)
        mock_hf_model = AutoModelForSequenceClassification.from_config(config)

        # Create wrapper
        model = DebertaMisconceptionClassifier.__new__(DebertaMisconceptionClassifier)
        torch.nn.Module.__init__(model)
        model.encoder_model = mock_hf_model
        model.num_classes = 17
        model.freeze_embeddings = True
        model.freeze_layers_count = 6
        model.loss_fn = FocalLoss(gamma=2.0)

        # Apply freezing
        freeze_stats = model.apply_partial_training_freezing(
            freeze_embeddings=True,
            freeze_layers_count=6,
        )

        self.assertGreater(freeze_stats["frozen_params"], 0)
        self.assertGreater(freeze_stats["trainable_params"], 0)
        self.assertGreater(freeze_stats["frozen_params"], freeze_stats["trainable_params"])

        # Verify specifically:
        # 1. Embeddings are frozen
        for name, param in model.encoder_model.named_parameters():
            if "embeddings" in name and "rel_embeddings" not in name:
                self.assertFalse(param.requires_grad, f"Embedding parameter {name} should be frozen!")

        # 2. Bottom 6 layers (0 to 5) are frozen
        for name, param in model.encoder_model.named_parameters():
            if "encoder.layer." in name:
                layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_idx < 6:
                    self.assertFalse(param.requires_grad, f"Layer {layer_idx} parameter {name} should be frozen!")
                else:
                    self.assertTrue(param.requires_grad, f"Layer {layer_idx} parameter {name} should be trainable!")

        # 3. Classifier head is trainable
        for name, param in model.encoder_model.named_parameters():
            if "classifier" in name or "pooler" in name:
                self.assertTrue(param.requires_grad, f"Head parameter {name} should be trainable!")

    def test_forward_and_loss_calculation(self):
        config = AutoConfig.from_pretrained("microsoft/deberta-v3-base", num_labels=17)
        mock_hf_model = AutoModelForSequenceClassification.from_config(config)

        model = DebertaMisconceptionClassifier.__new__(DebertaMisconceptionClassifier)
        torch.nn.Module.__init__(model)
        model.encoder_model = mock_hf_model
        model.num_classes = 17
        model.loss_fn = FocalLoss(gamma=2.0)

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))
        labels = torch.tensor([1, 14], dtype=torch.long)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        self.assertIn("logits", out)
        self.assertIn("loss", out)
        self.assertEqual(out["logits"].shape, (batch_size, 17))
        self.assertTrue(torch.is_tensor(out["loss"]))
        self.assertGreater(out["loss"].item(), 0.0)

    def test_deberta_inference_methods(self):
        config = AutoConfig.from_pretrained("microsoft/deberta-v3-base", num_labels=17)
        mock_hf_model = AutoModelForSequenceClassification.from_config(config)

        model = DebertaMisconceptionClassifier.__new__(DebertaMisconceptionClassifier)
        torch.nn.Module.__init__(model)
        model.encoder_model = mock_hf_model
        model.num_classes = 17
        model.id2label = {i: f"CLASS_{i}" for i in range(17)}

        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))

        # Test predict_logits
        logits = model.predict_logits(input_ids=input_ids, attention_mask=attention_mask)
        self.assertEqual(logits.shape, (batch_size, 17))

        # Test predict_proba
        probs = model.predict_proba(input_ids=input_ids, attention_mask=attention_mask)
        self.assertEqual(probs.shape, (batch_size, 17))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5))

        # Test predict
        preds = model.predict(input_ids=input_ids, attention_mask=attention_mask)
        self.assertEqual(len(preds), batch_size)
        self.assertTrue(all(isinstance(p, str) for p in preds))


if __name__ == "__main__":
    unittest.main()
