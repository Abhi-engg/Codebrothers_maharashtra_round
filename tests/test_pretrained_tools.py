"""
Unit Tests for Pretrained Tools (no training needed):
1. PaddleOCR Engine: Reads handwriting and text from photos of worked solutions.
2. Vision-Language Model: Interprets diagrams and graphs (labels, axes, arrows), feeds evidence to DeBERTa without replacing it.
3. Qwen2.5-Instruct + RAG: Writes targeted explanations grounded in NCERT textbooks.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from PIL import Image

from backend.ingestion.paddle_ocr_engine import PaddleOCREngine, PaddleOCRResult
from backend.ingestion.vlm_diagram_interpreter import VLMDiagramInterpreter, DiagramInterpretationResult
from backend.ingestion.multimodal_normalizer import MultimodalNormalizer
from backend.models.classifier import format_misconception_input
from backend.intervention.rag_knowledge_base import TextbookRAGKnowledgeBase, TextbookKnowledgeChunk
from backend.intervention.generator import QwenInterventionGenerator, TargetedExplanation
from backend.models.combiner import DiagnosticDecision


class TestPaddleOCREngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PaddleOCREngine(enable_paddle=False)  # test fallback & preprocessing

    def test_image_preprocessing(self) -> None:
        """Verify image enhancement runs cleanly on PIL images."""
        test_img = Image.new("RGB", (200, 100), color="white")
        enhanced = self.engine.preprocess_image(test_img)
        self.assertEqual(enhanced.mode, "L")
        self.assertEqual(enhanced.size, (200, 100))

    def test_extract_from_image_fallback(self) -> None:
        """Verify extraction returns structured lines and confidence."""
        test_img = Image.new("RGB", (300, 150), color="white")
        res = self.engine.extract_from_image(test_img)
        self.assertIsInstance(res, PaddleOCRResult)
        self.assertGreater(len(res.lines), 0)
        self.assertGreaterEqual(res.mean_confidence, 0.5)
        self.assertIn("1/v", res.full_text)

    def test_invalid_input_handling(self) -> None:
        """Verify corrupt input returns safe empty result."""
        res = self.engine.extract_from_image(b"invalid_garbage")
        self.assertEqual(res.full_text, "")
        self.assertEqual(res.mean_confidence, 0.0)


class TestVLMDiagramInterpreter(unittest.TestCase):
    def setUp(self) -> None:
        self.vlm = VLMDiagramInterpreter(backend="heuristic")
        self.normalizer = MultimodalNormalizer(vlm_interpreter=self.vlm)

    def test_interpret_optics_ray_diagram(self) -> None:
        """Test ray diagram interpretation: labels, axes, and arrow paths."""
        img = Image.new("RGB", (400, 300), color="white")
        res = self.vlm.interpret_diagram(
            image_input=img,
            question_context="Concave mirror focal length 15 cm with f = +15 cm ray diagram",
        )
        self.assertEqual(res.diagram_type, "ray_diagram")
        self.assertIn("Pole P", res.detected_labels)
        self.assertIn("Focus F", res.detected_labels)
        self.assertGreater(len(res.arrows_and_paths), 0)
        self.assertIn("Diagram Type: ray_diagram", res.evidence_text)
        self.assertIn("positive sign inversion", res.evidence_text)

    def test_interpret_circuit_schematic(self) -> None:
        """Test circuit schematic interpretation."""
        img = Image.new("RGB", (400, 300), color="white")
        res = self.vlm.interpret_diagram(
            image_input=img,
            question_context="Find parallel resistance of two resistors R1 and R2 connected across battery V",
        )
        self.assertEqual(res.diagram_type, "circuit_schematic")
        self.assertIn("Resistor R1", res.detected_labels)
        self.assertIn("Rays/Arrows", res.evidence_text)

    def test_diagram_evidence_feeds_into_deberta_without_replacing(self) -> None:
        """
        Verify architectural requirement:
        VLM evidence feeds into DeBERTa formatted input string and DOES NOT replace it.
        """
        img = Image.new("RGB", (200, 200), color="white")
        raw_submission = {
            "question_text": "An object is placed in front of a concave mirror. Trace the rays.",
            "expected_solution": "Incident parallel ray reflects through focus F (f = -15 cm).",
            "student_final_answer": "v = +60 cm",
            "working_steps": "f = +15 cm, 1/v + 1/u = 1/f",
            "diagram_image": img,
            "input_mode": "typed_text",
        }
        norm = self.normalizer.normalize(raw_submission)

        # Check that diagram evidence is attached
        self.assertIsNotNone(norm.diagram_evidence)
        self.assertIn("Diagram Evidence:", norm.model1_formatted_input)

        # Check that standard DeBERTa fields are fully preserved
        self.assertIn("Question: An object is placed", norm.model1_formatted_input)
        self.assertIn("Expected Physics Summary: Incident parallel ray", norm.model1_formatted_input)
        self.assertIn("Student Final Answer: v = +60 cm", norm.model1_formatted_input)
        self.assertIn("Student Working Steps: f = +15 cm", norm.model1_formatted_input)


class TestQwenRAGInterventionGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.rag_kb = TextbookRAGKnowledgeBase()
        self.generator = QwenInterventionGenerator(backend="rag_synthesizer", rag_kb=self.rag_kb)

    def test_rag_knowledge_retrieval(self) -> None:
        """Verify exact NCERT textbook chunk retrieval for optics misconception."""
        chunk = self.rag_kb.retrieve_by_misconception_id("MISC-G10-OPT-01")
        self.assertIsNotNone(chunk)
        self.assertIn("NCERT Class 10", chunk.ncert_source)
        self.assertIn("New Cartesian Sign Convention", chunk.scientific_principle)
        self.assertIn("1/v + 1/u = 1/f", chunk.standard_formulae)

    def test_rag_electricity_retrieval(self) -> None:
        """Verify retrieval for parallel circuit misconception."""
        chunk = self.rag_kb.retrieve_by_misconception_id("MISC-G10-ELC-01")
        self.assertIsNotNone(chunk)
        self.assertIn("Chapter 12 'Electricity'", chunk.ncert_source)
        self.assertIn("1/Rp = 1/R1 + 1/R2", chunk.standard_formulae)

    def test_generate_targeted_intervention_3_steps(self) -> None:
        """
        Verify the 3-step intervention generated by Qwen RAG:
        1. Cognitive Conflict Probe
        2. Grounded NCERT Textbook Explanation
        3. Step-by-Step Interactive Whiteboard Resolution
        """
        decision = DiagnosticDecision(
            final_diagnosis="MISC-G10-OPT-01",
            final_confidence=0.96,
            decision_action="TRIGGER_INTERVENTION",
            pattern_type="PERSISTENT_MISCONCEPTION",
            rationale="Focal sign inversion detected consistently across multiple attempts.",
            model1_diagnosis="MISC-G10-OPT-01",
            model1_confidence=0.96,
            model2_pattern="PERSISTENT_MISCONCEPTION",
            model2_confidence=0.95,
            whiteboard_remediation_needed=True,
        )
        submission = {
            "question_text": "An object is 20 cm in front of a concave mirror of focal length 15 cm. Find image distance.",
            "final_answer": "v = +60 cm",
            "working_steps": "f = +15 cm, 1/v = 1/15 - 1/(-20)",
        }

        intervention = self.generator.generate_intervention(decision, submission)
        self.assertIsInstance(intervention, TargetedExplanation)
        self.assertEqual(intervention.misconception_id, "MISC-G10-OPT-01")
        self.assertIn("NCERT Class 10", intervention.ncert_reference)
        # Step 1: Socratic question
        self.assertTrue(len(intervention.cognitive_conflict_prompt) > 20)
        self.assertIn("?", intervention.cognitive_conflict_prompt)
        # Step 2: Textbook explanation
        self.assertIn("New Cartesian Sign Convention", intervention.textbook_grounded_explanation)
        # Step 3: Actionable remediation steps & whiteboard cue
        self.assertGreaterEqual(len(intervention.remediation_steps), 3)
        self.assertTrue(len(intervention.whiteboard_cue) > 10)

    def test_qwen_chat_prompt_construction(self) -> None:
        """Verify chat template formatting for Qwen2.5-Instruct."""
        decision = DiagnosticDecision(
            final_diagnosis="MISC-G10-OPT-01",
            final_confidence=0.95,
            decision_action="TRIGGER_INTERVENTION",
            pattern_type="PERSISTENT_MISCONCEPTION",
            rationale="Sign error.",
            model1_diagnosis="MISC-G10-OPT-01",
            model1_confidence=0.95,
            model2_pattern="PERSISTENT_MISCONCEPTION",
            model2_confidence=0.95,
        )
        chunk = self.rag_kb.retrieve_by_misconception_id("MISC-G10-OPT-01")
        prompt = self.generator._build_qwen_prompt(
            decision, chunk, "Find image distance", "f = 15 cm", "v = 60 cm"
        )
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("<|im_start|>assistant", prompt)
        self.assertIn("NCERT Class 10", prompt)


if __name__ == "__main__":
    unittest.main()
