"""
Unit tests for Re:Learn Ingestion Pipeline
Tests HandwrittenOCRParser and MultimodalNormalizer.
"""

import unittest
from backend.ingestion.ocr_parser import HandwrittenOCRParser, OCRExtractionResult
from backend.ingestion.multimodal_normalizer import MultimodalNormalizer, NormalizedSubmission


class TestHandwrittenOCRParser(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = HandwrittenOCRParser()

    def test_equation_and_sign_extraction(self) -> None:
        text = "1/v - 1/u = 1/f => 1/v = 1/(-15) + 1/(-20); f = -15 cm; u = -20 cm"
        res = self.parser.process_extracted_text(text)
        self.assertGreater(len(res.extracted_steps), 1)
        self.assertIn("f", res.detected_signs)
        self.assertEqual(res.detected_signs["f"], "-")
        self.assertIn("u", res.detected_signs)
        self.assertEqual(res.detected_signs["u"], "-")
        self.assertTrue(res.has_fractions)
        self.assertGreaterEqual(res.confidence, 0.70)

    def test_empty_ocr_text_handling(self) -> None:
        res = self.parser.process_extracted_text("")
        self.assertEqual(res.extracted_text, "")
        self.assertEqual(len(res.extracted_steps), 0)
        self.assertEqual(res.confidence, 0.0)
        self.assertIsNotNone(res.quality_warning)

    def test_simulated_image_bytes(self) -> None:
        # Invalid small bytes
        res = self.parser.parse_image_data(b"abc")
        self.assertEqual(res.confidence, 0.0)

        # Mock PNG header
        mock_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        res = self.parser.parse_image_data(mock_png)
        self.assertGreater(res.confidence, 0.5)


class TestMultimodalNormalizer(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = MultimodalNormalizer()

    def test_typed_text_normalization(self) -> None:
        payload = {
            "question_text": "Find focal length of concave mirror.",
            "expected_solution": "f = -15 cm",
            "input_mode": "typed_text",
            "final_answer": "-15 cm",
            "working_steps": "1/v - 1/u = 1/f => f = -15",
        }
        sub = self.normalizer.normalize(payload)
        self.assertEqual(sub.input_mode, "typed_text")
        self.assertEqual(sub.final_answer, "-15 cm")
        self.assertIn("Question:", sub.model1_formatted_input)
        self.assertIn("Student Final Answer:", sub.model1_formatted_input)
        self.assertIn("Student Working Steps:", sub.model1_formatted_input)

    def test_mcq_selection_normalization(self) -> None:
        payload = {
            "question_text": "What is the unit of power of lens?",
            "expected_solution": "Dioptre",
            "input_mode": "mcq_selection",
            "selected_option": "Dioptre (D)",
        }
        sub = self.normalizer.normalize(payload)
        self.assertEqual(sub.input_mode, "mcq_selection")
        self.assertEqual(sub.final_answer, "Dioptre (D)")
        self.assertIn("Dioptre", sub.working_steps)

    def test_missing_working_warning(self) -> None:
        payload = {
            "question_text": "Calculate equivalent resistance.",
            "expected_solution": "4 ohms",
            "input_mode": "typed_text",
            "final_answer": "4 ohms",
            "working_steps": "",
        }
        sub = self.normalizer.normalize(payload)
        self.assertIn("No working steps provided.", sub.warnings)


if __name__ == "__main__":
    unittest.main()
