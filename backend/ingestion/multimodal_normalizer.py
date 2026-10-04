"""
Re:Learn Multimodal Submission Normalizer
Normalizes diverse student inputs (typed text, MCQ options, OCR images)
into standardized payloads for Model 1 (Classifier) and Model 2 (Sequence Tracker).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from backend.ingestion.ocr_parser import HandwrittenOCRParser, OCRExtractionResult
from backend.ingestion.vlm_diagram_interpreter import VLMDiagramInterpreter, DiagramInterpretationResult
from backend.models.classifier import format_misconception_input

logger = logging.getLogger("relearn.multimodal_normalizer")


@dataclass
class NormalizedSubmission:
    """Standardized representation of any student submission across all modalities."""
    input_mode: str  # 'typed_text' | 'mcq_selection' | 'ocr_handwritten'
    question_text: str
    expected_solution: str
    final_answer: str
    working_steps: str
    discrete_steps: List[str]
    model1_formatted_input: str
    ocr_confidence: Optional[float] = None
    detected_signs: Dict[str, str] = field(default_factory=dict)
    diagram_evidence: Optional[str] = None
    diagram_result: Optional[DiagramInterpretationResult] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultimodalNormalizer:
    """
    Ingests raw payloads from frontend and normalizes into Model 1/2 inputs.
    Augments text with VLM diagram interpretation evidence when diagrams are supplied.
    """

    def __init__(
        self,
        ocr_parser: Optional[HandwrittenOCRParser] = None,
        vlm_interpreter: Optional[VLMDiagramInterpreter] = None,
    ) -> None:
        self.ocr_parser = ocr_parser or HandwrittenOCRParser()
        self.vlm_interpreter = vlm_interpreter or VLMDiagramInterpreter()

    def normalize(self, raw_payload: Dict[str, Any]) -> NormalizedSubmission:
        """
        Normalize raw request payload.
        Expected raw_payload structure:
        {
            "question_text": str,
            "expected_solution": str,
            "input_mode": "typed_text" | "mcq_selection" | "ocr_handwritten",
            "final_answer": str,
            "working_steps": Optional[str],
            "handwritten_image": Optional[str | bytes],  # base64 or bytes
            "selected_option": Optional[str],
        }
        """
        q_text = str(raw_payload.get("question_text") or "").strip()
        expected = str(raw_payload.get("expected_solution") or "").strip()
        input_mode = str(raw_payload.get("input_mode") or "typed_text").strip().lower()

        final_answer = str(
            raw_payload.get("final_answer")
            or raw_payload.get("student_final_answer")
            or raw_payload.get("student_answer")
            or ""
        ).strip()
        working_steps = str(
            raw_payload.get("working_steps")
            or raw_payload.get("student_working")
            or raw_payload.get("student_working_steps")
            or ""
        ).strip()
        discrete_steps: List[str] = []
        ocr_conf: Optional[float] = None
        detected_signs: Dict[str, str] = {}
        warnings: List[str] = []

        # Handle MCQ selection
        if input_mode == "mcq_selection":
            selected = raw_payload.get("selected_option") or final_answer
            final_answer = str(selected).strip()
            if not working_steps:
                working_steps = f"Selected MCQ option: '{final_answer}'."
            discrete_steps = [working_steps]

        # Handle OCR Handwritten submission
        elif input_mode in ("ocr_handwritten", "image", "handwritten"):
            img_data = raw_payload.get("handwritten_image")
            if img_data:
                ocr_res = self.ocr_parser.parse_image_data(img_data)
                ocr_conf = ocr_res.confidence
                detected_signs = ocr_res.detected_signs
                if ocr_res.quality_warning:
                    warnings.append(ocr_res.quality_warning)

                if ocr_res.extracted_steps:
                    discrete_steps = ocr_res.extracted_steps
                    working_steps = ocr_res.extracted_text
                elif not working_steps:
                    warnings.append("OCR failed to extract legible working steps.")
            else:
                # Fallback to working_steps text if image was pre-extracted
                ocr_res = self.ocr_parser.process_extracted_text(working_steps)
                ocr_conf = ocr_res.confidence
                detected_signs = ocr_res.detected_signs
                discrete_steps = ocr_res.extracted_steps

        # Handle standard typed text
        else:
            input_mode = "typed_text"
            if working_steps:
                ocr_res = self.ocr_parser.process_extracted_text(working_steps)
                detected_signs = ocr_res.detected_signs
                discrete_steps = ocr_res.extracted_steps
            else:
                warnings.append("No working steps provided.")

        # Handle optional diagram/graph image input via Vision-Language Model
        diagram_img = raw_payload.get("diagram_image") or raw_payload.get("diagram_data")
        diagram_evidence: Optional[str] = None
        diagram_res: Optional[DiagramInterpretationResult] = None

        if diagram_img:
            diagram_res = self.vlm_interpreter.interpret_diagram(
                image_input=diagram_img,
                question_context=q_text,
                diagram_hint=raw_payload.get("diagram_hint"),
            )
            diagram_evidence = diagram_res.evidence_text
            if diagram_res.visual_inconsistencies:
                warnings.extend([f"Visual inconsistency: {inc}" for inc in diagram_res.visual_inconsistencies])

        # Format input string for Model 1 (DeBERTa / Baseline)
        formatted_input = format_misconception_input(
            question_text=q_text,
            expected_solution=expected,
            final_answer=final_answer,
            working_steps=working_steps,
            diagram_evidence=diagram_evidence,
        )

        return NormalizedSubmission(
            input_mode=input_mode,
            question_text=q_text,
            expected_solution=expected,
            final_answer=final_answer,
            working_steps=working_steps,
            discrete_steps=discrete_steps,
            model1_formatted_input=formatted_input,
            ocr_confidence=ocr_conf,
            detected_signs=detected_signs,
            diagram_evidence=diagram_evidence,
            diagram_result=diagram_res,
            warnings=warnings,
            metadata={"raw_keys": list(raw_payload.keys())},
        )
