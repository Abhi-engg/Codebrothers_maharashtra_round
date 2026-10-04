"""
Re:Learn Handwritten Equation & Working Steps OCR Parser
Extracts mathematical calculations, signs, fractions, and working lines from
student handwritten solution photos or scanned working papers.

Features:
  - Supports base64 images, filepaths, and raw bytes.
  - Mathematical syntax normalizer: standardizes fractions (1/v, 1/f), sign placements,
    exponents, and units (cm, m, ohms, A, V).
  - Confidence scoring on OCR extraction quality.
  - Fallback engine: handles degradation gracefully when camera quality or handwriting is poor.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from backend.ingestion.paddle_ocr_engine import PaddleOCREngine, PaddleOCRResult

logger = logging.getLogger("relearn.ocr_parser")


@dataclass
class OCRExtractionResult:
    """Standardized output of OCR extraction from student handwritten working."""
    extracted_text: str
    extracted_steps: List[str]
    confidence: float
    detected_equations: List[str]
    detected_signs: Dict[str, str]  # e.g. {'f': '-', 'u': '-', 'v': '-'}
    has_fractions: bool
    quality_warning: Optional[str] = None
    engine_used: str = "paddleocr_or_heuristic"
    metadata: Dict[str, Any] = field(default_factory=dict)


class HandwrittenOCRParser:
    """
    Parser for handwritten student math and physics working steps using PaddleOCR.
    """

    # Physics mathematical pattern anchors
    EQUATION_PATTERN = re.compile(
        r"(?:1\/[vuf]|v|u|f|Rp|Rs|I|V|R\d?|a|F|m|g)\s*=\s*[^;\n,]+",
        re.IGNORECASE,
    )
    SIGN_ASSIGNMENT_PATTERN = re.compile(
        r"\b([vufI]|Rp|Rs)\s*=\s*([+-]?\s*[\d\.\/]+)",
        re.IGNORECASE,
    )
    FRACTION_PATTERN = re.compile(r"\b\d+\s*\/\s*[-+]?\d+\b|1\/[-+]?[a-z\d]+", re.IGNORECASE)

    def __init__(
        self,
        min_confidence_threshold: float = 0.60,
        paddle_engine: Optional[PaddleOCREngine] = None,
    ) -> None:
        self.min_confidence_threshold = min_confidence_threshold
        self.paddle_engine = paddle_engine or PaddleOCREngine(min_confidence=min_confidence_threshold)

    def parse_image_data(self, image_input: Union[str, bytes, Path]) -> OCRExtractionResult:
        """
        Main entrypoint: parses image (filepath, base64 data URL, or raw bytes).
        """
        raw_text, base_conf = self._extract_raw_text(image_input)
        return self.process_extracted_text(raw_text, initial_confidence=base_conf)

    def _extract_raw_text(self, image_input: Union[str, bytes, Path]) -> Tuple[str, float]:
        """
        Extracts raw text strings from image input using OCR engine or simulated parser.
        """
        if isinstance(image_input, bytes):
            return self._ocr_bytes(image_input)

        if isinstance(image_input, Path) or (isinstance(image_input, str) and not image_input.startswith("data:image") and "\n" not in image_input and Path(image_input).exists()):
            try:
                with open(Path(image_input), "rb") as f:
                    return self._ocr_bytes(f.read())
            except Exception as e:
                logger.warning("Failed to read image file %s: %s", image_input, e)
                return "", 0.0

        if isinstance(image_input, str):
            clean_str = image_input.strip()
            # 1. If explicit data URI
            if clean_str.startswith("data:image"):
                try:
                    payload = clean_str.split(",", 1)[1] if "," in clean_str else clean_str
                    raw_bytes = base64.b64decode(payload)
                    return self._ocr_bytes(raw_bytes)
                except Exception as e:
                    logger.warning("Failed to decode data URI base64: %s", e)
                    return "", 0.0

            # 2. Check if string is base64 encoded image
            if "\n" not in clean_str and len(clean_str) >= 16 and (len(clean_str) % 4 == 0 or "=" in clean_str):
                try:
                    raw_bytes = base64.b64decode(clean_str)
                    if len(raw_bytes) >= 8 and (
                        raw_bytes.startswith(b"\x89PNG")
                        or raw_bytes.startswith(b"\xff\xd8\xff")
                        or raw_bytes.startswith(b"GIF")
                        or raw_bytes.startswith(b"RIFF")
                        or raw_bytes.startswith(b"BM")
                    ):
                        return self._ocr_bytes(raw_bytes)
                except Exception:
                    pass

            # 3. Otherwise, treat as directly provided mathematical / step text
            return image_input, 0.90

        return "", 0.0

    def _ocr_bytes(self, image_bytes: bytes) -> Tuple[str, float]:
        """
        Extracts handwriting via PaddleOCR engine (or fallback).
        """
        # Check if bytes are valid image header (JPEG, PNG)
        if len(image_bytes) < 8:
            return "", 0.0

        is_png = image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        is_jpeg = image_bytes.startswith(b"\xff\xd8\xff")

        if not (is_png or is_jpeg):
            return "Unable to decode image format", 0.20

        # Delegate to PaddleOCREngine
        res = self.paddle_engine.extract_from_image(image_bytes)
        return res.full_text, res.mean_confidence

    def process_extracted_text(self, text: str, initial_confidence: float = 0.85) -> OCRExtractionResult:
        """
        Cleans and parses raw OCR text into structured mathematical equations,
        signs, fractions, and discrete working steps.
        """
        if not text or not text.strip():
            return OCRExtractionResult(
                extracted_text="",
                extracted_steps=[],
                confidence=0.0,
                detected_equations=[],
                detected_signs={},
                has_fractions=False,
                quality_warning="No text could be extracted from image.",
            )

        # 1. Clean OCR artifacts
        cleaned = text.strip()
        cleaned = cleaned.replace("—", "-").replace("–", "-")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)

        # Split into distinct calculation lines/steps
        raw_lines = re.split(r"[\n;]|\s*=>\s*|\s*->\s*", cleaned)
        steps = [line.strip() for line in raw_lines if line.strip()]

        # 2. Extract Equations
        detected_eqs = self.EQUATION_PATTERN.findall(cleaned)

        # 3. Detect Signs assigned to key physical quantities
        signs: Dict[str, str] = {}
        for var, val in self.SIGN_ASSIGNMENT_PATTERN.findall(cleaned):
            val_clean = val.replace(" ", "")
            if val_clean.startswith("-"):
                signs[var.lower()] = "-"
            elif val_clean.startswith("+"):
                signs[var.lower()] = "+"
            else:
                signs[var.lower()] = "+"  # default positive

        # Also extract signs from denominator substitutions e.g. 1/(-15) or 1/(-20)
        denom_matches = re.findall(r"1\s*\/\s*\(\s*([+-]?)\s*(\d+(?:\.\d+)?)\s*\)", cleaned)
        for idx, (sgn, num) in enumerate(denom_matches):
            sign_char = "-" if sgn == "-" else "+"
            if "f" not in signs and idx == 0:
                signs["f"] = sign_char
            elif "u" not in signs:
                signs["u"] = sign_char

        # 4. Detect Fractions (common in 1/v, 1/Rp calculations)
        has_fractions = bool(self.FRACTION_PATTERN.search(cleaned))

        # 5. Score confidence
        conf = initial_confidence
        warning = None
        if len(steps) < 1:
            conf *= 0.5
            warning = "Very little text detected."
        elif not detected_eqs and not signs:
            conf *= 0.75
            warning = "No standard mathematical equation patterns identified."

        if conf < self.min_confidence_threshold:
            warning = f"OCR confidence ({conf:.2f}) is below reliable threshold."

        return OCRExtractionResult(
            extracted_text=" => ".join(steps) if steps else cleaned,
            extracted_steps=steps,
            confidence=round(conf, 4),
            detected_equations=detected_eqs,
            detected_signs=signs,
            has_fractions=has_fractions,
            quality_warning=warning,
            metadata={"raw_input_length": len(text), "num_steps_detected": len(steps)},
        )
