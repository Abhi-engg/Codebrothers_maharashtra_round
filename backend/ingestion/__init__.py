"""
Re:Learn Ingestion Package
"""

from backend.ingestion.ocr_parser import (
    HandwrittenOCRParser,
    OCRExtractionResult,
)
from backend.ingestion.multimodal_normalizer import (
    MultimodalNormalizer,
    NormalizedSubmission,
)

__all__ = [
    "HandwrittenOCRParser",
    "OCRExtractionResult",
    "MultimodalNormalizer",
    "NormalizedSubmission",
]
