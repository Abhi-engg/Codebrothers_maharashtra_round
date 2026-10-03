import os
import json
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any

# Assuming google-genai is used for LLM interaction as per the standard recommendations
from google import genai
from google.genai import types

# Import the whiteboard schema for structured output
from .whiteboard_schema import WhiteboardSchema, WhiteboardCommand

class InterventionResponse(BaseModel):
    explanation: str
    whiteboard_commands: List[Any]  # Will be validated by WhiteboardSchema downstream

class InterventionGenerator:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.taxonomy = self._load_taxonomy()
        self.cag_cache = self._load_cag_cache()
        # Initialize the official Gemini SDK
        self.client = genai.Client()

    def _load_taxonomy(self) -> Dict:
        taxonomy_path = self.data_dir / "taxonomy.json"
        if taxonomy_path.exists():
            with open(taxonomy_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _load_cag_cache(self) -> str:
        """Loads the entire Class 10 Physics NCERT Markdown textbook as the cache."""
        cache_path = self.data_dir / "ncert_physics_cag_cache.md"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
        return "NCERT Physics textbook data not found."

    def _build_system_prompt(self) -> str:
        """Builds the Cache-Augmented Generation (CAG) System Prompt."""
        return f"""You are an expert Class 10 Physics tutor. Your job is to correct student misconceptions with highly targeted, encouraging feedback.

You have access to the complete NCERT Class 10 Physics curriculum below. Use this as your absolute source of truth.
Do not hallucinate physics laws outside of this text.

<NCERT_TEXTBOOK_CACHE>
{self.cag_cache}
</NCERT_TEXTBOOK_CACHE>

Your tasks:
1. Provide a maximum 3-sentence targeted explanation addressing the exact misconception. Do NOT just give the correct answer. Explain WHY their mental model is flawed based on the physics principles.
2. Generate a sequence of structured whiteboard commands to visually illustrate the correct concept (e.g., ray tracing, circuit diagrams). Use normalized coordinates [0.0, 1.0].
"""

    def generate_intervention(
        self, 
        student_id: str, 
        misconception_id: str, 
        student_error: str, 
        correct_principle: str,
        question_text: str
    ) -> InterventionResponse:
        """
        Generates a pedagogical intervention and whiteboard vector drawing using CAG.
        """
        system_instruction = self._build_system_prompt()
        
        user_prompt = f"""
Student has made an error on the following question:
Question: {question_text}

Diagnosed Misconception ID: {misconception_id}
Student Error: {student_error}
Correct Principle: {correct_principle}

Generate the targeted explanation and the corresponding whiteboard visualization.
"""
        # Call the Interactions API to get structured output
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=InterventionResponse,
                temperature=0.2, # Low temperature for factual physics adherence
            ),
        )
        
        # Parse the structured JSON response
        result = json.loads(response.text)
        return InterventionResponse(**result)

if __name__ == "__main__":
    # Quick test harness
    generator = InterventionGenerator(data_dir="../../data")
    result = generator.generate_intervention(
        student_id="STU-001",
        misconception_id="MISC-LGT-01",
        student_error="Calculated focal length as +15cm for a concave mirror instead of -15cm.",
        correct_principle="By Cartesian sign convention, concave mirrors always have a negative focal length.",
        question_text="An object is placed 20 cm in front of a concave mirror of focal length 15 cm. Find the image distance."
    )
    print("Explanation:", result.explanation)
    print("Commands:", json.dumps(result.whiteboard_commands, indent=2))
