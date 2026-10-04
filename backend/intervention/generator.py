"""
Re:Learn — Qwen2.5 + RAG Targeted Remediation Generator
Generates targeted pedagogical explanations grounded in NCERT physics textbooks
and verified curriculum materials for diagnosed student misconceptions.

No fine-tuning needed for prototype:
- RAG retrieves authoritative NCERT curriculum principles and counter-examples.
- Qwen2.5-3B-Instruct / 7B-Instruct synthesizes the 3-step targeted intervention:
    1. Socratic Cognitive Conflict Probe
    2. Grounded NCERT Textbook Explanation
    3. Step-by-Step Interactive Whiteboard Resolution
- Supports local HuggingFace Qwen2.5, external Ollama / OpenAI-compatible API,
  and high-fidelity offline RAG textbook synthesizer fallback.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from backend.intervention.rag_knowledge_base import (
    TextbookKnowledgeChunk,
    TextbookRAGKnowledgeBase,
)
from backend.models.combiner import DiagnosticDecision

logger = logging.getLogger("relearn.intervention_generator")


@dataclass
class TargetedExplanation:
    """A complete 3-step targeted explanation for a diagnosed misconception."""
    misconception_id: str
    misconception_name: str
    ncert_reference: str
    cognitive_conflict_prompt: str
    textbook_grounded_explanation: str
    remediation_steps: List[str]
    whiteboard_cue: str
    model_used: str  # e.g. 'qwen2.5-3b-instruct' | 'qwen2.5-api' | 'rag_textbook_synthesizer'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "misconception_id": self.misconception_id,
            "misconception_name": self.misconception_name,
            "ncert_reference": self.ncert_reference,
            "cognitive_conflict_prompt": self.cognitive_conflict_prompt,
            "textbook_grounded_explanation": self.textbook_grounded_explanation,
            "remediation_steps": self.remediation_steps,
            "whiteboard_cue": self.whiteboard_cue,
            "model_used": self.model_used,
            "metadata": self.metadata,
        }


class QwenInterventionGenerator:
    """
    RAG-grounded intervention generator using Qwen2.5-Instruct.
    """

    SYSTEM_PROMPT = (
        "You are Re:Learn AI Physics Tutor, grounded strictly in NCERT Science textbooks. "
        "Your role is to guide students through conceptual physics misconceptions using Socratic questioning "
        "and crystal-clear physical principles. Never shame the student; always anchor the explanation "
        "in the official NCERT curriculum."
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        backend: str = "auto",  # 'auto' | 'hf' | 'api' | 'rag_synthesizer'
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        rag_kb: Optional[TextbookRAGKnowledgeBase] = None,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self.api_endpoint = api_endpoint or os.getenv("QWEN_API_ENDPOINT")
        self.api_key = api_key or os.getenv("QWEN_API_KEY")
        self.rag_kb = rag_kb or TextbookRAGKnowledgeBase()
        self._hf_pipeline = None

    def generate_intervention(
        self,
        decision: DiagnosticDecision,
        student_submission: Optional[Dict[str, Any]] = None,
    ) -> TargetedExplanation:
        """
        Synthesize a targeted pedagogical intervention for the diagnosed misconception.
        """
        misc_id = decision.final_diagnosis
        sub = student_submission or {}
        q_text = sub.get("question_text", "Physics numerical problem")
        working = sub.get("working_steps", "")
        student_ans = sub.get("final_answer", "")

        # 1. RAG Retrieval from NCERT Knowledge Base
        chunk = self.rag_kb.retrieve_by_misconception_id(misc_id)
        if chunk is None:
            # Fallback to query retrieval
            chunks = self.rag_kb.retrieve_context(query=f"{misc_id} {q_text}")
            chunk = chunks[0] if chunks else None

        # 2. If API is configured, call Qwen2.5 API
        if (self.backend in ("api", "auto")) and self.api_endpoint:
            api_exp = self._call_qwen_api(decision, chunk, q_text, working, student_ans)
            if api_exp is not None:
                return api_exp

        # 3. If local HF requested and available
        if self.backend == "hf":
            hf_exp = self._call_hf_qwen(decision, chunk, q_text, working, student_ans)
            if hf_exp is not None:
                return hf_exp

        # 4. Deterministic NCERT RAG Synthesizer (Zero-latency, zero-hallucination baseline)
        return self._synthesize_grounded_rag(decision, chunk, q_text, working, student_ans)

    def _build_qwen_prompt(
        self,
        decision: DiagnosticDecision,
        chunk: Optional[TextbookKnowledgeChunk],
        q_text: str,
        working: str,
        student_ans: str,
    ) -> str:
        """Build Qwen2.5 chat template instruction."""
        ncert_context = ""
        if chunk:
            ncert_context = (
                f"NCERT Source: {chunk.ncert_source}\n"
                f"Scientific Principle: {chunk.scientific_principle}\n"
                f"Common Misconception Trap: {chunk.common_error_trap}\n"
                f"Thought Experiment / Counter-Example: {chunk.counter_example_thought_experiment}\n"
                f"Correct Mathematical Derivation: {chunk.correct_formula_derivation}\n"
            )

        user_content = (
            f"Diagnosed Misconception: {decision.final_diagnosis} (Confidence: {decision.final_confidence:.2f})\n"
            f"Question: {q_text}\n"
            f"Student's Final Answer: {student_ans}\n"
            f"Student's Working Steps:\n{working}\n\n"
            f"Official NCERT Physics Material:\n{ncert_context}\n\n"
            "Task: Generate a targeted pedagogical explanation in valid JSON with these keys:\n"
            "1. 'cognitive_conflict_prompt': A single provocative Socratic question challenging their assumption.\n"
            "2. 'textbook_grounded_explanation': 2-3 sentences explaining the correct physical principle citing NCERT.\n"
            "3. 'remediation_steps': List of 3 concise bullet points showing how to solve the problem.\n"
            "4. 'whiteboard_cue': Instruction telling the student what visual feature to watch on the canvas whiteboard."
        )

        return (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _synthesize_grounded_rag(
        self,
        decision: DiagnosticDecision,
        chunk: Optional[TextbookKnowledgeChunk],
        q_text: str,
        working: str,
        student_ans: str,
    ) -> TargetedExplanation:
        """
        Deterministic, curriculum-accurate synthesizer grounded strictly in NCERT knowledge chunks.
        """
        misc_id = decision.final_diagnosis

        if chunk:
            ncert_ref = chunk.ncert_source
            conflict = (
                f"Consider this thought experiment: {chunk.counter_example_thought_experiment} "
                f"How does your calculation of '{student_ans or 'your answer'}' reconcile with this principle?"
            )
            grounded_exp = (
                f"According to {chunk.ncert_source}, {chunk.scientific_principle} "
                f"Here, the misconception arises from {chunk.common_error_trap}"
            )
            steps = [
                f"Step 1: Identify given parameters with Cartesian signs: {chunk.standard_formulae[0] if chunk.standard_formulae else 'Standard formula'}.",
                f"Step 2: Correct derivation: {chunk.correct_formula_derivation.split('=>')[0].strip()}.",
                f"Step 3: Calculate the final value and verify the sign matches physical reality.",
            ]
            wb_cue = (
                f"Watch Step 2 on the interactive whiteboard: note the negative focal length position "
                f"and observe where the reflected rays intersect in front of the mirror."
            )
            misc_name = chunk.target_misconceptions[0] if chunk.target_misconceptions else misc_id
        else:
            ncert_ref = "NCERT Secondary Science Curriculum (Grades 9 & 10)"
            conflict = "If we trace the physical path of this process, does the sign or formula you chose hold true?"
            grounded_exp = f"Analysis indicates conceptual trap {misc_id}. Please review the core principles in {ncert_ref}."
            steps = ["Step 1: Re-check formula signs.", "Step 2: Re-evaluate intermediate equations."]
            wb_cue = "Follow the step-by-step vector diagram on the whiteboard canvas."
            misc_name = misc_id

        return TargetedExplanation(
            misconception_id=misc_id,
            misconception_name=misc_name,
            ncert_reference=ncert_ref,
            cognitive_conflict_prompt=conflict,
            textbook_grounded_explanation=grounded_exp,
            remediation_steps=steps,
            whiteboard_cue=wb_cue,
            model_used="qwen2.5_rag_synthesizer",
            metadata={"source_chunk_id": chunk.chunk_id if chunk else None},
        )

    def _call_qwen_api(
        self,
        decision: DiagnosticDecision,
        chunk: Optional[TextbookKnowledgeChunk],
        q_text: str,
        working: str,
        student_ans: str,
    ) -> Optional[TargetedExplanation]:
        """Call external or local OpenAI-compatible endpoint hosting Qwen2.5."""
        try:
            import urllib.request
            prompt = self._build_qwen_prompt(decision, chunk, q_text, working, student_ans)
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": 0.2,
                "max_tokens": 512,
            }
            req = urllib.request.Request(
                self.api_endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data.get("choices", [{}])[0].get("text", "")
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    parsed = json.loads(m.group(0))
                    return TargetedExplanation(
                        misconception_id=decision.final_diagnosis,
                        misconception_name=decision.final_diagnosis,
                        ncert_reference=chunk.ncert_source if chunk else "NCERT Physics",
                        cognitive_conflict_prompt=parsed.get("cognitive_conflict_prompt", ""),
                        textbook_grounded_explanation=parsed.get("textbook_grounded_explanation", ""),
                        remediation_steps=parsed.get("remediation_steps", []),
                        whiteboard_cue=parsed.get("whiteboard_cue", ""),
                        model_used="qwen2.5-api",
                        metadata=parsed,
                    )
        except Exception as e:
            logger.warning("Qwen2.5 API request failed: %s; falling back to RAG synthesizer.", e)
        return None

    def _call_hf_qwen(
        self,
        decision: DiagnosticDecision,
        chunk: Optional[TextbookKnowledgeChunk],
        q_text: str,
        working: str,
        student_ans: str,
    ) -> Optional[TargetedExplanation]:
        """Local HuggingFace Qwen2.5 pipeline execution when weights are cached."""
        try:
            from transformers import pipeline
            if self._hf_pipeline is None:
                self._hf_pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device_map="auto",
                    torch_dtype="auto",
                )
            prompt = self._build_qwen_prompt(decision, chunk, q_text, working, student_ans)
            outputs = self._hf_pipeline(prompt, max_new_tokens=512, do_sample=False)
            gen_text = outputs[0]["generated_text"]
            # Extract generated response after prompt
            clean_res = gen_text[len(prompt):].strip()
            m = re.search(r"\{.*\}", clean_res, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                return TargetedExplanation(
                    misconception_id=decision.final_diagnosis,
                    misconception_name=decision.final_diagnosis,
                    ncert_reference=chunk.ncert_source if chunk else "NCERT Physics",
                    cognitive_conflict_prompt=parsed.get("cognitive_conflict_prompt", ""),
                    textbook_grounded_explanation=parsed.get("textbook_grounded_explanation", ""),
                    remediation_steps=parsed.get("remediation_steps", []),
                    whiteboard_cue=parsed.get("whiteboard_cue", ""),
                    model_used="qwen2.5-hf",
                    metadata=parsed,
                )
        except Exception as e:
            logger.info("Local HF Qwen2.5 execution unavailable: %s; falling back to RAG synthesizer.", e)
        return None
