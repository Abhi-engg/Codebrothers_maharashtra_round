"""
Re:Learn — RAG Textbook & Reviewed Material Knowledge Base
Indexed NCERT Physics repository for Grade 9 & Grade 10 curriculum.
Provides grounded reference texts, principles, formulas, and cognitive conflict anchors.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("relearn.rag_kb")


@dataclass
class TextbookKnowledgeChunk:
    """A verified, curriculum-grounded textbook knowledge chunk."""
    chunk_id: str
    domain: str  # 'optics' | 'electricity' | 'kinematics' | 'gravitation'
    ncert_source: str  # e.g. "NCERT Class 10 Science, Chapter 10, Section 10.2.3"
    target_misconceptions: List[str]
    scientific_principle: str
    common_error_trap: str
    counter_example_thought_experiment: str
    correct_formula_derivation: str
    standard_formulae: List[str] = field(default_factory=list)


# Curated, authoritative NCERT Physics Knowledge Base
NCERT_PHYSICS_CHUNKS: List[TextbookKnowledgeChunk] = [
    TextbookKnowledgeChunk(
        chunk_id="NCERT-G10-OPT-01",
        domain="optics",
        ncert_source="NCERT Class 10 Science, Chapter 10 'Light: Reflection and Refraction', Section 10.2.3 (Sign Convention for Reflection by Spherical Mirrors)",
        target_misconceptions=["MISC-G10-OPT-01"],
        scientific_principle=(
            "New Cartesian Sign Convention: The pole (P) of the spherical mirror is taken as origin. "
            "All distances measured in the direction of incident light (to the right of pole) are taken as positive (+). "
            "All distances measured against the direction of incident light (to the left of pole) are taken as negative (-). "
            "For a concave mirror, the focus (F) lies in front of the reflecting surface (to the left of pole); "
            "therefore, the focal length of a concave mirror is ALWAYS negative (f < 0)."
        ),
        common_error_trap=(
            "Treating focal length as an absolute positive magnitude (e.g. f = +15 cm instead of f = -15 cm). "
            "This inverts the mirror formula geometry and causes students to compute a virtual image behind the mirror "
            "instead of a real, inverted image in front of the mirror."
        ),
        counter_example_thought_experiment=(
            "If a concave mirror had a positive focal length, rays parallel to the principal axis would appear to diverge "
            "from behind the mirror, which contradicts the fundamental property of a converging concave surface."
        ),
        correct_formula_derivation=(
            "Mirror Formula: 1/v + 1/u = 1/f => 1/v = 1/f - 1/u. "
            "For concave mirror with f = -15 cm, u = -20 cm: "
            "1/v = 1/(-15) - 1/(-20) = -1/15 + 1/20 = (-4 + 3)/60 = -1/60 => v = -60 cm. "
            "The negative sign confirms the real image is formed 60 cm in front of the mirror."
        ),
        standard_formulae=["1/v + 1/u = 1/f", "m = -v/u = h'/h", "f = R/2"],
    ),
    TextbookKnowledgeChunk(
        chunk_id="NCERT-G10-OPT-02",
        domain="optics",
        ncert_source="NCERT Class 10 Science, Chapter 10 'Light: Reflection and Refraction', Section 10.2.4 (Mirror Formula)",
        target_misconceptions=["MISC-G10-OPT-02"],
        scientific_principle=(
            "The Mirror Formula connects image distance (v), object distance (u), and focal length (f): 1/v + 1/u = 1/f. "
            "In contrast, the Thin Lens Formula has a negative sign between fractions: 1/v - 1/u = 1/f."
        ),
        common_error_trap=(
            "Transposing mirror and lens formulae (using 1/v - 1/u = 1/f for mirrors, or 1/v + 1/u = 1/f for lenses)."
        ),
        counter_example_thought_experiment=(
            "For a mirror, both object and real image are on the same side, requiring additive fractions 1/v + 1/u. "
            "For a lens, light passes through, so a real image is formed on the opposite side."
        ),
        correct_formula_derivation=(
            "For mirrors: 1/v = 1/f - 1/u. For lenses: 1/v = 1/f + 1/u."
        ),
        standard_formulae=["1/v + 1/u = 1/f (Mirrors)", "1/v - 1/u = 1/f (Lenses)"],
    ),
    TextbookKnowledgeChunk(
        chunk_id="NCERT-G10-OPT-03",
        domain="optics",
        ncert_source="NCERT Class 10 Science, Chapter 10 'Light: Reflection and Refraction', Section 10.2.5 (Magnification)",
        target_misconceptions=["MISC-G10-OPT-03"],
        scientific_principle=(
            "Magnification produced by spherical mirrors is given by m = h'/h = -v/u. "
            "A negative magnification indicates a real and inverted image. "
            "A positive magnification indicates a virtual and erect image."
        ),
        common_error_trap=(
            "Confusing the sign of magnification with the size of the image, or omitting the negative sign (using m = v/u)."
        ),
        counter_example_thought_experiment=(
            "An image with m = -2.5 is not smaller than the object; the magnitude |m| = 2.5 means it is enlarged by 2.5x, "
            "while the minus sign signifies that the image is upside down (inverted)."
        ),
        correct_formula_derivation=(
            "m = -v/u. If v = -60 cm and u = -20 cm, m = -(-60)/(-20) = -3. "
            "|m| = 3 > 1 (magnified), and m < 0 (real and inverted)."
        ),
        standard_formulae=["m = -v/u", "m = h'/h"],
    ),
    TextbookKnowledgeChunk(
        chunk_id="NCERT-G10-ELC-01",
        domain="electricity",
        ncert_source="NCERT Class 10 Science, Chapter 12 'Electricity', Section 12.6 (Resistance of a System of Resistors)",
        target_misconceptions=["MISC-G10-ELC-01"],
        scientific_principle=(
            "In a parallel combination of resistors, the reciprocal of the equivalent resistance (1/Rp) is equal to "
            "the sum of the reciprocals of the individual resistances: 1/Rp = 1/R1 + 1/R2 + ... + 1/Rn. "
            "The equivalent resistance Rp is always smaller than the smallest individual resistance in the combination."
        ),
        common_error_trap=(
            "Adding parallel resistances linearly (Rp = R1 + R2), which is only valid for series circuits."
        ),
        counter_example_thought_experiment=(
            "Connecting a second water pipe in parallel allows more water to flow, reducing total resistance to flow. "
            "Similarly, adding resistors in parallel creates more current paths, lowering total electrical resistance."
        ),
        correct_formula_derivation=(
            "1/Rp = 1/R1 + 1/R2 => Rp = (R1 * R2) / (R1 + R2). "
            "For 6 ohm and 12 ohm in parallel: 1/Rp = 1/6 + 1/12 = 3/12 = 1/4 => Rp = 4 ohms."
        ),
        standard_formulae=["1/Rp = 1/R1 + 1/R2", "Rs = R1 + R2"],
    ),
    TextbookKnowledgeChunk(
        chunk_id="NCERT-G10-ELC-02",
        domain="electricity",
        ncert_source="NCERT Class 10 Science, Chapter 12 'Electricity', Section 12.6.1 (Resistors in Series)",
        target_misconceptions=["MISC-G10-ELC-02"],
        scientific_principle=(
            "Conservation of Charge: In a series circuit, current (I) is identical through every single resistor. "
            "Current is not 'consumed' or 'attenuated' as it traverses successive components."
        ),
        common_error_trap=(
            "Believing the first resistor 'uses up' current so subsequent resistors receive less current."
        ),
        counter_example_thought_experiment=(
            "Current is the rate of flow of electrons. If fewer electrons left the second resistor than entered the first, "
            "electrons would be accumulating inside the wire, creating immense electrostatic repulsion."
        ),
        correct_formula_derivation=(
            "I_total = I1 = I2 = ... = In. V_total = V1 + V2 + ... + Vn."
        ),
        standard_formulae=["I = V / Rs", "V = IR"],
    ),
    TextbookKnowledgeChunk(
        chunk_id="NCERT-G09-MOT-01",
        domain="kinematics",
        ncert_source="NCERT Class 9 Science, Chapter 8 'Motion', Section 8.1 (Describing Motion)",
        target_misconceptions=["MISC-G09-MOT-01"],
        scientific_principle=(
            "Distance is the total actual path length traveled by an object (scalar, always >= 0). "
            "Displacement is the shortest straight-line distance from initial to final position (vector with direction). "
            "For any closed round trip where the object returns to its starting point, displacement is exactly ZERO."
        ),
        common_error_trap=(
            "Treating displacement as scalar distance, resulting in non-zero displacement or velocity for closed round trips."
        ),
        counter_example_thought_experiment=(
            "An athlete running 400 m around a circular track returns to the starting line. "
            "Distance traveled = 400 m, but displacement = 0 m. Average velocity = displacement / time = 0 m/s."
        ),
        correct_formula_derivation=(
            "Displacement s = x_final - x_initial. If x_final = x_initial, s = 0."
        ),
        standard_formulae=["Average Speed = Total Distance / Total Time", "Average Velocity = Total Displacement / Total Time"],
    ),
    TextbookKnowledgeChunk(
        chunk_id="NCERT-G09-MOT-02",
        domain="kinematics",
        ncert_source="NCERT Class 9 Science, Chapter 9 'Force and Laws of Motion', Section 9.1 (Balanced and Unbalanced Forces)",
        target_misconceptions=["MISC-G09-MOT-02"],
        scientific_principle=(
            "Newton's First Law of Motion: An object remains in a state of rest or of uniform motion in a straight line "
            "unless acted upon by an unbalanced external force. "
            "A continuous net force is NOT required to maintain uniform velocity; net force causes acceleration (F_net = ma)."
        ),
        common_error_trap=(
            "Aristotelian impetus fallacy: assuming that if an object is moving forward, there must be a forward net force."
        ),
        counter_example_thought_experiment=(
            "A spacecraft traveling through deep interstellar space coasting at constant velocity of 10 km/s "
            "has its engines turned off; it continues moving indefinitely at 10 km/s with F_net = 0."
        ),
        correct_formula_derivation=(
            "F_net = m * a. If velocity is constant, a = 0 => F_net = 0."
        ),
        standard_formulae=["F_net = m * a", "v = constant <=> a = 0 <=> F_net = 0"],
    ),
    TextbookKnowledgeChunk(
        chunk_id="NCERT-G09-GRV-01",
        domain="gravitation",
        ncert_source="NCERT Class 9 Science, Chapter 10 'Gravitation', Section 10.2 (Free Fall)",
        target_misconceptions=["MISC-G09-GRV-01"],
        scientific_principle=(
            "Acceleration due to gravity near Earth's surface is g = G*M_earth / R_earth^2. "
            "Notice that the mass of the falling object (m) cancels out completely and does NOT appear in the equation. "
            "All objects in free fall accelerate at the exact same rate regardless of their mass."
        ),
        common_error_trap=(
            "Believing heavier objects accelerate faster towards the ground in vacuum."
        ),
        counter_example_thought_experiment=(
            "Galileo's Leaning Tower experiment and Apollo 15's hammer and feather drop on the Moon: "
            "in the absence of air resistance, the heavy hammer and light feather hit the ground at the exact same instant."
        ),
        correct_formula_derivation=(
            "F = G*M*m / R^2 = m*a => a = g = G*M / R^2 (independent of m)."
        ),
        standard_formulae=["g = G*M / R^2", "v = u + gt", "s = ut + (1/2)gt^2"],
    ),
]


class TextbookRAGKnowledgeBase:
    """
    RAG retrieval engine over NCERT Physics textbooks.
    """

    def __init__(self, chunks: Optional[List[TextbookKnowledgeChunk]] = None) -> None:
        self.chunks = chunks or NCERT_PHYSICS_CHUNKS
        self._misc_index: Dict[str, TextbookKnowledgeChunk] = {}
        for c in self.chunks:
            for misc_id in c.target_misconceptions:
                self._misc_index[misc_id] = c

    def retrieve_by_misconception_id(self, misconception_id: str) -> Optional[TextbookKnowledgeChunk]:
        """
        Direct exact-match retrieval for diagnosed misconception.
        """
        return self._misc_index.get(misconception_id)

    def retrieve_context(
        self,
        misconception_id: Optional[str] = None,
        query: Optional[str] = None,
        top_k: int = 1,
    ) -> List[TextbookKnowledgeChunk]:
        """
        Retrieve best matching NCERT textbook chunks for intervention.
        """
        # 1. Primary lookup by diagnosed misconception
        if misconception_id and misconception_id in self._misc_index:
            return [self._misc_index[misconception_id]]

        # 2. Keyword/domain search over query
        if query:
            q_lower = query.lower()
            scored: List[Tuple[int, TextbookKnowledgeChunk]] = []
            for c in self.chunks:
                score = 0
                if c.domain in q_lower:
                    score += 5
                for word in c.scientific_principle.lower().split():
                    if len(word) > 4 and word in q_lower:
                        score += 1
                scored.append((score, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [chunk for score, chunk in scored[:top_k] if score > 0]

        # Default fallback: return first chunk (Optics)
        return [self.chunks[0]]
