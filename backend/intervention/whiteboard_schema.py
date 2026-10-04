"""
Re:Learn Structured Whiteboard Tool Contract
Defines the safe, typed JSON API for visual pedagogical remediation.

Security & Architecture Invariants:
  1. The LLM / backend DOES NOT emit executable JavaScript or Python code.
  2. All coordinates are strictly normalized to [0.0, 1.0].
  3. The backend validates and sanitizes all bounds before transmitting to the client.
  4. The frontend canvas maps [0.0, 1.0] -> [W, H] responsively.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("relearn.whiteboard_schema")

VALID_HEX_COLOR = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
VALID_TOOL_NAMES = {
    "draw_axes",
    "draw_line",
    "draw_arrow",
    "draw_shape",
    "draw_text",
    "highlight_region",
    "clear_canvas",
}
VALID_SHAPES = {
    "concave_mirror",
    "convex_mirror",
    "convex_lens",
    "concave_lens",
    "resistor",
    "battery",
    "bulb",
    "switch",
    "rectangle",
    "circle",
}


def clamp_coord(val: float) -> float:
    """Clamps coordinate strictly to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(val)))


def sanitize_color(color: Optional[str], default: str = "#2B6CB0") -> str:
    """Ensures color is a safe hex string, preventing CSS/JS injection."""
    if not color:
        return default
    color_clean = color.strip()
    if VALID_HEX_COLOR.match(color_clean):
        return color_clean
    return default


@dataclass
class WhiteboardCommand:
    """Single discrete drawing operation on the normalized canvas."""
    tool: str
    params: Dict[str, Any]
    step_description: str = ""
    annotation: Optional[str] = None

    def validate(self) -> WhiteboardCommand:
        """Validates parameters, clamps coordinates, and sanitizes strings."""
        if self.tool not in VALID_TOOL_NAMES:
            raise ValueError(f"Invalid whiteboard tool: {self.tool}. Allowed: {VALID_TOOL_NAMES}")

        p = self.params

        if self.tool == "draw_axes":
            origin = p.get("origin", [0.5, 0.5])
            p["origin"] = [clamp_coord(origin[0]), clamp_coord(origin[1])]
            p["scale"] = p.get("scale", "cartesian")
            p["color"] = sanitize_color(p.get("color"), "#718096")

        elif self.tool in ("draw_line", "draw_arrow"):
            start = p.get("start", [0.0, 0.0])
            end = p.get("end", [1.0, 1.0])
            p["start"] = [clamp_coord(start[0]), clamp_coord(start[1])]
            p["end"] = [clamp_coord(end[0]), clamp_coord(end[1])]
            p["color"] = sanitize_color(p.get("color"), "#E53E3E" if self.tool == "draw_arrow" else "#2D3748")
            p["style"] = p.get("style", "solid")  # 'solid', 'dashed', 'dotted'
            p["width"] = max(1.0, min(10.0, float(p.get("width", 2.0))))
            if self.tool == "draw_arrow" and "label" in p:
                p["label"] = str(p["label"])[:60]

        elif self.tool == "draw_shape":
            shape_type = str(p.get("shape_type", "rectangle")).lower()
            if shape_type not in VALID_SHAPES:
                shape_type = "rectangle"
            p["shape_type"] = shape_type
            p["x"] = clamp_coord(p.get("x", 0.5))
            p["y"] = clamp_coord(p.get("y", 0.5))
            p["width"] = clamp_coord(p.get("width", 0.1))
            p["height"] = clamp_coord(p.get("height", 0.1))
            p["color"] = sanitize_color(p.get("color"), "#3182CE")

        elif self.tool == "draw_text":
            p["x"] = clamp_coord(p.get("x", 0.5))
            p["y"] = clamp_coord(p.get("y", 0.5))
            p["text"] = str(p.get("text", ""))[:120]  # prevent buffer bloat
            p["size"] = max(10, min(36, int(p.get("size", 14))))
            p["color"] = sanitize_color(p.get("color"), "#1A202C")
            p["align"] = p.get("align", "left")

        elif self.tool == "highlight_region":
            bounds = p.get("bounds", [0.0, 0.0, 1.0, 1.0])
            p["bounds"] = [clamp_coord(b) for b in bounds[:4]]
            p["color"] = sanitize_color(p.get("color"), "#ECC94B")
            p["pulse"] = bool(p.get("pulse", True))

        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredWhiteboardPayload:
    """Complete sequence of whiteboard steps sent to the frontend vector renderer."""
    misconception_id: str
    concept_title: str
    pedagogical_explanation: str
    commands: List[WhiteboardCommand]
    total_steps: int = field(init=False)

    def __post_init__(self) -> None:
        self.commands = [cmd.validate() for cmd in self.commands]
        self.total_steps = len(self.commands)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "misconception_id": self.misconception_id,
            "concept_title": self.concept_title,
            "pedagogical_explanation": self.pedagogical_explanation,
            "total_steps": self.total_steps,
            "commands": [c.to_dict() for c in self.commands],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ==============================================================================
# BUILT-IN PEDAGOGICAL WHITEBOARD PRESETS
# ==============================================================================

def get_concave_mirror_preset() -> StructuredWhiteboardPayload:
    """Preset remediating Cartesian Sign Inversion in Concave Mirrors (MISC-G10-OPT-01)."""
    commands = [
        WhiteboardCommand(
            tool="draw_axes",
            params={"origin": [0.50, 0.50], "scale": "cartesian", "color": "#718096"},
            step_description="Establish Principal Axis & Cartesian Reference Frame",
            annotation="The pole P of the mirror is placed at origin (0, 0).",
        ),
        WhiteboardCommand(
            tool="draw_shape",
            params={"shape_type": "concave_mirror", "x": 0.50, "y": 0.20, "height": 0.60, "color": "#2B6CB0"},
            step_description="Draw Concave (Converging) Mirror",
            annotation="Reflecting surface curves inward toward the left.",
        ),
        WhiteboardCommand(
            tool="draw_text",
            params={"x": 0.52, "y": 0.53, "text": "Pole (P)", "size": 13, "color": "#2D3748"},
            step_description="Mark Pole",
        ),
        WhiteboardCommand(
            tool="draw_arrow",
            params={"start": [0.50, 0.50], "end": [0.25, 0.50], "label": "f = -15 cm", "color": "#E53E3E"},
            step_description="Highlight Focal Length (Negative Sign Convention)",
            annotation="Distances measured AGAINST incident light (to the left) are NEGATIVE!",
        ),
        WhiteboardCommand(
            tool="draw_text",
            params={"x": 0.25, "y": 0.47, "text": "Focus (F)", "size": 14, "color": "#E53E3E"},
            step_description="Mark Focus Point",
        ),
        WhiteboardCommand(
            tool="draw_arrow",
            params={"start": [0.15, 0.50], "end": [0.15, 0.32], "label": "Object (h=4cm)", "color": "#38A169"},
            step_description="Place Object Beyond Focus",
            annotation="Object distance u = -20 cm (measured to the left).",
        ),
        WhiteboardCommand(
            tool="draw_line",
            params={"start": [0.15, 0.32], "end": [0.50, 0.32], "style": "solid", "color": "#D69E2E"},
            step_description="Ray 1: Parallel to Principal Axis",
        ),
        WhiteboardCommand(
            tool="draw_line",
            params={"start": [0.50, 0.32], "end": [0.05, 0.68], "style": "solid", "color": "#D69E2E"},
            step_description="Ray 1 Reflects Through Focus F",
        ),
        WhiteboardCommand(
            tool="highlight_region",
            params={"bounds": [0.20, 0.40, 0.30, 0.60], "color": "#ECC94B", "pulse": True},
            step_description="Identify Sign Trap Resolution",
            annotation="Both f and u MUST be substituted with NEGATIVE signs in 1/v = 1/f - 1/u!",
        ),
    ]

    return StructuredWhiteboardPayload(
        misconception_id="MISC-G10-OPT-01",
        concept_title="Cartesian Sign Convention for Spherical Mirrors",
        pedagogical_explanation=(
            "In optical ray tracing, the pole P serves as the origin (0, 0). "
            "Light travels from left to right. Any distance measured from P against the incident light "
            "(to the left) has a NEGATIVE sign. Because a concave mirror's focus lies in front of the mirror, "
            "its focal length f is ALWAYS negative (f = -15 cm)."
        ),
        commands=commands,
    )


def get_parallel_circuit_preset() -> StructuredWhiteboardPayload:
    """Preset remediating Parallel Resistance & Current Split (MISC-G10-ELE-02 & 03)."""
    commands = [
        WhiteboardCommand(
            tool="draw_shape",
            params={"shape_type": "battery", "x": 0.15, "y": 0.50, "width": 0.08, "height": 0.15, "color": "#C53030"},
            step_description="Place Voltage Source (12V Battery)",
        ),
        WhiteboardCommand(
            tool="draw_line",
            params={"start": [0.15, 0.42], "end": [0.35, 0.42], "style": "solid", "color": "#2D3748"},
            step_description="Lead Wire to Node A",
        ),
        WhiteboardCommand(
            tool="draw_arrow",
            params={"start": [0.20, 0.38], "end": [0.30, 0.38], "label": "Total I = 3A", "color": "#3182CE"},
            step_description="Main Branch Current",
        ),
        WhiteboardCommand(
            tool="draw_shape",
            params={"shape_type": "resistor", "x": 0.55, "y": 0.30, "width": 0.18, "height": 0.08, "color": "#D69E2E"},
            step_description="Top Branch Resistor R1 = 6 ohms",
        ),
        WhiteboardCommand(
            tool="draw_shape",
            params={"shape_type": "resistor", "x": 0.55, "y": 0.60, "width": 0.18, "height": 0.08, "color": "#38A169"},
            step_description="Bottom Branch Resistor R2 = 12 ohms",
        ),
        WhiteboardCommand(
            tool="draw_text",
            params={"x": 0.55, "y": 0.22, "text": "I1 = 12V / 6Ω = 2A", "size": 13, "color": "#D69E2E"},
            step_description="Calculate Top Branch Current",
        ),
        WhiteboardCommand(
            tool="draw_text",
            params={"x": 0.55, "y": 0.72, "text": "I2 = 12V / 12Ω = 1A", "size": 13, "color": "#38A169"},
            step_description="Calculate Bottom Branch Current",
        ),
        WhiteboardCommand(
            tool="highlight_region",
            params={"bounds": [0.45, 0.20, 0.75, 0.75], "color": "#ECC94B", "pulse": True},
            step_description="Reciprocal Summing Rule",
            annotation="1/Rp = 1/6 + 1/12 = 3/12 => Invert to get Rp = 12/3 = 4 ohms! Do not omit inversion!",
        ),
    ]

    return StructuredWhiteboardPayload(
        misconception_id="MISC-G10-ELE-03",
        concept_title="Parallel Circuits: Current Splitting and Reciprocal Inversion",
        pedagogical_explanation=(
            "In parallel connections, voltage across both branches is identical (V = 12V). "
            "Current splits inversely proportional to resistance: the lower resistor draws MORE current (2A vs 1A). "
            "To find equivalent resistance, sum reciprocals (1/Rp = 1/6 + 1/12 = 3/12 = 1/4) "
            "and ALWAYS perform the final inversion: Rp = 4 ohms."
        ),
        commands=commands,
    )
