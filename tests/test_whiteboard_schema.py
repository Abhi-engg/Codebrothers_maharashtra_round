"""
Unit tests for Re:Learn Whiteboard Schema and Tool Contract
Tests coordinate clamping, security sanitization, and command serialization.
"""

import json
import unittest
from backend.intervention.whiteboard_schema import (
    WhiteboardCommand,
    StructuredWhiteboardPayload,
    get_concave_mirror_preset,
    get_parallel_circuit_preset,
    clamp_coord,
    sanitize_color,
)


class TestWhiteboardSchema(unittest.TestCase):
    def test_coordinate_clamping(self) -> None:
        self.assertEqual(clamp_coord(-0.5), 0.0)
        self.assertEqual(clamp_coord(1.5), 1.0)
        self.assertEqual(clamp_coord(0.42), 0.42)

    def test_color_sanitization_and_security(self) -> None:
        # Valid hex colors
        self.assertEqual(sanitize_color("#3182CE"), "#3182CE")
        self.assertEqual(sanitize_color("#FFF"), "#FFF")

        # Invalid / XSS attempt string should fall back safely to default
        malicious = "red; alert('xss');"
        self.assertEqual(sanitize_color(malicious, default="#2B6CB0"), "#2B6CB0")

    def test_whiteboard_command_validation(self) -> None:
        cmd = WhiteboardCommand(
            tool="draw_arrow",
            params={
                "start": [-0.2, 0.5],
                "end": [1.4, 0.5],
                "color": "#E53E3E",
                "label": "f = -15 cm",
            },
            step_description="Draw focal length arrow",
        )
        validated = cmd.validate()
        self.assertEqual(validated.params["start"], [0.0, 0.5])
        self.assertEqual(validated.params["end"], [1.0, 0.5])
        self.assertEqual(validated.params["color"], "#E53E3E")

    def test_invalid_tool_rejection(self) -> None:
        cmd = WhiteboardCommand(tool="execute_code", params={})
        with self.assertRaises(ValueError):
            cmd.validate()

    def test_payload_presets_and_json_export(self) -> None:
        optics = get_concave_mirror_preset()
        self.assertEqual(optics.misconception_id, "MISC-G10-OPT-01")
        self.assertGreater(optics.total_steps, 5)

        optics_json = optics.to_json()
        parsed = json.loads(optics_json)
        self.assertEqual(parsed["misconception_id"], "MISC-G10-OPT-01")
        self.assertIn("commands", parsed)

        circuits = get_parallel_circuit_preset()
        self.assertEqual(circuits.misconception_id, "MISC-G10-ELE-03")
        self.assertGreater(circuits.total_steps, 4)


if __name__ == "__main__":
    unittest.main()
