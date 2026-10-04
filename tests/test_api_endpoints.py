"""
Automated End-to-End API Integration Tests for Re:Learn Backend
Tests all 6 unified endpoints:
  1. GET  /api/quiz/questions
  2. POST /api/ingest/multimodal
  3. POST /api/diagnose
  4. POST /api/intervention
  5. POST /api/reassessment/submit
  6. GET  /api/learner/{id}/profile
"""

import base64
import io
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app, ensure_student_exists
from backend.database.db import SessionLocal, Student, Attempt, MisconceptionRecord, MisconceptionStatus


class TestRelearnAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Using TestClient as a context manager triggers the lifespan event (startup & shutdown)
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def setUp(self):
        # Setup clean test DB records and isolate tests
        self.db = SessionLocal()
        self.db.query(Attempt).filter(Attempt.student_id >= 1000).delete(synchronize_session=False)
        self.db.query(MisconceptionRecord).filter(MisconceptionRecord.student_id >= 1000).delete(synchronize_session=False)
        self.db.query(Student).filter(Student.id >= 1000).delete(synchronize_session=False)
        self.db.commit()

    def tearDown(self):
        try:
            self.db.query(Attempt).filter(Attempt.student_id >= 1000).delete(synchronize_session=False)
            self.db.query(MisconceptionRecord).filter(MisconceptionRecord.student_id >= 1000).delete(synchronize_session=False)
            self.db.query(Student).filter(Student.id >= 1000).delete(synchronize_session=False)
            self.db.commit()
        except Exception:
            self.db.rollback()
        finally:
            self.db.close()

    # ==========================================================================
    # 1. /api/quiz/questions Tests
    # ==========================================================================
    def test_quiz_questions_default(self):
        """Verify fetching quiz questions returns valid catalog with pagination."""
        response = self.client.get("/api/quiz/questions")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total", data)
        self.assertIn("questions", data)
        self.assertGreater(data["total"], 0)
        self.assertLessEqual(len(data["questions"]), 10)
        q = data["questions"][0]
        self.assertIn("question_id", q)
        self.assertIn("question_text", q)

    def test_quiz_questions_pagination(self):
        """Verify limit and offset behave accurately."""
        res_limit_2 = self.client.get("/api/quiz/questions?limit=2&offset=0")
        self.assertEqual(res_limit_2.status_code, 200)
        data_2 = res_limit_2.json()
        self.assertEqual(len(data_2["questions"]), 2)

        res_offset_1 = self.client.get("/api/quiz/questions?limit=2&offset=1")
        self.assertEqual(res_offset_1.status_code, 200)
        data_offset_1 = res_offset_1.json()
        self.assertEqual(len(data_offset_1["questions"]), 2)
        # First question of offset 1 should match second question of offset 0
        self.assertEqual(
            data_offset_1["questions"][0]["question_id"],
            data_2["questions"][1]["question_id"],
        )

    def test_quiz_questions_filtering(self):
        """Verify filtering by grade, chapter, and concept_id."""
        response = self.client.get("/api/quiz/questions?grade=10&concept_id=G10-OPT")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        for q in data["questions"]:
            self.assertEqual(q["grade"], 10)
            self.assertEqual(q["concept_id"], "G10-OPT")

    def test_quiz_questions_edge_cases(self):
        """Test boundary cases: offset larger than total, invalid query params."""
        # Offset beyond total
        res = self.client.get("/api/quiz/questions?offset=99999")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()["questions"]), 0)

        # Invalid limit (< 1)
        res_invalid = self.client.get("/api/quiz/questions?limit=0")
        self.assertEqual(res_invalid.status_code, 422)

        # Filter that matches nothing
        res_empty = self.client.get("/api/quiz/questions?concept_id=NON_EXISTENT_CONCEPT")
        self.assertEqual(res_empty.status_code, 200)
        self.assertEqual(res_empty.json()["total"], 0)
        self.assertEqual(len(res_empty.json()["questions"]), 0)

    def test_quiz_question_by_id_and_filter(self):
        """Verify fetching a specific question by question_id filter and /api/quiz/questions/{id}."""
        # 1. Fetch first question to get a known question_id
        res_all = self.client.get("/api/quiz/questions?limit=1")
        self.assertEqual(res_all.status_code, 200)
        known_id = res_all.json()["questions"][0]["question_id"]

        # 2. Test query parameter filter
        res_filter = self.client.get(f"/api/quiz/questions?question_id={known_id}")
        self.assertEqual(res_filter.status_code, 200)
        self.assertEqual(res_filter.json()["total"], 1)
        self.assertEqual(res_filter.json()["questions"][0]["question_id"], known_id)

        # 3. Test direct single question endpoint
        res_detail = self.client.get(f"/api/quiz/questions/{known_id}")
        self.assertEqual(res_detail.status_code, 200)
        self.assertEqual(res_detail.json()["question_id"], known_id)

        # 4. Test 404 for unknown question_id
        res_404 = self.client.get("/api/quiz/questions/UNKNOWN-NONEXISTENT-QID")
        self.assertEqual(res_404.status_code, 404)
        self.assertIn("not found", res_404.json()["detail"].lower())

    # ==========================================================================
    # 2. /api/ingest/multimodal Tests
    # ==========================================================================
    def test_multimodal_ingest_typed_text(self):
        """Verify typed text ingestion and sign detection."""
        payload = {
            "question_text": "An object is 20 cm in front of a concave mirror of focal length 15 cm. Find v.",
            "expected_solution": "f = -15 cm, u = -20 cm, v = -60 cm.",
            "input_mode": "typed_text",
            "final_answer": "+60 cm",
            "working_steps": "u = -20 cm, f = +15 cm\n1/v = 1/f - 1/u = 1/15 - (-1/20) = 1/15 + 1/20",
        }
        res = self.client.post("/api/ingest/multimodal", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["input_mode"], "typed_text")
        self.assertIn("f", data["detected_signs"])
        self.assertEqual(data["detected_signs"]["f"], "+")
        self.assertIn("Student Final Answer: +60 cm", data["model1_formatted_input"])

    def test_multimodal_ingest_mcq(self):
        """Verify MCQ selection mode."""
        payload = {
            "question_text": "Is the focal length of a concave mirror positive or negative?",
            "input_mode": "mcq_selection",
            "selected_option": "Positive",
        }
        res = self.client.post("/api/ingest/multimodal", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["input_mode"], "mcq_selection")
        self.assertEqual(data["final_answer"], "Positive")
        self.assertTrue(len(data["discrete_steps"]) > 0)

    def test_multimodal_ingest_image_and_diagram(self):
        """Verify handwritten image and diagram ingestion."""
        # Create a tiny 1x1 test PNG base64
        tiny_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        payload = {
            "question_text": "Analyze the ray diagram.",
            "input_mode": "ocr_handwritten",
            "handwritten_image": tiny_png_b64,
            "diagram_image": tiny_png_b64,
            "diagram_hint": "concave mirror ray diagram",
        }
        res = self.client.post("/api/ingest/multimodal", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["input_mode"], "ocr_handwritten")
        self.assertIsNotNone(data["diagram_evidence"])

    def test_multimodal_ingest_edge_cases(self):
        """Verify corrupted base64 or empty fields are handled gracefully without 500 error."""
        # Corrupted base64
        payload_bad_img = {
            "input_mode": "ocr_handwritten",
            "handwritten_image": "INVALID_NOT_BASE64!@#$%",
        }
        res = self.client.post("/api/ingest/multimodal", json=payload_bad_img)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("warnings", data)
        self.assertTrue(len(data["warnings"]) > 0)

        # Completely empty payload
        res_empty = self.client.post("/api/ingest/multimodal", json={})
        self.assertEqual(res_empty.status_code, 200)
        self.assertEqual(res_empty.json()["status"], "partial_fallback")

    def test_multimodal_ingest_multipart_upload(self):
        """Verify multipart/form-data image upload works end-to-end."""
        # 1x1 PNG bytes
        png_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
        files = {
            "handwritten_image": ("solution.png", png_bytes, "image/png"),
            "diagram_image": ("ray_diagram.png", png_bytes, "image/png"),
        }
        data = {
            "question_text": "An object is 20 cm in front of a concave mirror.",
            "expected_solution": "v = -60 cm",
            "input_mode": "ocr_handwritten",
            "final_answer": "-60 cm",
            "working_steps": "u = -20 cm, f = -15 cm",
        }
        res = self.client.post("/api/ingest/multimodal", data=data, files=files)
        self.assertEqual(res.status_code, 200)
        json_data = res.json()
        self.assertEqual(json_data["input_mode"], "ocr_handwritten")
        self.assertIn("diagram_evidence", json_data)

    def test_multimodal_ingest_payload_too_large(self):
        """Verify payloads with images > 15MB return 413 Payload Too Large."""
        oversized_str = "A" * (26 * 1024 * 1024)
        payload = {
            "input_mode": "ocr_handwritten",
            "handwritten_image": oversized_str,
        }
        res = self.client.post("/api/ingest/multimodal", json=payload)
        self.assertEqual(res.status_code, 413)

    # ==========================================================================
    # 3. /api/diagnose Tests
    # ==========================================================================
    def test_diagnose_high_confidence_trigger_intervention(self):
        """Verify Model 1 + Combiner triggers intervention on clear misconception."""
        payload = {
            "question_id": "Q-GRP-G10-MIRROR-01-01",
            "question_text": "An object is placed at a distance of 20 cm in front of a concave mirror of focal length 15 cm. Find v.",
            "expected_solution": "f = -15 cm, u = -20 cm, v = -60 cm.",
            "final_answer": "+60.0 cm",
            "working_steps": "Given u = -20 cm, f = +15 cm\n1/v = 1/f - 1/u = 1/15 - (-1/20) = 1/15 + 1/20\n1/v = 0.1167 => v = +60.0 cm behind mirror",
            "model1_override": {
                "primary_label": "MISC-G10-OPT-01",
                "confidence": 0.88,
            },
        }
        res = self.client.post("/api/diagnose", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["final_diagnosis"], "MISC-G10-OPT-01")
        self.assertEqual(data["decision_action"], "TRIGGER_INTERVENTION")
        self.assertTrue(data["whiteboard_remediation_needed"])

    def test_diagnose_abstain_on_missing_working(self):
        """Verify abstention policy: missing working steps yields ABSTAIN_UNCERTAIN."""
        payload = {
            "question_id": "Q-GRP-G10-MIRROR-01-01",
            "question_text": "Find v.",
            "final_answer": "+60.0 cm",
            "working_steps": "",  # No steps provided
            "model1_override": {
                "primary_label": "MISC-G10-OPT-01",
                "confidence": 0.90,
            },
        }
        res = self.client.post("/api/diagnose", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["decision_action"], "ABSTAIN_UNCERTAIN")
        self.assertEqual(data["final_diagnosis"], "UNCERTAIN-GUESS")
        self.assertFalse(data["whiteboard_remediation_needed"])
        self.assertIsNotNone(data["probe_question_prompt"])

    def test_diagnose_moderate_confidence_diagnostic_probe(self):
        """Verify moderate confidence (0.45 <= conf < 0.75) triggers DIAGNOSTIC_PROBE."""
        payload = {
            "question_id": "Q-GRP-G10-MIRROR-01-01",
            "question_text": "Find v.",
            "final_answer": "+60.0 cm",
            "working_steps": "Calculated value is positive 60 cm.",
            "model1_override": {
                "primary_label": "MISC-G10-OPT-01",
                "confidence": 0.60,
            },
        }
        res = self.client.post("/api/diagnose", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["decision_action"], "DIAGNOSTIC_PROBE")
        self.assertIsNotNone(data["probe_question_prompt"])

    def test_diagnose_with_database_persistence(self):
        """Verify diagnosing with student_id persists Attempt and MisconceptionRecord in DB."""
        test_student_id = 1001
        payload = {
            "student_id": test_student_id,
            "question_id": "Q-PERSIST-01",
            "final_answer": "+60.0 cm",
            "working_steps": "u = -20 cm, f = +15 cm, v = +60 cm",
            "model1_override": {
                "primary_label": "MISC-G10-OPT-01",
                "confidence": 0.85,
            },
        }
        res = self.client.post("/api/diagnose", json=payload)
        self.assertEqual(res.status_code, 200)

        # Check DB
        student = self.db.query(Student).filter(Student.id == test_student_id).first()
        self.assertIsNotNone(student)
        attempt = (
            self.db.query(Attempt)
            .filter(Attempt.student_id == test_student_id, Attempt.question_id == "Q-PERSIST-01")
            .first()
        )
        self.assertIsNotNone(attempt)
        self.assertEqual(attempt.model_1_diagnosis, "MISC-G10-OPT-01")

        rec = (
            self.db.query(MisconceptionRecord)
            .filter(
                MisconceptionRecord.student_id == test_student_id,
                MisconceptionRecord.misconception_id == "MISC-G10-OPT-01",
            )
            .first()
        )
        self.assertIsNotNone(rec)
        self.assertEqual(rec.status, MisconceptionStatus.ACTIVE)

    def test_diagnose_live_model_without_override(self):
        """Verify calling /api/diagnose without model1_override uses fitted classifier and combiner."""
        payload = {
            "question_text": "An object is placed 20 cm in front of a concave mirror of focal length 15 cm. Find v.",
            "expected_solution": "f = -15 cm, u = -20 cm, v = -60 cm.",
            "final_answer": "+60.0 cm",
            "working_steps": "Given u = -20 cm, f = +15 cm\n1/v = 1/f - 1/u = 1/15 - (-1/20)\n1/v = 1/15 + 1/20 => v = +60 cm",
        }
        res = self.client.post("/api/diagnose", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("final_diagnosis", data)
        self.assertIn("decision_action", data)
        self.assertGreater(data["final_confidence"], 0.0)

    def test_diagnose_invalid_student_id(self):
        """Verify student_id <= 0 returns 422 Unprocessable Entity."""
        res_zero = self.client.post("/api/diagnose", json={"student_id": 0, "final_answer": "ans"})
        self.assertEqual(res_zero.status_code, 422)

        res_neg = self.client.post("/api/diagnose", json={"student_id": -5, "final_answer": "ans"})
        self.assertEqual(res_neg.status_code, 422)

    def test_diagnose_auto_fill_from_question_id(self):
        """Verify providing question_id auto-fills question context if question_text is omitted."""
        payload = {
            "question_id": "Q-GRP-G10-MIRROR-01-01",
            "final_answer": "+60 cm",
            "working_steps": "f = +15 cm, u = -20 cm => v = +60 cm",
            "model1_override": {
                "primary_label": "MISC-G10-OPT-01",
                "confidence": 0.82,
            },
        }
        res = self.client.post("/api/diagnose", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["final_diagnosis"], "MISC-G10-OPT-01")

    def test_diagnose_persistence_updates_proficiency(self):
        """Verify that persisting a diagnosis attempt recalculates student overall_proficiency."""
        test_student_id = 1002
        payload = {
            "student_id": test_student_id,
            "question_id": "Q-PROF-01",
            "final_answer": "+60 cm",
            "working_steps": "u = -20 cm, f = +15 cm, v = +60 cm",
            "model1_override": {
                "primary_label": "MISC-G10-OPT-01",
                "confidence": 0.90,
            },
        }
        res = self.client.post("/api/diagnose", json=payload)
        self.assertEqual(res.status_code, 200)

        # Query student from DB
        self.db.expire_all()
        student = self.db.query(Student).filter(Student.id == test_student_id).first()
        self.assertIsNotNone(student)
        self.assertIsNotNone(student.overall_proficiency)

    # ==========================================================================
    # 4. /api/intervention Tests
    # ==========================================================================
    def test_intervention_generation_valid(self):
        """Verify intervention generator returns explanation and normalized whiteboard commands."""
        payload = {
            "student_id": "student_test_1",
            "misconception_id": "MISC-G10-OPT-01",
            "student_error": "Assumed focal length of concave mirror is positive (+15 cm).",
            "correct_principle": "By Cartesian sign convention, focal point of concave mirror is in front, so f is negative.",
            "question_text": "An object is 20 cm in front of a concave mirror of focal length 15 cm.",
        }
        res = self.client.post("/api/intervention", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["misconception_id"], "MISC-G10-OPT-01")
        self.assertIn("explanation", data)
        self.assertTrue(len(data["explanation"]) > 0)
        self.assertIn("whiteboard_commands", data)
        self.assertTrue(len(data["whiteboard_commands"]) > 0)

        # Validate that coordinates are strictly within [0.0, 1.0]
        for cmd in data["whiteboard_commands"]:
            for key in ["x", "y", "x1", "y1", "x2", "y2"]:
                if key in cmd:
                    val = float(cmd[key])
                    self.assertGreaterEqual(val, 0.0)
                    self.assertLessEqual(val, 1.0)

    def test_intervention_validation_edge_cases(self):
        """Verify empty or missing misconception_id raises 422 Unprocessable Entity."""
        res = self.client.post("/api/intervention", json={"misconception_id": ""})
        self.assertEqual(res.status_code, 422)

        res_missing = self.client.post("/api/intervention", json={})
        self.assertEqual(res_missing.status_code, 422)

    # ==========================================================================
    # 5. /api/reassessment/submit Tests
    # ==========================================================================
    def test_reassessment_likely_resolved(self):
        """Verify correct transfer attempt transitions status to 'Likely Resolved' / RESOLVED."""
        test_student_id = 2001
        payload = {
            "student_id": test_student_id,
            "misconception_id": "MISC-G10-OPT-01",
            "transfer_question_id": "Q-TRANSFER-MIRROR-01",
            "student_answer": "-60.0 cm",
            "student_working": "Given: u = -20 cm, f = -15 cm. Mirror formula: 1/v = 1/f - 1/u => v = -60.0 cm",
            "diagnosis_result": {
                "is_correct_answer": True,
                "is_valid_reasoning": True,
                "diagnosed_misconception": "CORRECT",
                "confidence": 0.98,
            },
        }
        res = self.client.post("/api/reassessment/submit", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["mastery_state"], "Likely Resolved")
        self.assertEqual(data["db_status"], "resolved")

        # Verify DB updated
        rec = (
            self.db.query(MisconceptionRecord)
            .filter(
                MisconceptionRecord.student_id == test_student_id,
                MisconceptionRecord.misconception_id == "MISC-G10-OPT-01",
            )
            .first()
        )
        self.assertIsNotNone(rec)
        self.assertEqual(rec.status, MisconceptionStatus.RESOLVED)

    def test_reassessment_still_present(self):
        """Verify repeated misconception transitions to 'Still Present' / ACTIVE and increments count."""
        test_student_id = 2002
        payload = {
            "student_id": test_student_id,
            "misconception_id": "MISC-G10-OPT-01",
            "transfer_question_id": "Q-TRANSFER-MIRROR-02",
            "student_answer": "+60.0 cm",
            "student_working": "f is positive so v = +60 cm",
            "diagnosis_result": {
                "is_correct_answer": False,
                "is_valid_reasoning": False,
                "diagnosed_misconception": "MISC-G10-OPT-01",
                "confidence": 0.90,
            },
        }
        res = self.client.post("/api/reassessment/submit", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["mastery_state"], "Still Present")
        self.assertEqual(data["db_status"], "active")

    def test_reassessment_uncertain_lucky_guess(self):
        """Verify lucky guess without valid reasoning transitions to 'Uncertain' / IMPROVING."""
        test_student_id = 2003
        payload = {
            "student_id": test_student_id,
            "misconception_id": "MISC-G10-OPT-01",
            "transfer_question_id": "Q-TRANSFER-MIRROR-03",
            "student_answer": "-60.0 cm",
            "student_working": "I guessed -60 cm",
            "diagnosis_result": {
                "is_correct_answer": True,
                "is_valid_reasoning": False,  # Missing clear valid steps
                "diagnosed_misconception": "UNCERTAIN-GUESS",
                "confidence": 0.40,
            },
        }
        res = self.client.post("/api/reassessment/submit", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["mastery_state"], "Uncertain")
        self.assertEqual(data["db_status"], "improving")

    def test_reassessment_edge_cases(self):
        """Verify invalid student_id raises 422."""
        res = self.client.post(
            "/api/reassessment/submit",
            json={
                "student_id": 0,
                "misconception_id": "MISC-G10-OPT-01",
                "transfer_question_id": "Q-01",
                "student_answer": "ans",
            },
        )
        self.assertEqual(res.status_code, 422)

    # ==========================================================================
    # 6. /api/learner/{id}/profile Tests
    # ==========================================================================
    def test_learner_profile_success(self):
        """Verify retrieving learner profile with active/improving/resolved records and attempts."""
        test_student_id = 3001
        ensure_student_exists(self.db, test_student_id, username="alice_physics")

        # Insert some records
        r1 = MisconceptionRecord(
            student_id=test_student_id,
            misconception_id="MISC-G10-OPT-01",
            status=MisconceptionStatus.RESOLVED,
        )
        r2 = MisconceptionRecord(
            student_id=test_student_id,
            misconception_id="MISC-G10-ELE-01",
            status=MisconceptionStatus.ACTIVE,
        )
        att = Attempt(
            student_id=test_student_id,
            question_id="Q-ALICE-01",
            student_answer="-60 cm",
            model_1_diagnosis="CORRECT",
            confidence=0.95,
        )
        self.db.add_all([r1, r2, att])
        self.db.commit()

        res = self.client.get(f"/api/learner/{test_student_id}/profile")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["student_id"], test_student_id)
        self.assertEqual(data["username"], "alice_physics")
        self.assertEqual(len(data["resolved_misconceptions"]), 1)
        self.assertEqual(len(data["active_misconceptions"]), 1)
        self.assertGreaterEqual(data["total_attempts"], 1)

    def test_learner_profile_not_found(self):
        """Verify querying non-existent learner ID returns 404."""
        res = self.client.get("/api/learner/999999/profile")
        self.assertEqual(res.status_code, 404)
        self.assertIn("not found", res.json()["detail"].lower())

    def test_learner_profile_invalid_id(self):
        """Verify invalid student ID (<= 0) returns 422."""
        res = self.client.get("/api/learner/0/profile")
        self.assertEqual(res.status_code, 422)

    def test_list_learners_roster(self):
        """Verify GET /api/learners returns paginated list of students."""
        res = self.client.get("/api/learners?limit=10&offset=0")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total", data)
        self.assertIn("learners", data)
        self.assertGreaterEqual(data["total"], 1)
        l = data["learners"][0]
        self.assertIn("student_id", l)
        self.assertIn("username", l)
        self.assertIn("overall_proficiency", l)

    # ==========================================================================
    # Health Check
    # ==========================================================================
    def test_health_check(self):
        """Verify /api/health endpoint returns 200 OK."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "healthy")


if __name__ == "__main__":
    unittest.main()
