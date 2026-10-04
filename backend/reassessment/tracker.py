from typing import Dict, Any
from backend.database.db import SessionLocal, MisconceptionRecord, MisconceptionStatus, Attempt
import datetime

class ReassessmentStateMachine:
    """
    Evaluates student response on transfer questions to update mastery states.
    - Likely Resolved: Student succeeds on transfer question with valid physical reasoning.
    - Still Present: Student repeats the exact same misconception or falls into a connected trap.
    - Uncertain: Student gets the correct numerical answer through an unclear method or incomplete steps.
    """
    
    @staticmethod
    def evaluate_transfer_attempt(
        student_id: int,
        misconception_id: str,
        transfer_question_id: str,
        student_answer: str,
        student_working: str,
        diagnosis_result: Dict[str, Any]
    ) -> str:
        """
        Evaluates a transfer question attempt and updates the misconception status.
        
        Args:
            student_id: The ID of the student
            misconception_id: The ID of the targeted misconception being reassessed
            transfer_question_id: The ID of the transfer question
            student_answer: The student's final answer
            student_working: The student's working/steps
            diagnosis_result: The result from the diagnosis model (Model 1 & Combiner), expected to have:
                              - is_correct_answer: bool (matches numerical answer)
                              - is_valid_reasoning: bool (has clear/correct steps)
                              - diagnosed_misconception: str (e.g., MISC-LGT-01)
                              - confidence: float
        
        Returns:
            The determined mastery state string ('Likely Resolved', 'Still Present', 'Uncertain')
        """
        is_correct_answer = diagnosis_result.get("is_correct_answer", False)
        is_valid_reasoning = diagnosis_result.get("is_valid_reasoning", False)
        diagnosed_misc = diagnosis_result.get("diagnosed_misconception", None)
        confidence = diagnosis_result.get("confidence", 0.0)

        # State evaluation logic
        if is_correct_answer and is_valid_reasoning:
            new_state = "Likely Resolved"
            new_db_status = MisconceptionStatus.RESOLVED
        elif diagnosed_misc and diagnosed_misc.startswith("MISC-"):
            # If they hit the exact same misconception, or a new related one
            new_state = "Still Present"
            new_db_status = MisconceptionStatus.ACTIVE
        else:
            # e.g., correct answer but invalid/missing reasoning, or uncertain model prediction
            # "One lucky guess never grants complete mastery."
            new_state = "Uncertain"
            new_db_status = MisconceptionStatus.IMPROVING

        # Update Database
        db = SessionLocal()
        try:
            # Log the attempt
            attempt = Attempt(
                student_id=student_id,
                timestamp=datetime.datetime.utcnow(),
                question_id=transfer_question_id,
                student_answer=student_answer,
                extracted_steps=student_working,
                model_1_diagnosis=diagnosed_misc,
                confidence=confidence,
                is_reassessment=1,
                transfer_question_id=transfer_question_id
            )
            db.add(attempt)

            # Update MisconceptionRecord
            record = db.query(MisconceptionRecord).filter(
                MisconceptionRecord.student_id == student_id,
                MisconceptionRecord.misconception_id == misconception_id
            ).first()

            if record:
                record.status = new_db_status
                if new_db_status == MisconceptionStatus.ACTIVE:
                    # They failed it again, increment occurrences
                    record.occurrence_count += 1
            else:
                # If for some reason the record doesn't exist, create it
                record = MisconceptionRecord(
                    student_id=student_id,
                    misconception_id=misconception_id,
                    occurrence_count=1,
                    status=new_db_status
                )
                db.add(record)
                
            db.commit()
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

        return new_state
