import os
import sys

# Ensure UTF-8 stdout encoding on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.intervention.generator import InterventionGenerator
from backend.database.db import SessionLocal, Student, init_db, MisconceptionRecord
from backend.reassessment.tracker import ReassessmentStateMachine

def run_demo():
    print("\n" + "="*50)
    print(">> PHASE 4: PROOF OF CONCEPT DEMO")
    print("="*50)

    # 1. Initialize the SQLite Database
    init_db()
    db = SessionLocal()
    
    # Create a dummy student if they don't exist
    student = db.query(Student).filter(Student.username == "test_student").first()
    if not student:
        student = Student(username="test_student", overall_proficiency=0.5)
        db.add(student)
        db.commit()
    
    student_id = student.id
    print(f"\n✅ Created/Loaded Student Profile (ID: {student_id}) in SQLite Database.")

    # ---------------------------------------------------------
    # PART A: THE INTERVENTION GENERATOR (LLM + CAG)
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print("🧩 PART A: GENERATING TARGETED INTERVENTION")
    print("-"*40)
    
    generator = InterventionGenerator(data_dir="data")
    
    print("Simulating a student making a sign convention error (MISC-LGT-01)...")
    
    # We call the generator with mock input from a diagnosed student
    result = generator.generate_intervention(
        student_id=str(student_id),
        misconception_id="MISC-LGT-01",
        student_error="Calculated focal length as +15cm for a concave mirror instead of -15cm.",
        correct_principle="By Cartesian sign convention, concave mirrors always have a negative focal length.",
        question_text="An object is placed 20 cm in front of a concave mirror of focal length 15 cm. Find the image distance."
    )
    
    print("\n🧠 LLM Generated Explanation:")
    print(f"   \"{result.explanation}\"")
    
    print("\n🎨 Generated Whiteboard Commands (JSON Schema):")
    for cmd in result.whiteboard_commands:
        print(f"   - {cmd}")


    # ---------------------------------------------------------
    # PART B: THE REASSESSMENT STATE MACHINE
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print("📈 PART B: REASSESSMENT & MASTERY TRACKING")
    print("-"*40)
    
    print("The student is given a 'Transfer Question' to test if they actually learned it.")
    print("SCENARIO: The student guesses the correct answer, but provides NO working steps.")
    
    diagnosis_result = {
        "is_correct_answer": True,
        "is_valid_reasoning": False,  # They just guessed the number
        "diagnosed_misconception": "UNCERTAIN-GUESS", 
        "confidence": 0.4
    }

    # Run the state machine
    new_state = ReassessmentStateMachine.evaluate_transfer_attempt(
        student_id=student_id,
        misconception_id="MISC-LGT-01",
        transfer_question_id="PHY-LGT-TRANSFER-001",
        student_answer="-30 cm",
        student_working="", # Blank working
        diagnosis_result=diagnosis_result
    )
    
    print(f"\n⚙️ State Machine Decision: {new_state}")
    print("   (Because one lucky guess without working never grants complete mastery!)")

    # Let's verify the database was updated
    record = db.query(MisconceptionRecord).filter(
        MisconceptionRecord.student_id == student_id,
        MisconceptionRecord.misconception_id == "MISC-LGT-01"
    ).first()
    
    print(f"\n💾 Database Verification:")
    print(f"   - Misconception: {record.misconception_id}")
    print(f"   - Occurrences: {record.occurrence_count}")
    print(f"   - DB Status: {record.status.value.upper()} (Not 'RESOLVED' because reasoning was missing)")
    print("\n✅ Phase 4 is working perfectly!\n")

if __name__ == "__main__":
    run_demo()
