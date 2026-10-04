import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Enum, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import enum

# Determine database file path (store in data directory or backend)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "..", "data", "relearn.db")

# Create data directory if it doesn't exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class MisconceptionStatus(str, enum.Enum):
    ACTIVE = "active"
    IMPROVING = "improving"
    RESOLVED = "resolved"

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    overall_proficiency = Column(Float, default=0.0)

    attempts = relationship("Attempt", back_populates="student")
    misconception_records = relationship("MisconceptionRecord", back_populates="student")

class Attempt(Base):
    __tablename__ = "attempts"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    question_id = Column(String, index=True)
    student_answer = Column(Text)
    extracted_steps = Column(Text)
    model_1_diagnosis = Column(String, nullable=True) # Misconception ID or baseline class
    model_2_sequence_pattern = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    
    # Optional field if it is a reassessment attempt
    is_reassessment = Column(Integer, default=0) # 0 or 1
    transfer_question_id = Column(String, nullable=True)

    student = relationship("Student", back_populates="attempts")

class MisconceptionRecord(Base):
    __tablename__ = "misconception_records"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    misconception_id = Column(String, index=True) # Matches taxonomy e.g., MISC-LGT-01
    occurrence_count = Column(Integer, default=1)
    status = Column(Enum(MisconceptionStatus), default=MisconceptionStatus.ACTIVE)

    student = relationship("Student", back_populates="misconception_records")

def init_db():
    Base.metadata.create_all(bind=engine)
