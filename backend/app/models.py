from sqlalchemy import Column, Integer, String, Text, Float, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from .database import Base


class Applicant(Base):
    __tablename__ = "applicants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=True)
    email = Column(String(120), nullable=True, index=True)

    resume_text = Column(Text, nullable=True)
    jd_text = Column(Text, nullable=True)

    resume_match_score = Column(Float, nullable=True)
    status = Column(String(50), default="pending", index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Interview fields
    conversation_history = Column(JSONB, nullable=True, default=list)
    questions = Column(JSONB, nullable=True, default=list)  # list of question strings
    current_question_index = Column(Integer, default=0, nullable=False)

    # JSONB for per-question data
    question_scores = Column(JSONB, nullable=True, default=list)  # [7, 8, None, ...]
    user_answers = Column(JSONB, nullable=True, default=list)     # ["answer1", "answer2", None]

    # Total score (sum of all question scores) - stored in DB
    total_interview_score = Column(Float, default=0.0, nullable=False)

    def __repr__(self):
        return f"<Applicant(id={self.id}, name='{self.name}', status='{self.status}')>"