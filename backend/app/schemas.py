# app/schemas.py

from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional, List, Dict, Any


class ApplicantOut(BaseModel):
    id: int
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    status: str
    resume_match_score: Optional[float] = None
    created_at: datetime
    questions: Optional[List[str]] = None              # list of generated questions
    current_question_index: Optional[int] = None        # current progress
    question_scores: Optional[List[Optional[int]]] = None  # scores per question
    user_answers: Optional[List[Optional[str]]] = None     # user's submitted answers (optional)

    class Config:
        from_attributes = True  # ← enables compatibility with SQLAlchemy ORM objects
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnswerSubmission(BaseModel):
    question_index: int = Field(..., ge=0, description="Index of the question being answered")
    answer_text: str = Field(..., min_length=10, description="The candidate's written or spoken answer")


# Optional: Response model for submit-answer (cleaner API docs)
class AnswerResponse(BaseModel):
    applicant_id: int
    question_index: int
    question: str
    submitted_answer: str
    score: int
    explanation: str
    current_progress: int
    total_questions: int

    class Config:
        from_attributes = True