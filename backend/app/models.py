from sqlalchemy import Column, Integer, String, Text, Float, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from .database import Base


class Applicant(Base):
    __tablename__ = "applicants"

    id    = Column(Integer, primary_key=True, index=True)
    name  = Column(String(100), nullable=True)
    email = Column(String(120), nullable=True, index=True)

    resume_text = Column(Text, nullable=True)
    jd_text     = Column(Text, nullable=True)

    resume_match_score = Column(Float, nullable=True)
    status             = Column(String(50), default="pending", index=True)
    created_at         = Column(DateTime(timezone=True), server_default=func.now())

    # ── Interview ─────────────────────────────────────────────────
    # questions: list of dicts, grows one at a time
    # [{"question":"...","type":"personal|technical|experience|followup",
    #   "topic":"...","phase":1|2|3,"difficulty":1-5,"based_on":"..."}]
    questions              = Column(JSONB, nullable=True, default=list)
    current_question_index = Column(Integer, default=0, nullable=False)
    conversation_history   = Column(JSONB, nullable=True, default=list)

    # index-aligned with questions list
    question_scores = Column(JSONB, nullable=True, default=list)
    user_answers    = Column(JSONB, nullable=True, default=list)

    total_interview_score = Column(Float, default=0.0, nullable=False)

    # ── New columns ───────────────────────────────────────────────
    # how many follow-up questions injected so far (max 3)
    followup_count = Column(Integer, default=0, nullable=False)

    # current phase: "personal" | "technical" | "complete"
    interview_phase = Column(String(20), default="personal", nullable=True)

    # structured info extracted by PersonalInfoAgent from resume
    # {"name","location","education","hobbies_interests","career_goal_clues",...}
    personal_info = Column(JSONB, nullable=True, default=dict)

    def __repr__(self):
        return f"<Applicant id={self.id} name={self.name} status={self.status}>"