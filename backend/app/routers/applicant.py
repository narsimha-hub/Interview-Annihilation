from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from sqlalchemy.orm import Session
from typing import Annotated

from pydantic import BaseModel

from sqlalchemy.orm.attributes import flag_modified
from ..database import get_db
from .. import models
from ..schemas import ApplicantOut
from ..services.pdf import extract_text_from_pdf
from ..services.ollama_service import (
    ollama_semantic_score,
    PersonalInfoAgent,
    NextQuestionAgent,
    score_answer,
)

router = APIRouter(prefix="/applicants", tags=["applicants"])


class AnswerSubmission(BaseModel):
    question_index: int
    answer_text: str


def _q_text(q) -> str:
    """Safely extract question string from dict or plain string."""
    if isinstance(q, dict):
        return q.get("question", "")
    return str(q)


def _q_type(q) -> str:
    if isinstance(q, dict):
        return q.get("type", "unknown")
    return "unknown"


def _q_topic(q) -> str:
    if isinstance(q, dict):
        return q.get("topic", "")
    return ""


# ─────────────────────────────────────────────────────────
#  POST /applicants/
#  1. Extract text from PDFs
#  2. Semantic screening score
#  3. PersonalInfoAgent: extract info + generate Q1 ONLY
#  4. Store everything — interview starts immediately
# ─────────────────────────────────────────────────────────
@router.post("/", response_model=None)
async def create_applicant(
    name:   Annotated[str, Form()],
    email:  Annotated[str, Form()],
    resume: UploadFile = File(...),
    jd:     UploadFile = File(...),
    db:     Session = Depends(get_db)
):
    if not resume.content_type.startswith("application/pdf"):
        raise HTTPException(400, "Resume must be a PDF")
    if not jd.content_type.startswith("application/pdf"):
        raise HTTPException(400, "JD must be a PDF")

    try:
        resume_text = extract_text_from_pdf(await resume.read())
        jd_text     = extract_text_from_pdf(await jd.read())

        # ── Screening ────────────────────────────────────
        match_score, score_breakdown = ollama_semantic_score(resume_text, jd_text)
        print(f"[CREATE] Semantic score: {match_score}")

        if match_score < 30:
            status = "rejected_low_match"
        elif match_score < 60:
            status = "screening_weak"
        else:
            status = "screening_passed"

        first_question = None
        personal_info  = {}

        if status == "screening_passed":
            # ── PersonalInfoAgent: extract + Q1 ─────────
            p_agent      = PersonalInfoAgent()
            personal_info = p_agent.extract_info(resume_text)
            first_q_dict  = p_agent.generate_first_question(personal_info)
            first_question = first_q_dict
            print(f"[CREATE] Q1 ready: {_q_text(first_q_dict)[:80]}...")

        applicant = models.Applicant(
            name=name,
            email=email,
            resume_text=resume_text,
            jd_text=jd_text,
            resume_match_score=match_score,
            status=status,
            # questions list starts with only Q1
            questions=[first_question] if first_question else [],
            current_question_index=0,
            conversation_history=[],
            question_scores=[],
            user_answers=[],
            total_interview_score=0.0,
            followup_count=0,
            interview_phase="personal",
            personal_info=personal_info,
        )

        db.add(applicant)
        db.commit()
        db.refresh(applicant)

        return {
            "id":                 applicant.id,
            "name":               applicant.name,
            "email":              applicant.email,
            "status":             applicant.status,
            "resume_match_score": applicant.resume_match_score,
            "created_at":         applicant.created_at,
            "semantic_breakdown": score_breakdown,
            "total_questions":    len(applicant.questions),
            "first_question":     _q_text(first_question) if first_question else None,
            "question_type":      _q_type(first_question) if first_question else None,
            "question_topic":     _q_topic(first_question) if first_question else None,
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(500, detail=f"Error: {str(e)}")


# ─────────────────────────────────────────────────────────
#  GET /applicants/{id}
# ─────────────────────────────────────────────────────────
@router.get("/{applicant_id}", response_model=ApplicantOut)
def get_applicant(applicant_id: int, db: Session = Depends(get_db)):
    applicant = db.query(models.Applicant).filter(
        models.Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(404, "Not found")
    return applicant


# ─────────────────────────────────────────────────────────
#  GET /applicants/{id}/next-question
# ─────────────────────────────────────────────────────────
@router.get("/{applicant_id}/next-question")
def get_next_question(applicant_id: int, db: Session = Depends(get_db)):
    applicant = db.query(models.Applicant).filter(
        models.Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(404, "Not found")
    if not applicant.questions:
        raise HTTPException(400, "No questions yet")

    idx = applicant.current_question_index
    if idx >= len(applicant.questions):
        return {"message": "All questions done", "interview_complete": True}

    q = applicant.questions[idx]
    return {
        "applicant_id":    applicant_id,
        "question_index":  idx,
        "question":        _q_text(q),
        "question_type":   _q_type(q),
        "topic":           _q_topic(q),
        "total_questions": len(applicant.questions),
        "progress":        f"{idx + 1} so far",
    }


# ─────────────────────────────────────────────────────────
#  POST /applicants/{id}/submit-answer
#
#  Flow per call:
#   1. Validate + score the answer
#   2. Append to history
#   3. NextQuestionAgent decides + generates next Q
#      (or returns None = interview done)
#   4. If next Q exists → append to questions list
#   5. Return score + next question to frontend
# ─────────────────────────────────────────────────────────
@router.post("/{applicant_id}/submit-answer")
def submit_answer(
    applicant_id: int,
    submission:   AnswerSubmission,
    db:           Session = Depends(get_db)
):
    applicant = db.query(models.Applicant).filter(
        models.Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(404, "Not found")
    if not applicant.questions:
        raise HTTPException(400, "No questions generated")

    idx = submission.question_index
    if idx != applicant.current_question_index:
        raise HTTPException(400, "Can only answer the current question")

    answer = submission.answer_text.strip()
    if not answer:
        raise HTTPException(400, "Answer cannot be empty")

    current_q = applicant.questions[idx]
    q_text    = _q_text(current_q)

    # ── 1. Score ─────────────────────────────────────────
    q_type_val = _q_type(current_q)
    q_topic_val = _q_topic(current_q)
    score, explanation, eval_result = score_answer(
        question = q_text,
        answer   = answer,
        q_type   = q_type_val,
        topic    = q_topic_val,
    )
    print(f"[SUBMIT] Q{idx} score={score} type={q_type_val} explanation={explanation}")

    # ── 2. Update state ──────────────────────────────────
    history         = list(applicant.conversation_history or [])
    current_scores  = list(applicant.question_scores or [])
    current_answers = list(applicant.user_answers or [])

    history.append({
        "question":     q_text,
        "answer":       answer,
        "score":        score,
        "explanation":  explanation,
        "type":         q_type_val,
        "topic":        q_topic_val,
        "correctness":  eval_result.get("correctness",  score),
        "depth":        eval_result.get("depth",        score),
        "clarity":      eval_result.get("clarity",      score),
        "practical":    eval_result.get("practical",    score),
        "completeness": eval_result.get("completeness", score),
        "strength":     eval_result.get("strength",     ""),
        "weakness":     eval_result.get("weakness",     ""),
    })
    current_scores.append(score)
    current_answers.append(answer)

    applicant.question_scores        = current_scores
    applicant.user_answers           = current_answers
    applicant.conversation_history   = history
    applicant.current_question_index = idx + 1

    # Force SQLAlchemy to detect JSONB column changes
    flag_modified(applicant, "question_scores")
    flag_modified(applicant, "user_answers")
    flag_modified(applicant, "conversation_history")

    submitted = [s for s in current_scores if s is not None]
    applicant.total_interview_score  = round(
        sum(submitted) / len(submitted), 1) if submitted else 0.0

    # ── 3. Generate next question ─────────────────────────
    followup_count = getattr(applicant, 'followup_count', 0) or 0
    nq_agent       = NextQuestionAgent()

    next_q, jd_analysis = nq_agent.generate_next(
        resume_text    = applicant.resume_text,
        jd_text        = applicant.jd_text,
        personal_info  = applicant.personal_info or {},
        history        = history,
        scores         = current_scores,
        followup_count = followup_count,
        session_key    = str(applicant_id),
    )

    # Save analysis (matched_skills, skill_gaps etc) to DB so report can use it
    if jd_analysis:
        p_info = dict(applicant.personal_info or {})
        p_info["matched_skills"]             = jd_analysis.get("matched_skills", [])
        p_info["missing_skills"]             = jd_analysis.get("missing_skills", [])
        p_info["candidate_experience_level"] = jd_analysis.get("candidate_experience_level", "")
        p_info["key_resume_projects"]        = jd_analysis.get("key_resume_projects", [])
        p_info["critical_jd_requirements"]   = jd_analysis.get("critical_jd_requirements", [])
        applicant.personal_info = p_info
        flag_modified(applicant, "personal_info")

    # ── 4. Append next question if exists ─────────────────
    if next_q:
        questions_copy = list(applicant.questions)
        questions_copy.append(next_q)
        applicant.questions = questions_copy
        flag_modified(applicant, "questions")

        # track follow-up count
        if next_q.get("type") == "followup":
            applicant.followup_count = followup_count + 1

    is_finished = next_q is None

    db.commit()
    db.refresh(applicant)

    # ── 5. Build response ─────────────────────────────────
    response = {
        "applicant_id":          applicant_id,
        "question_index":        idx,
        "question":              q_text,
        "question_type":         q_type_val,
        "submitted_answer":      answer,
        "score":                 score,
        "explanation":           explanation,
        "evaluation": {
            "correctness":  eval_result.get("correctness",  score),
            "depth":        eval_result.get("depth",        score),
            "clarity":      eval_result.get("clarity",      score),
            "practical":    eval_result.get("practical",    score),
            "completeness": eval_result.get("completeness", score),
            "strength":     eval_result.get("strength",     ""),
            "weakness":     eval_result.get("weakness",     ""),
        },
        "current_progress":      applicant.current_question_index,
        "total_questions":       len(applicant.questions),
        "total_interview_score": applicant.total_interview_score,
        "scored_questions":      len(submitted),
        "interview_complete":    is_finished,
    }

    if not is_finished and next_q:
        response["next_question"] = {
            "index":    applicant.current_question_index,
            "question": _q_text(next_q),
            "type":     _q_type(next_q),
            "topic":    _q_topic(next_q),
            "progress": f"{applicant.current_question_index + 1} so far",
        }

    return response


# ─────────────────────────────────────────────────────────
#  GET /applicants/{id}/summary
# ─────────────────────────────────────────────────────────
@router.get("/{applicant_id}/summary")
def get_interview_summary(applicant_id: int, db: Session = Depends(get_db)):
    applicant = db.query(models.Applicant).filter(
        models.Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(404, "Not found")
    if not applicant.questions:
        raise HTTPException(400, "No interview data")

    submitted = [s for s in (applicant.question_scores or []) if s is not None]
    total     = round(sum(submitted) / len(submitted), 1) if submitted else 0.0

    items = []
    for i, q in enumerate(applicant.questions):
        ans   = applicant.user_answers[i]    if i < len(applicant.user_answers or [])    else None
        sc    = applicant.question_scores[i] if i < len(applicant.question_scores or []) else None
        items.append({
            "question_index": i,
            "question":       _q_text(q),
            "question_type":  _q_type(q),
            "topic":          _q_topic(q),
            "answer":         ans,
            "score":          sc,
            "status":         "answered" if ans else "pending",
        })

    return {
        "applicant_id":          applicant_id,
        "name":                  applicant.name,
        "email":                 applicant.email,
        "status":                applicant.status,
        "resume_match_score":    applicant.resume_match_score,
        "total_questions":       len(applicant.questions),
        "answered_questions":    len(submitted),
        "current_progress":      applicant.current_question_index,
        "total_interview_score": total,
        "interview_summary":     items,
    }

# ─────────────────────────────────────────────────────────
#  GET /applicants/{id}/report
#  Generates a full PDF interview report and streams it.
# ─────────────────────────────────────────────────────────
@router.get("/{applicant_id}/report")
def download_report(applicant_id: int, db: Session = Depends(get_db)):
    from fastapi.responses import Response
    from ..services.report_service import ReportAnalysisAgent, PDFReportBuilder

    applicant = db.query(models.Applicant).filter(
        models.Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(404, "Applicant not found")
    if not applicant.questions:
        raise HTTPException(400, "No interview data available for this applicant")

    questions    = applicant.questions    or []
    answers      = applicant.user_answers or []
    scores_raw   = applicant.question_scores or []
    history      = applicant.conversation_history or []
    personal_info = applicant.personal_info or {}

    # pull explanations from conversation history
    explanations = [h.get("explanation", "") for h in history]

    # pull matched_skills / skill_gaps — saved by submit-answer from jd_analysis
    matched_skills = (personal_info.get("matched_skills") or
                      personal_info.get("analysis_matched_skills") or [])
    skill_gaps     = (personal_info.get("missing_skills") or
                      personal_info.get("analysis_skill_gaps") or [])

    # fallback: check in-memory analysis cache (same process, same session)
    if not matched_skills or not skill_gaps:
        from ..services.ollama_service import _analysis_cache
        cached = _analysis_cache.get(str(applicant_id), {})
        if not matched_skills:
            matched_skills = cached.get("matched_skills", [])
        if not skill_gaps:
            skill_gaps     = cached.get("missing_skills", [])

    submitted = [s for s in scores_raw if s is not None]
    interview_score = round(sum(submitted) / len(submitted), 1) if submitted else 0.0

    role = personal_info.get("career_goal_clues", "Software Engineer")

    # ── Run LLM analysis ─────────────────────────────────
    agent    = ReportAnalysisAgent()
    analysis = agent.analyse(
        name            = applicant.name,
        role            = role,
        match_score     = applicant.resume_match_score or 0,
        interview_score = interview_score,
        matched_skills  = matched_skills,
        skill_gaps      = skill_gaps,
        questions       = questions,
        answers         = answers,
        scores          = scores_raw,
    )

    # ── Build PDF ─────────────────────────────────────────
    builder   = PDFReportBuilder()
    pdf_bytes = builder.build(
        applicant_id    = applicant_id,
        name            = applicant.name,
        email           = applicant.email,
        created_at      = applicant.created_at,
        match_score     = applicant.resume_match_score or 0,
        interview_score = interview_score,
        status          = applicant.status,
        matched_skills  = matched_skills,
        skill_gaps      = skill_gaps,
        questions       = questions,
        answers         = answers,
        scores          = scores_raw,
        explanations    = explanations,
        analysis        = analysis,
    )

    filename = f"interview_report_{applicant.name.replace(' ','_')}_{applicant_id}.pdf"
    return Response(
        content     = pdf_bytes,
        media_type  = "application/pdf",
        headers     = {"Content-Disposition": f'attachment; filename="{filename}"'},
    )