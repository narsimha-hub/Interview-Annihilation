"""
applicant.py — 7-Section Interview Support
==========================================
Changes from previous version:
  1. question type "opening" replaces "personal" for Section 1
  2. New types: "technical", "jd", "behavioral", "situational", "closing"
  3. _build_agent_history passes based_on correctly (already done, unchanged)
  4. should_end() receives answered_count correctly (already done, unchanged)
  5. form_name passed to generate_first_question (already done, unchanged)
  6. personal_info now saves all plan fields: resume_topics, tech_skills,
     jd_requirements, behavioral_scenarios, situational_scenarios
     (needed so the report agent has full context)
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import io, pdfplumber, logging

from ..database import get_db
from .. import models
from ..services.ollama_service import (
    PersonalInfoAgent,
    NextQuestionAgent,
    score_answer,
)
from ..services.screening_agent import ResumeScreeningAgent
from ..services.report_service  import ReportAnalysisAgent, PDFReportBuilder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/applicants", tags=["applicants"])


# ── Pydantic schemas ────────────────────────────────────────────────────────────
class AnswerSubmit(BaseModel):
    applicant_id:   int
    answer:         str
    question_index: int

class ApplicantOut(BaseModel):
    id:                 int
    name:               Optional[str]
    email:              Optional[str]
    resume_match_score: Optional[float]
    status:             str
    class Config:
        from_attributes = True


# ── PDF extraction with OCR fallback ───────────────────────────────────────────
def _extract_pdf_text(file_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        if text.strip():
            return text
    except Exception as e:
        logger.warning(f"[PDF] pdfplumber failed: {e}")

    try:
        import pytesseract
        from pdf2image import convert_from_bytes
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        images    = convert_from_bytes(
            file_bytes, dpi=300,
            poppler_path=r"C:\Users\narsi\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin",
        )
        full_text = "\n".join(pytesseract.image_to_string(img, lang="eng") for img in images)
        return full_text
    except ImportError:
        logger.error("[PDF] pytesseract/pdf2image not installed.")
        return ""
    except Exception as e:
        logger.error(f"[PDF] OCR failed: {e}")
        return ""


# ── Helpers ─────────────────────────────────────────────────────────────────────
def _q_text(q) -> str:
    return q.get("question", str(q)) if isinstance(q, dict) else str(q)

def _q_type(q) -> str:
    return q.get("type", "resume") if isinstance(q, dict) else "resume"

def _q_topic(q) -> str:
    return q.get("topic", "") if isinstance(q, dict) else ""

def _q_based_on(q) -> str:
    return q.get("based_on", "") if isinstance(q, dict) else ""


def _build_agent_history(
    questions:       list,
    user_answers:    list,
    question_scores: list,
    conv_history:    list,
) -> list:
    """
    Build the structured history list that NextQuestionAgent.generate_next() expects.
    Each entry: {question, answer, score, type, topic, based_on, explanation}

    This is different from the raw conversation_history stored in DB
    (which uses role/content format). The agent needs the structured format.
    based_on is critical — it is used to track which resume/tech/JD/behavioral
    topics have already been asked, preventing repetition.
    """
    agent_history = []
    for i, q in enumerate(questions):
        answer = user_answers[i]    if i < len(user_answers)    else None
        score  = question_scores[i] if i < len(question_scores) else None
        if answer is None or score is None:
            continue

        # Extract explanation + weakness from conv_history
        explanation = ""
        weakness    = ""
        for entry in conv_history:
            if entry.get("role") == "answer" and entry.get("index") == i:
                explanation = entry.get("explanation", "")
                eval_data   = entry.get("eval", {})
                weakness    = eval_data.get("weakness", "") if isinstance(eval_data, dict) else ""
                break

        agent_history.append({
            "question":    _q_text(q),
            "answer":      answer,
            "score":       score,
            "type":        _q_type(q),
            "topic":       _q_topic(q),
            "based_on":    _q_based_on(q),   # key for dedup tracking in NextQuestionAgent
            "explanation": weakness or explanation,
        })
    return agent_history


# ── POST / — Upload resume + JD ─────────────────────────────────────────────────
@router.post("/")
async def create_applicant(
    name:     str        = Form(...),
    email:    str        = Form(...),
    resume:   UploadFile = File(...),
    job_desc: UploadFile = File(...),
    db:       Session    = Depends(get_db),
):
    # 1. Extract PDF text
    resume_bytes = await resume.read()
    jd_bytes     = await job_desc.read()
    resume_text  = _extract_pdf_text(resume_bytes)
    jd_text      = _extract_pdf_text(jd_bytes)

    if not resume_text.strip():
        raise HTTPException(400, "Could not extract text from resume PDF.")
    if not jd_text.strip():
        raise HTTPException(400, "Could not extract text from JD PDF.")

    # 2. Screening
    print(f"[Screening] Starting for {name}...")
    screener  = ResumeScreeningAgent()
    screening = screener.screen(resume_text, jd_text)

    final_score    = screening["final_score"]
    verdict        = screening["verdict"]
    matched_skills = screening["matched_skills"]
    missing_skills = screening["missing_skills"]
    status_map     = {"PROCEED": "shortlisted", "REVIEW": "review", "REJECT": "rejected"}
    status         = status_map.get(verdict, "pending")

    # 3. Save to DB
    applicant = models.Applicant(
        name                 = name,
        email                = email,
        resume_text          = resume_text,
        jd_text              = jd_text,
        resume_match_score   = final_score,
        status               = status,
        questions            = [],
        user_answers         = [],
        question_scores      = [],
        conversation_history = [],
        followup_count       = 0,
        interview_phase      = "opening",
        personal_info        = {
            "form_name":         name,           # stored so all agents can use it
            "screening_verdict": verdict,
            "verdict_color":     screening["verdict_color"],
            "dimension_scores":  screening["dimension_scores"],
            "dimension_notes":   screening["dimension_notes"],
            "matched_skills":    matched_skills,
            "missing_skills":    missing_skills,
            "candidate_level":   screening["candidate_level"],
            "required_level":    screening["required_level"],
            "education_field":   screening["education_field"],
            "key_projects":      screening["key_projects"],
            "screening_notes":   screening["screening_notes"],
            "screening_summary": screening["summary"],
        },
    )
    db.add(applicant)
    db.commit()
    db.refresh(applicant)

    # 4. Rejected — no interview
    if verdict == "REJECT":
        return {
            "applicant_id":     applicant.id,
            "name":             name,
            "status":           status,
            "final_score":      final_score,
            "verdict":          verdict,
            "verdict_color":    screening["verdict_color"],
            "screening_passed": False,
            "message":          "Candidate does not meet minimum requirements.",
            "matched_skills":   matched_skills,
            "missing_skills":   missing_skills,
            "screening_notes":  screening["screening_notes"],
            "dimension_scores": screening["dimension_scores"],
        }

    # 5. PersonalInfoAgent — extract resume info + generate Section 1 (opening Q)
    try:
        pi_agent           = PersonalInfoAgent()
        personal_info_data = pi_agent.extract_info(resume_text)
        personal_info_data["form_name"] = name  # always keep form name

        # Merge: screening data base, personal data on top, screening keys re-asserted
        merged = dict(applicant.personal_info)
        merged.update(personal_info_data)
        merged["form_name"]          = name
        merged["screening_verdict"]  = verdict
        merged["verdict_color"]      = screening["verdict_color"]
        merged["dimension_scores"]   = screening["dimension_scores"]
        merged["dimension_notes"]    = screening["dimension_notes"]
        merged["matched_skills"]     = matched_skills
        merged["missing_skills"]     = missing_skills
        merged["candidate_level"]    = screening["candidate_level"]
        merged["required_level"]     = screening["required_level"]
        merged["education_field"]    = screening["education_field"]
        merged["key_projects"]       = screening["key_projects"]
        merged["screening_notes"]    = screening["screening_notes"]
        merged["screening_summary"]  = screening["summary"]

        applicant.personal_info = merged
        flag_modified(applicant, "personal_info")

        # Generate Q1 — Section 1: Opening / Introduction
        q1 = pi_agent.generate_first_question(merged, candidate_name=name)
        applicant.questions = [q1]
        flag_modified(applicant, "questions")
        db.commit()
        db.refresh(applicant)

        first_q_text  = _q_text(q1)
        first_q_type  = _q_type(q1)   # "opening"
        first_q_topic = _q_topic(q1)

    except Exception as e:
        logger.error(f"PersonalInfoAgent failed: {e}")
        fallback_q = {
            "question": f"Hi {name}, could you start by introducing yourself — your educational background, the skills you have built, and what you have been working on recently?",
            "type":     "opening",
            "topic":    "introduction",
            "based_on": "self-introduction",
            "phase":    1,
        }
        applicant.questions = [fallback_q]
        flag_modified(applicant, "questions")
        db.commit()
        db.refresh(applicant)
        first_q_text  = fallback_q["question"]
        first_q_type  = "opening"
        first_q_topic = "introduction"

    return {
        "applicant_id":      applicant.id,
        "name":              name,
        "status":            status,
        "final_score":       final_score,
        "verdict":           verdict,
        "verdict_color":     screening["verdict_color"],
        "screening_passed":  True,
        "dimension_scores":  screening["dimension_scores"],
        "dimension_notes":   screening["dimension_notes"],
        "matched_skills":    matched_skills,
        "missing_skills":    missing_skills,
        "candidate_level":   screening["candidate_level"],
        "required_level":    screening["required_level"],
        "screening_notes":   screening["screening_notes"],
        "screening_summary": screening["summary"],
        "first_question":    first_q_text,
        "question_type":     first_q_type,
        "question_topic":    first_q_topic,
        "question_index":    0,
    }


# ── POST /submit-answer ─────────────────────────────────────────────────────────
@router.post("/submit-answer")
def submit_answer(payload: AnswerSubmit, db: Session = Depends(get_db)):
    applicant = db.query(models.Applicant).filter(
        models.Applicant.id == payload.applicant_id
    ).first()
    if not applicant:
        raise HTTPException(404, "Applicant not found")

    idx       = payload.question_index
    answer    = payload.answer.strip()
    questions = list(applicant.questions or [])

    if idx >= len(questions):
        raise HTTPException(
            400, f"Invalid question index {idx} — only {len(questions)} questions exist"
        )

    current_q       = questions[idx]
    current_q_text  = _q_text(current_q)
    current_q_type  = _q_type(current_q)
    current_q_topic = _q_topic(current_q)

    # ── Score the answer ────────────────────────────────────────────────────────
    try:
        score, explanation, eval_result = score_answer(
            current_q_text, answer, current_q_type, current_q_topic
        )
    except Exception as e:
        logger.error(f"score_answer failed: {e}")
        score       = 5
        explanation = "Scoring unavailable — default applied."
        eval_result = {
            "correctness": 5, "depth": 5, "clarity": 5,
            "practical": 5, "completeness": 5,
            "final_score": 5, "explanation": explanation,
            "strength": "", "weakness": "",
        }

    # ── Update DB arrays ────────────────────────────────────────────────────────
    user_answers    = list(applicant.user_answers    or [])
    question_scores = list(applicant.question_scores or [])
    conv_history    = list(applicant.conversation_history or [])

    while len(user_answers)    <= idx: user_answers.append(None)
    while len(question_scores) <= idx: question_scores.append(None)

    user_answers[idx]    = answer
    question_scores[idx] = score

    conv_history.append({
        "role":    "question",
        "content": current_q_text,
        "type":    current_q_type,
        "topic":   current_q_topic,
        "index":   idx,
    })
    conv_history.append({
        "role":        "answer",
        "content":     answer,
        "score":       score,
        "explanation": explanation,
        "eval":        eval_result if isinstance(eval_result, dict) else {},
        "index":       idx,
    })

    applicant.user_answers         = user_answers
    applicant.question_scores      = question_scores
    applicant.conversation_history = conv_history
    applicant.total_interview_score = sum(
        s for s in question_scores if isinstance(s, (int, float))
    )
    flag_modified(applicant, "user_answers")
    flag_modified(applicant, "question_scores")
    flag_modified(applicant, "conversation_history")

    # ── Stats ────────────────────────────────────────────────────────────────────
    valid_scores   = [s for s in question_scores if isinstance(s, (int, float))]
    answered_count = len(valid_scores)
    avg_score      = round(sum(valid_scores) / answered_count, 1) if answered_count else 0

    # ── Lightweight end check (full check runs inside generate_next) ────────────
    nq_agent = NextQuestionAgent()
    early_stop, stop_reason = nq_agent.should_end(
        scores        = valid_scores,
        answered_count = answered_count,
    )
    if early_stop:
        applicant.status = "completed"
        flag_modified(applicant, "status")
        db.commit()
        print(f"[Interview] Early stop — {stop_reason}")
        return {
            "score":              score,
            "explanation":        explanation,
            "eval":               eval_result if isinstance(eval_result, dict) else {},
            "total_score":        applicant.total_interview_score,
            "answers_given":      answered_count,
            "average_score":      avg_score,
            "interview_complete": True,
            "next_question":      None,
        }

    # ── Build agent_history for NextQuestionAgent ───────────────────────────────
    agent_history = _build_agent_history(
        questions, user_answers, question_scores, conv_history
    )

    personal_info  = dict(applicant.personal_info or {})
    followup_count = int(applicant.followup_count or 0)

    # ── Generate next question ──────────────────────────────────────────────────
    next_q      = None
    jd_analysis = None
    try:
        next_q, jd_analysis = nq_agent.generate_next(
            resume_text    = applicant.resume_text or "",
            jd_text        = applicant.jd_text     or "",
            personal_info  = personal_info,
            history        = agent_history,
            scores         = valid_scores,
            followup_count = followup_count,
            session_key    = str(applicant.id),
        )
    except Exception as e:
        logger.error(f"NextQuestionAgent.generate_next failed: {e}")
        # Emergency fallback — a technical question so the interview can continue
        next_q = {
            "question": "Can you describe a technical challenge you solved recently and explain your approach step by step?",
            "type":     "technical",
            "topic":    "problem solving",
            "based_on": "tech:general",
            "phase":    3, "difficulty": 3,
        }

    # ── Save plan data back to DB for the report ────────────────────────────────
    # The report agent needs matched_skills, missing_skills, and plan fields.
    # We save all plan fields the first time jd_analysis is returned (first tech question).
    if jd_analysis:
        plan_keys = [
            ("matched_skills",          "matched_skills"),
            ("missing_skills",          "missing_skills"),
            ("candidate_level",         "candidate_experience_level"),
            ("resume_topics",           "resume_topics"),
            ("tech_skills",             "tech_skills"),
            ("jd_requirements",         "jd_requirements"),
            ("behavioral_scenarios",    "behavioral_scenarios"),
            ("situational_scenarios",   "situational_scenarios"),
            ("gap_skills",              "gap_skills"),
        ]
        updated = False
        for plan_key, info_key in plan_keys:
            val = jd_analysis.get(plan_key)
            if val:
                personal_info[info_key] = val
                updated = True
        # Also build key_resume_projects for report compat
        resume_topics = jd_analysis.get("resume_topics", [])
        if resume_topics:
            personal_info["key_resume_projects"] = [
                t.get("specific_ref", "") for t in resume_topics
                if isinstance(t, dict) and t.get("specific_ref")
            ]
            updated = True
        if updated:
            applicant.personal_info = personal_info
            flag_modified(applicant, "personal_info")

    # ── Track follow-up count ────────────────────────────────────────────────────
    if next_q and _q_type(next_q) == "followup":
        applicant.followup_count = followup_count + 1

    # ── Append next question ────────────────────────────────────────────────────
    if next_q:
        questions.append(next_q)
        applicant.questions = questions
        flag_modified(applicant, "questions")

    db.commit()

    # ── Interview complete (agent returned None) ─────────────────────────────────
    if not next_q:
        applicant.status = "completed"
        db.commit()
        return {
            "score":              score,
            "explanation":        explanation,
            "eval":               eval_result if isinstance(eval_result, dict) else {},
            "total_score":        applicant.total_interview_score,
            "answers_given":      answered_count,
            "average_score":      avg_score,
            "interview_complete": True,
            "next_question":      None,
        }

    return {
        "score":              score,
        "explanation":        explanation,
        "eval":               eval_result if isinstance(eval_result, dict) else {},
        "total_score":        applicant.total_interview_score,
        "answers_given":      answered_count,
        "average_score":      avg_score,
        "interview_complete": False,
        "next_question":      _q_text(next_q),
        "question_type":      _q_type(next_q),
        "question_topic":     _q_topic(next_q),
        "question_index":     len(questions) - 1,
    }


# ── GET /{id}/summary ───────────────────────────────────────────────────────────
@router.get("/{applicant_id}/summary")
def get_summary(applicant_id: int, db: Session = Depends(get_db)):
    applicant = db.query(models.Applicant).filter(
        models.Applicant.id == applicant_id
    ).first()
    if not applicant:
        raise HTTPException(404, "Applicant not found")

    questions    = applicant.questions            or []
    user_answers = applicant.user_answers         or []
    scores       = applicant.question_scores      or []
    history      = applicant.conversation_history or []
    personal     = applicant.personal_info        or {}

    valid_scores = [s for s in scores if isinstance(s, (int, float))]
    answered     = len(valid_scores)
    avg_score    = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else 0

    qa_pairs = []
    for i, q in enumerate(questions):
        ans_entry = next(
            (e for e in history if e.get("role") == "answer" and e.get("index") == i),
            None,
        )
        qa_pairs.append({
            "question":    _q_text(q),
            "type":        _q_type(q),
            "topic":       _q_topic(q),
            "answer":      user_answers[i] if i < len(user_answers) else None,
            "score":       scores[i]       if i < len(scores)       else None,
            "explanation": ans_entry.get("explanation", "") if ans_entry else "",
            "eval":        ans_entry.get("eval", {})        if ans_entry else {},
        })

    return {
        "applicant_id":       applicant.id,
        "name":               applicant.name,
        "email":              applicant.email,
        "status":             applicant.status,
        "resume_match_score": applicant.resume_match_score,
        "screening_verdict":  personal.get("screening_verdict", ""),
        "verdict_color":      personal.get("verdict_color", "red"),
        "dimension_scores":   personal.get("dimension_scores", {}),
        "screening_summary":  personal.get("screening_summary", ""),
        "screening_notes":    personal.get("screening_notes", []),
        "matched_skills":     personal.get("matched_skills", []),
        "missing_skills":     personal.get("missing_skills", []),
        "candidate_level":    personal.get("candidate_level", ""),
        "total_questions":    len(questions),
        "answers_given":      answered,
        "average_score":      avg_score,
        "total_score":        applicant.total_interview_score,
        "qa_pairs":           qa_pairs,
    }


# ── GET /{id}/report ────────────────────────────────────────────────────────────
@router.get("/{applicant_id}/report")
def download_report(applicant_id: int, db: Session = Depends(get_db)):
    from fastapi.responses import StreamingResponse

    applicant = db.query(models.Applicant).filter(
        models.Applicant.id == applicant_id
    ).first()
    if not applicant:
        raise HTTPException(404, "Applicant not found")

    personal  = applicant.personal_info   or {}
    questions = applicant.questions        or []
    scores    = applicant.question_scores  or []
    answers   = applicant.user_answers     or []
    history   = applicant.conversation_history or []

    valid_scores  = [s for s in scores if isinstance(s, (int, float))]
    average_score = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else 0

    transcript = []
    for i, q in enumerate(questions):
        ans_entry = next(
            (e for e in history if e.get("role") == "answer" and e.get("index") == i),
            None,
        )
        transcript.append({
            "question":    _q_text(q),
            "type":        _q_type(q),
            "topic":       _q_topic(q),
            "answer":      answers[i] if i < len(answers) else "No answer",
            "score":       scores[i]  if i < len(scores)  else 0,
            "explanation": ans_entry.get("explanation", "") if ans_entry else "",
        })

    analysis_agent = ReportAnalysisAgent()
    analysis = analysis_agent.analyze(
        transcript              = transcript,
        resume_match_score      = applicant.resume_match_score or 0,
        average_interview_score = average_score,
        matched_skills          = personal.get("matched_skills", []),
        missing_skills          = personal.get("missing_skills", []),
    )

    builder   = PDFReportBuilder()
    pdf_bytes = builder.build(
        name               = applicant.name or "Unknown",
        email              = applicant.email or "",
        resume_match_score = applicant.resume_match_score or 0,
        screening_verdict  = personal.get("screening_verdict", ""),
        verdict_color      = personal.get("verdict_color", "red"),
        dimension_scores   = personal.get("dimension_scores", {}),
        matched_skills     = personal.get("matched_skills", []),
        missing_skills     = personal.get("missing_skills", []),
        average_score      = average_score,
        total_questions    = len(questions),
        transcript         = transcript,
        analysis           = analysis,
    )

    safe_name = (applicant.name or "candidate").replace(" ", "_")
    filename  = f"interview_report_{safe_name}_{applicant_id}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── GET / — list all ────────────────────────────────────────────────────────────
@router.get("/", response_model=List[ApplicantOut])
def list_applicants(db: Session = Depends(get_db)):
    return (
        db.query(models.Applicant)
        .order_by(models.Applicant.created_at.desc())
        .limit(100)
        .all()
    )


# ── GET /{id} ───────────────────────────────────────────────────────────────────
@router.get("/{applicant_id}", response_model=ApplicantOut)
def get_applicant(applicant_id: int, db: Session = Depends(get_db)):
    applicant = db.query(models.Applicant).filter(
        models.Applicant.id == applicant_id
    ).first()
    if not applicant:
        raise HTTPException(404, "Not found")
    return applicant