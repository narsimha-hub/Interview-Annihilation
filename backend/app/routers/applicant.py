from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from sqlalchemy.orm import Session
from typing import Annotated, List, Dict, Any

from pydantic import BaseModel

from ..database import get_db
from .. import models
from ..schemas import ApplicantOut
from ..services.pdf import extract_text_from_pdf
from ..services.ollama_service import (
    ollama_semantic_score,
    generate_first_question,
    generate_next_question,
    score_answer
)

router = APIRouter(prefix="/applicants", tags=["applicants"])

# Pydantic model for answer submission
class AnswerSubmission(BaseModel):
    question_index: int
    answer_text: str


@router.post("/", response_model=None)
async def create_applicant(
    name: Annotated[str, Form()],
    email: Annotated[str, Form()],
    resume: UploadFile = File(...),
    jd: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not resume.content_type.startswith("application/pdf"):
        raise HTTPException(400, "Resume must be a PDF file")
    if not jd.content_type.startswith("application/pdf"):
        raise HTTPException(400, "Job Description must be a PDF file")

    try:
        resume_bytes = await resume.read()
        jd_bytes = await jd.read()

        resume_text = extract_text_from_pdf(resume_bytes)
        jd_text = extract_text_from_pdf(jd_bytes)

        match_score, score_breakdown = ollama_semantic_score(resume_text, jd_text)
        print(f"[DEBUG] Semantic score: {match_score}")

        if match_score < 30:
            status = "rejected_low_match"
        elif match_score < 60:
            status = "screening_weak"
        else:
            status = "screening_passed"

        questions = []
        if status == "screening_passed":
            first_question = generate_first_question(jd_text, resume_text)
            questions = [first_question]
            print("[ADAPTIVE] Generated first question:", first_question)

        applicant = models.Applicant(
            name=name,
            email=email,
            resume_text=resume_text,
            jd_text=jd_text,
            resume_match_score=match_score,
            status=status,
            questions=questions,
            current_question_index=0,
            conversation_history=[],
            question_scores=[],
            user_answers=[],
            total_interview_score=0.0
        )

        db.add(applicant)
        db.commit()
        db.refresh(applicant)

        print("[DEBUG] Initial applicant created with first question:", applicant.questions)

        return {
            "id": applicant.id,
            "name": applicant.name,
            "email": applicant.email,
            "status": applicant.status,
            "resume_match_score": applicant.resume_match_score,
            "created_at": applicant.created_at,
            "semantic_breakdown": score_breakdown,
            "first_question": applicant.questions[0] if applicant.questions else None
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(500, detail=f"Unexpected error: {str(e)}")


@router.get("/{applicant_id}", response_model=ApplicantOut)
def get_applicant(applicant_id: int, db: Session = Depends(get_db)):
    applicant = db.query(models.Applicant).filter(models.Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(404, "Applicant not found")
    return applicant


@router.get("/{applicant_id}/questions")
def get_questions(applicant_id: int, db: Session = Depends(get_db)):
    applicant = db.query(models.Applicant).filter(models.Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(404, "Applicant not found")
    return {"questions": applicant.questions or []}


@router.get("/{applicant_id}/next-question")
def get_next_question(applicant_id: int, db: Session = Depends(get_db)):
    applicant = db.query(models.Applicant).filter(models.Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(404, "Applicant not found")

    if not applicant.questions or len(applicant.questions) == 0:
        raise HTTPException(400, "No questions generated yet")

    index = applicant.current_question_index

    if index >= len(applicant.questions):
        return {
            "applicant_id": applicant_id,
            "message": "All questions have been asked",
            "total_questions": len(applicant.questions),
            "current_index": index
        }

    next_question = applicant.questions[index]

    # Do NOT advance index here — advance only after successful submit
    return {
        "applicant_id": applicant_id,
        "question_index": index,
        "question": next_question,
        "total_questions": len(applicant.questions),
        "remaining": len(applicant.questions) - index,
        "progress": f"{index + 1}/{len(applicant.questions)}"
    }


@router.post("/{applicant_id}/submit-answer")
def submit_answer(
    applicant_id: int,
    submission: AnswerSubmission,
    db: Session = Depends(get_db)
):
    applicant = db.query(models.Applicant).filter(models.Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(404, "Applicant not found")

    if not applicant.questions or len(applicant.questions) == 0:
        raise HTTPException(400, "No questions generated")

    index = submission.question_index

    # Only allow answering the current (last) question
    if index != len(applicant.questions) - 1:
        raise HTTPException(400, "You can only answer the current question")

    user_answer = submission.answer_text.strip()
    if not user_answer:
        raise HTTPException(400, "Answer text cannot be empty")

    # Score it
    score, explanation = score_answer(
        question=applicant.questions[-1],
        answer=user_answer
    )

    # Save to history for context in next question
    history = applicant.conversation_history or []
    history.append({
        "question": applicant.questions[-1],
        "answer": user_answer,
        "score": score
    })

    # Update lists (re-assign to trigger SQLAlchemy change detection)
    current_scores = (applicant.question_scores or []).copy()
    current_answers = (applicant.user_answers or []).copy()

    current_scores.append(score)
    current_answers.append(user_answer)

    applicant.question_scores = current_scores
    applicant.user_answers = current_answers

    # Update total score as AVERAGE of all submitted scores
    submitted_scores = [s for s in current_scores if s is not None]
    applicant.total_interview_score = round(sum(submitted_scores) / len(submitted_scores), 1) if submitted_scores else 0.0

    # Generate next question if not finished
    next_question = None
    if len(applicant.questions) < 5:
        next_question = generate_next_question(
            jd_text=applicant.jd_text,
            resume_text=applicant.resume_text,
            history=history
        )
        applicant.questions.append(next_question)
        print("[ADAPTIVE] Generated next question:", next_question)

    # Advance progress
    applicant.current_question_index += 1
    applicant.conversation_history = history

    db.commit()
    db.refresh(applicant)

    # Prepare response
    submitted_scores = applicant.question_scores or []
    total_score = round(sum(submitted_scores) / len(submitted_scores), 1) if submitted_scores else 0.0

    response = {
        "applicant_id": applicant_id,
        "question_index": index,
        "question": applicant.questions[index],
        "submitted_answer": user_answer,
        "score": score,
        "explanation": explanation,
        "current_progress": applicant.current_question_index,
        "total_questions": len(applicant.questions),
        "total_interview_score": total_score,  # Now average
        "scored_questions": len([s for s in submitted_scores if s is not None])
    }

    if next_question:
        response["next_question"] = next_question

    return response


@router.get("/{applicant_id}/summary")
def get_interview_summary(applicant_id: int, db: Session = Depends(get_db)):
    applicant = db.query(models.Applicant).filter(models.Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(404, "Applicant not found")

    if not applicant.questions or len(applicant.questions) == 0:
        raise HTTPException(400, "No interview data available")

    submitted_scores = [s for s in applicant.question_scores if s is not None]
    total_score = round(sum(submitted_scores) / len(submitted_scores), 1) if submitted_scores else 0.0

    summary_items = []
    for i, question in enumerate(applicant.questions):
        answer = applicant.user_answers[i] if i < len(applicant.user_answers) else None
        score = applicant.question_scores[i] if i < len(applicant.question_scores) else None
        summary_items.append({
            "question_index": i,
            "question": question,
            "answer": answer,
            "score": score,
            "status": "answered" if answer else "pending"
        })

    return {
        "applicant_id": applicant_id,
        "name": applicant.name,
        "email": applicant.email,
        "status": applicant.status,
        "resume_match_score": applicant.resume_match_score,
        "total_questions": len(applicant.questions),
        "answered_questions": len(submitted_scores),
        "current_progress": applicant.current_question_index,
        "total_interview_score": total_score,  # Average
        "interview_summary": summary_items
    }