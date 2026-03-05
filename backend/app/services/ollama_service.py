# backend/app/services/ollama_service.py

import ollama
import re
from typing import List, Dict, Tuple
import numpy as np

def get_embedding(text: str, model: str = "nomic-embed-text") -> List[float]:
    """Generate embedding vector for a piece of text"""
    try:
        response = ollama.embeddings(model=model, prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"[Ollama Embedding Error] {str(e)}")
        return []


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors"""
    vec_a = np.array(a)
    vec_b = np.array(b)
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


def ollama_semantic_score(resume_text: str, jd_text: str) -> Tuple[float, Dict]:
    """Semantic similarity score using Ollama embeddings"""
    resume_emb = get_embedding(resume_text)
    jd_emb = get_embedding(jd_text)

    if not resume_emb or not jd_emb:
        return 0.0, {"error": "Embedding generation failed"}

    sim = cosine_similarity(resume_emb, jd_emb)
    score = float(round(sim * 100, 1))

    return score, {
        "overall_similarity": score,
        "note": "0–100 semantic match (higher = better alignment)"
    }


def generate_first_question(
    jd_text: str,
    resume_text: str,
    model: str = "llama3:latest"
) -> str:
    """
    Generate ONLY the first (opening) interview question.
    """
    prompt = f"""You are a strict, senior technical interviewer.

Job Description key requirements:
{jd_text[:1800]}

Candidate resume summary:
{resume_text[:1800]}

Generate ONE strong opening question.
It should:
- Target the most critical skill or experience from the JD
- Be specific and challenging
- Probe real-world application or depth

Output ONLY the question text.
No introduction, no numbering, no extra lines."""

    try:
        print(f"[FIRST QUESTION] Generating with model: {model}")
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.7}
        )
        question = response['response'].strip()
        print("[FIRST QUESTION GENERATED]", question)
        return question
    except Exception as e:
        print(f"[FIRST QUESTION ERROR] {str(e)}")
        return "Tell me about your background and experience relevant to this role."


def generate_next_question(
    jd_text: str,
    resume_text: str,
    history: List[Dict[str, any]],
    model: str = "llama3:latest"
) -> str:
    """
    Generate the next adaptive question based on previous conversation.
    """
    history_text = "\n".join(
        f"Previous Q: {h['question']}\nCandidate A: {h['answer']}\nScore: {h['score']}\n---"
        for h in history
    ) if history else "No previous answers yet."

    prompt = f"""You are continuing a technical interview.

Job Description:
{jd_text[:1500]}

Candidate resume:
{resume_text[:1500]}

Previous conversation:
{history_text}

Generate ONE next question that:
- Builds on previous answers
- Probes deeper into weak/unclear areas or interesting claims
- Targets remaining key JD requirements
- Is specific, behavioral or technical as appropriate
- Keeps total interview around 5 questions

Output ONLY the question text — nothing else."""

    try:
        print(f"[NEXT QUESTION] Generating with history length: {len(history)}")
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.65}
        )
        question = response['response'].strip()
        print("[NEXT QUESTION GENERATED]", question)
        return question
    except Exception as e:
        print(f"[NEXT QUESTION ERROR] {str(e)}")
        return "Can you elaborate more on your previous answer?"


def generate_interview_questions(
    jd_text: str,
    resume_text: str,
    num_questions: int = 5,
    model: str = "llama3:latest"
) -> List[str]:
    """
    Legacy function - generates all questions at once (for fallback or non-adaptive mode)
    """
    prompt = f"""You are a senior technical interviewer for a junior software developer role.

Job Description key requirements:
{jd_text[:2000]}

Candidate's relevant experience from resume:
{resume_text[:2000]}

Generate exactly {num_questions} targeted questions.
Mix technical deep-dive, behavioral, and problem-solving.

Output ONLY the numbered list.
Start directly with "1. Question text" — no introduction, no "Here are", no explanations."""

    try:
        print(f"[BATCH QUESTIONS] Generating {num_questions} questions with model: {model}")
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.7}
        )
        raw = response['response'].strip()
        print("[RAW BATCH RESPONSE]", raw)

        lines = raw.split('\n')
        questions = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                cleaned = re.sub(r'^\s*[\d\.\-\*Q]+\s*', '', line).strip()
                if cleaned and "Here are" not in cleaned and "questions:" not in cleaned.lower():
                    questions.append(cleaned)

        print("[PARSED BATCH QUESTIONS]", questions)
        return questions[:num_questions]

    except Exception as e:
        print(f"[BATCH QUESTION GEN ERROR] {str(e)}")
        return ["Error generating batch questions — check Ollama"]


def score_answer(question: str, answer: str, model: str = "llama3:latest") -> Tuple[int, str]:
    """
    Score candidate's answer from 0 to 10 with short explanation.
    """
    prompt = f"""You are a strict, fair technical interviewer.

Question:
{question}

Candidate answer:
{answer}

Score from 0 to 10:
- Relevance & correctness: 0–4
- Depth & technical accuracy: 0–3
- Clarity, structure & communication: 0–2
- Real-world examples / applicability: 0–1

Be strict:
- 10 = exceptional (perfect, detailed, real examples)
- 7–8 = strong/good
- 5 = average/acceptable
- <5 = weak or irrelevant

Output exactly:
Score: X
Explanation: one short, honest sentence"""

    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.3}
        )
        raw = response['response'].strip()
        print("[RAW SCORING]", raw)

        score = 5
        explanation = "Default scoring applied"

        lines = raw.split('\n')
        for line in lines:
            if line.startswith("Score:"):
                try:
                    score = int(line.split("Score:")[1].strip())
                except:
                    pass
            if line.startswith("Explanation:"):
                explanation = line.split("Explanation:", 1)[1].strip()

        return max(0, min(10, score)), explanation

    except Exception as e:
        print(f"[SCORING ERROR] {str(e)}")
        return 0, "Scoring failed — technical issue"