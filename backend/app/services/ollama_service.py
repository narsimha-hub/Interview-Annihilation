# backend/app/services/ollama_service.py
#
# Architecture: ONE question generated at a time.
# Upload  → PersonalInfoAgent extracts info → generates Q1 only
# Submit  → scores answer → NextQuestionAgent generates exactly ONE next question
# Ends    → avg score > 7 (after 3+ answers) OR 10 questions reached

import ollama
import re
import json
from typing import List, Dict, Tuple, Optional
import numpy as np


# ─────────────────────────────────────────────────────────
#  EMBEDDINGS + SEMANTIC SCORE
# ─────────────────────────────────────────────────────────

def get_embedding(text: str, model: str = "nomic-embed-text") -> List[float]:
    try:
        response = ollama.embeddings(model=model, prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return []


def cosine_similarity(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def ollama_semantic_score(resume_text: str, jd_text: str) -> Tuple[float, Dict]:
    re_emb = get_embedding(resume_text)
    jd_emb = get_embedding(jd_text)
    if not re_emb or not jd_emb:
        return 0.0, {"error": "Embedding failed"}
    score = round(cosine_similarity(re_emb, jd_emb) * 100, 1)
    return score, {"overall_similarity": score}


# ─────────────────────────────────────────────────────────
#  SHARED HELPERS
# ─────────────────────────────────────────────────────────

def _call_ollama(prompt: str, temperature: float,
                 model: str = "llama3:latest") -> str:
    resp = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": temperature, "num_predict": 1024}
    )
    return resp['response'].strip()


def _parse_json(raw: str) -> dict:
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r'(\{.*\})', cleaned, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    raise ValueError(f"No JSON found:\n{raw[:300]}")


# ─────────────────────────────────────────────────────────
#  AGENT 1 — PERSONAL INFO AGENT
#  Run ONCE on upload.
#  extract_info()           → stores structured info in DB
#  generate_first_question()→ returns Q1 immediately
# ─────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """You are a recruiter reading a resume.

RESUME:
{resume_text}

Extract personal background. Return ONLY valid JSON, no markdown:

{{
  "name": "",
  "location": "",
  "education": "",
  "total_experience_years": "",
  "hobbies_interests": [],
  "career_goal_clues": "",
  "notable_achievements": [],
  "previous_companies": [],
  "certifications": []
}}

Use empty string or empty list if not found. Return ONLY the JSON."""


_FIRST_Q_PROMPT = """You are a friendly interviewer opening a real conversation.

CANDIDATE BACKGROUND:
- Name              : {name}
- Education         : {education}
- Experience        : {experience} years
- Location          : {location}
- Hobbies/Interests : {hobbies}
- Career Goal       : {career_goal}
- Achievements      : {achievements}
- Previous Companies: {companies}

Generate exactly ONE warm opening question.

Rules:
- MUST reference a specific detail from the background above
- Do NOT ask "Tell me about yourself"
- Conversational tone, feels written for THIS person
- Invites storytelling about their background or motivation

Return ONLY valid JSON, no markdown:
{{
  "question": "Full question text",
  "topic":    "what this explores",
  "based_on": "specific resume detail that triggered this"
}}"""


class PersonalInfoAgent:
    """
    Runs once on upload.
    extract_info()            → returns structured dict (saved to DB)
    generate_first_question() → returns Q1 as a question dict
    """

    def __init__(self, model: str = "llama3:latest"):
        self.model = model

    def extract_info(self, resume_text: str) -> Dict:
        print("[PersonalInfoAgent] Extracting personal info...")
        prompt = _EXTRACT_PROMPT.format(resume_text=resume_text[:3000])
        try:
            raw  = _call_ollama(prompt, temperature=0.1, model=self.model)
            info = _parse_json(raw)
            print(f"[PersonalInfoAgent] name={info.get('name')} "
                  f"edu={info.get('education')} "
                  f"exp={info.get('total_experience_years')}yr")
            return info
        except Exception as e:
            print(f"[PersonalInfoAgent] Extract failed: {e}")
            return {
                "name": "", "location": "", "education": "",
                "total_experience_years": "", "hobbies_interests": [],
                "career_goal_clues": "", "notable_achievements": [],
                "previous_companies": [], "certifications": []
            }

    def generate_first_question(self, personal_info: Dict) -> Dict:
        print("[PersonalInfoAgent] Generating Q1...")
        prompt = _FIRST_Q_PROMPT.format(
            name        = personal_info.get("name", "the candidate"),
            education   = personal_info.get("education", "not specified"),
            experience  = personal_info.get("total_experience_years", "unknown"),
            location    = personal_info.get("location", "not specified"),
            hobbies     = ", ".join(personal_info.get("hobbies_interests", [])) or "not mentioned",
            career_goal = personal_info.get("career_goal_clues", "not stated"),
            achievements= "; ".join(personal_info.get("notable_achievements", [])) or "none listed",
            companies   = ", ".join(personal_info.get("previous_companies", [])) or "none listed",
        )
        try:
            raw  = _call_ollama(prompt, temperature=0.5, model=self.model)
            data = _parse_json(raw)
            q    = data.get("question", "").strip()
            if not q:
                raise ValueError("Empty question returned")
            return {
                "question":   q,
                "topic":      data.get("topic", "personal background"),
                "based_on":   data.get("based_on", "resume"),
                "type":       "personal",
                "phase":      1,
                "difficulty": 1,
            }
        except Exception as e:
            print(f"[PersonalInfoAgent] Q1 gen failed: {e}")
            return {
                "question":   "Can you walk me through your background and what led you to apply for this role?",
                "topic":      "career motivation",
                "based_on":   "resume overview",
                "type":       "personal",
                "phase":      1,
                "difficulty": 1,
            }


# ─────────────────────────────────────────────────────────
#  AGENT 2 — NEXT QUESTION AGENT
#  Called inside submit-answer after scoring.
#  Decides type → generates exactly ONE question.
#
#  Decision order:
#   1. Follow-up   if score < 5 or answer < 20 words  (max 3 total)
#   2. Personal    if personal_answered < 2
#   3. Technical   or Experience alternating
#   4. None        if avg > 7 (3+ answers) OR count >= 10
# ─────────────────────────────────────────────────────────

_ANALYSIS_PROMPT = """You are a technical recruiter.

JOB DESCRIPTION:
{jd_text}

CANDIDATE RESUME:
{resume_text}

Return ONLY valid JSON, no markdown:
{{
  "candidate_experience_level": "junior|mid|senior",
  "matched_skills":             [],
  "missing_skills":             [],
  "key_resume_projects":        [],
  "critical_jd_requirements":   [],
  "summary":                    ""
}}"""


_NEXT_TECH_PROMPT = """You are a senior interviewer mid-interview.

JOB ROLE        : {job_role}
EXPERIENCE LEVEL: {level}
MATCHED SKILLS  : {matched}
SKILL GAPS      : {gaps}
KEY PROJECTS    : {projects}
JD REQUIREMENTS : {jd_reqs}

QUESTIONS ALREADY ASKED — DO NOT REPEAT OR REPHRASE ANY OF THESE:
{asked_questions}

RECENT CONVERSATION (last 3 exchanges):
{history}

Generate exactly ONE {q_type} question that:
- Is COMPLETELY DIFFERENT from every question in the list above
- Covers a topic NOT already covered in the conversation
- Targets a skill from matched_skills OR exposes a gap not yet explored
- Matches difficulty to {level}: junior=1-2, mid=2-3, senior=4-5
- References actual skills or projects from the resume — not generic

Return ONLY valid JSON, no markdown:
{{
  "question":   "Full question text",
  "topic":      "specific skill being tested",
  "based_on":   "what in resume or JD triggered this",
  "difficulty": 2
}}"""


_FOLLOWUP_PROMPT = """You are a sharp interviewer. The candidate gave a weak or vague answer.

QUESTION  : {question}
ANSWER    : {answer}
SCORE     : {score}/10
REASON    : {explanation}

Generate ONE follow-up that:
- Probes exactly what was weak or missing in their answer
- References what they said or failed to say
- Pushes for real depth or a concrete example
- Is specific, not generic

Return ONLY valid JSON, no markdown:
{{
  "question": "Follow-up question text",
  "topic":    "what weakness this probes",
  "based_on": "what in their answer triggered this"
}}"""


_PERSONAL_FOLLOWON_PROMPT = """You are a friendly interviewer continuing a personal conversation.

CANDIDATE BACKGROUND:
{personal_info}

QUESTIONS ALREADY ASKED — DO NOT REPEAT ANY:
{asked_questions}

LAST EXCHANGE:
Q: {last_q}
A: {last_a}

Generate ONE follow-on personal question that:
- Is COMPLETELY DIFFERENT from every question already asked above
- Digs deeper into something specific they just said
- Feels natural and conversational — not generic

Return ONLY valid JSON, no markdown:
{{
  "question": "Question text",
  "topic":    "what this explores",
  "based_on": "what in their answer triggered this"
}}"""


# Module-level analysis cache  {session_key: analysis_dict}
# Avoids re-running the resume vs JD analysis on every question
_analysis_cache: Dict[str, Dict] = {}


class NextQuestionAgent:
    """
    Called after every submitted answer.
    Returns exactly ONE question dict, or None if interview should end.
    """

    def __init__(self, model: str = "llama3:latest"):
        self.model = model

    # ── end condition ─────────────────────────────────────
    def should_end(self, scores: List[int], question_count: int) -> Tuple[bool, str]:
        if question_count >= 10:
            return True, "max_questions_reached"
        if len(scores) >= 3:
            avg = sum(scores) / len(scores)
            if avg > 7:
                return True, f"high_performance_avg_{round(avg, 1)}"
        return False, ""

    # ── main entry point ──────────────────────────────────
    def generate_next(
        self,
        resume_text:    str,
        jd_text:        str,
        personal_info:  Dict,
        history:        List[Dict],  # [{question, answer, score, type, explanation}]
        scores:         List[int],
        followup_count: int,
        session_key:    str = "",
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Returns (next_question_dict_or_None, analysis_dict_or_None).
        analysis is populated only when a tech/experience question is generated.
        Caller should save analysis to DB when not None.
        """
        question_count = len(history)

        # ── 1. end check ──────────────────────────────────
        end, reason = self.should_end(scores, question_count)
        if end:
            print(f"[NextQuestionAgent] Ending interview: {reason}")
            return None, None

        last            = history[-1] if history else {}
        last_score      = last.get("score", 10)
        last_answer     = last.get("answer", "")
        last_question   = last.get("question", "")
        last_type       = last.get("type", "personal")
        last_expl       = last.get("explanation", "")

        # ── 2. follow-up? ─────────────────────────────────
        word_count     = len(last_answer.strip().split())
        needs_followup = (
            followup_count < 3
            and (last_score < 5 or word_count < 20)
            and last_type != "followup"   # never follow-up a follow-up
        )
        if needs_followup:
            return self._gen_followup(last_question, last_answer,
                                      last_score, last_expl), None

        # ── 3. personal phase check ───────────────────────
        personal_answered = sum(1 for h in history if h.get("type") == "personal")
        if personal_answered < 2 and last_type == "personal":
            return self._gen_personal_followon(personal_info,
                                               last_question, last_answer,
                                               history=history), None

        # ── 4. technical / experience alternating ─────────
        tech_answered = sum(
            1 for h in history
            if h.get("type") in ("technical", "experience")
        )
        q_type   = "experience" if tech_answered % 2 == 1 else "technical"
        analysis = self._get_analysis(resume_text, jd_text, session_key)
        return self._gen_tech_question(analysis, personal_info, history, q_type), analysis

    # ── generators ───────────────────────────────────────

    def _get_analysis(self, resume_text: str, jd_text: str,
                      key: str) -> Dict:
        global _analysis_cache
        if key and key in _analysis_cache:
            return _analysis_cache[key]

        print("[NextQuestionAgent] Running resume vs JD analysis...")
        prompt = _ANALYSIS_PROMPT.format(
            jd_text=jd_text[:3000],
            resume_text=resume_text[:3000]
        )
        try:
            raw      = _call_ollama(prompt, temperature=0.1, model=self.model)
            analysis = _parse_json(raw)
            print(f"[NextQuestionAgent] level={analysis.get('candidate_experience_level')} "
                  f"matched={len(analysis.get('matched_skills', []))} "
                  f"gaps={len(analysis.get('missing_skills', []))}")
            if key:
                _analysis_cache[key] = analysis
            return analysis
        except Exception as e:
            print(f"[NextQuestionAgent] Analysis failed: {e}")
            return {
                "candidate_experience_level": "mid",
                "matched_skills": [], "missing_skills": [],
                "key_resume_projects": [], "critical_jd_requirements": [],
                "summary": ""
            }

    def _gen_followup(self, question: str, answer: str,
                      score: int, explanation: str) -> Dict:
        print(f"[NextQuestionAgent] Generating follow-up (score={score})...")
        prompt = _FOLLOWUP_PROMPT.format(
            question=question, answer=answer[:400],
            score=score, explanation=explanation
        )
        try:
            raw  = _call_ollama(prompt, temperature=0.4, model=self.model)
            data = _parse_json(raw)
            return {
                "question":   data.get("question", "Can you give a concrete example?"),
                "topic":      data.get("topic", "clarification"),
                "based_on":   data.get("based_on", "previous answer"),
                "type":       "followup",
                "phase":      3,
                "difficulty": 3,
            }
        except Exception as e:
            print(f"[NextQuestionAgent] Follow-up failed: {e}")
            fu_fallbacks = [
                "Can you walk me through a concrete example that demonstrates what you just described?",
                "What would you do differently if you faced this situation again?",
                "Can you be more specific — what exactly did you personally do in that situation?",
                "What was the outcome and how did you measure success?",
            ]
            used_qs = [h.get("question","") for h in []] 
            chosen = fu_fallbacks[0]
            return {
                "question":   chosen,
                "topic":      "clarification",
                "based_on":   "previous answer",
                "type":       "followup",
                "phase":      3,
                "difficulty": 3,
            }

    def _gen_personal_followon(self, personal_info: Dict,
                               last_q: str, last_a: str,
                               history: List[Dict] = None) -> Dict:
        history = history or []
        print("[NextQuestionAgent] Generating personal follow-on...")
        def _safe_str(v):
            if isinstance(v, list):
                return [str(i) if not isinstance(i, dict) else
                        next((str(x) for x in i.values() if x), str(i))
                        for i in v if i]
            return v

        info_text = json.dumps(
            {k: _safe_str(v) for k, v in personal_info.items() if v and v != [] and v != ""},
            indent=2
        )
        asked_so_far = "\n".join(
            f"- {h['question']}" for h in history if h.get("question")
        ) or "None yet."
        prompt = _PERSONAL_FOLLOWON_PROMPT.format(
            personal_info=info_text[:1000],
            last_q=last_q, last_a=last_a[:400],
            asked_questions=asked_so_far,
        )
        try:
            raw  = _call_ollama(prompt, temperature=0.5, model=self.model)
            data = _parse_json(raw)
            return {
                "question":   data.get("question", ""),
                "topic":      data.get("topic", "personal background"),
                "based_on":   data.get("based_on", "previous answer"),
                "type":       "personal",
                "phase":      1,
                "difficulty": 1,
            }
        except Exception as e:
            print(f"[NextQuestionAgent] Personal follow-on failed: {e}")
            personal_fallbacks = [
                ("What aspect of your background do you think makes you strongest for this role?", "self-assessment"),
                ("How has your education shaped the way you approach technical problems?", "education impact"),
                ("What kind of work environment brings out your best performance?", "work style"),
                ("Where do you see yourself in two years and how does this role fit that path?", "career goals"),
            ]
            used_qs = [h.get("question","") for h in history if h.get("type") == "personal"]
            chosen_q, chosen_topic = personal_fallbacks[0]
            for q_text_fb, topic_fb in personal_fallbacks:
                if not any(q_text_fb.lower()[:30] in u.lower() for u in used_qs):
                    chosen_q, chosen_topic = q_text_fb, topic_fb
                    break
            return {
                "question":   chosen_q,
                "topic":      chosen_topic,
                "based_on":   "resume",
                "type":       "personal",
                "phase":      1,
                "difficulty": 1,
            }

    def _gen_tech_question(self, analysis: Dict, personal_info: Dict,
                           history: List[Dict], q_type: str) -> Dict:
        print(f"[NextQuestionAgent] Generating {q_type} question...")

        # Build full list of ALL questions asked so far — explicit blacklist for LLM
        asked_questions = "\n".join(
            f"- {h['question']}"
            for h in history
            if h.get("question")
        ) or "None yet."

        # Recent conversation context (last 3 for token efficiency)
        history_text = "\n".join(
            f"[{h.get('type','?').upper()}] Q: {h['question']}\n"
            f"A: {h['answer'][:150]}\nScore: {h['score']}/10"
            for h in history[-3:]
        )
        def _safe_join(items, sep=", "):
            """Join a list safely — handles dicts, None, mixed types."""
            if not items:
                return ""
            parts = []
            for item in items:
                if isinstance(item, dict):
                    # extract any string value from the dict
                    parts.append(next((str(v) for v in item.values() if v), str(item)))
                elif item:
                    parts.append(str(item))
            return sep.join(parts)

        prompt = _NEXT_TECH_PROMPT.format(
            job_role        = personal_info.get("career_goal_clues", "Software Engineer"),
            level           = analysis.get("candidate_experience_level", "mid"),
            matched         = _safe_join(analysis.get("matched_skills", [])) or "Not detected",
            gaps            = _safe_join(analysis.get("missing_skills", [])) or "None",
            projects        = _safe_join(analysis.get("key_resume_projects", []), "; ") or "Not specified",
            jd_reqs         = _safe_join(analysis.get("critical_jd_requirements", []), "; ") or "Not specified",
            asked_questions = asked_questions,
            history         = history_text or "No previous answers yet.",
            q_type          = q_type,
        )
        try:
            raw  = _call_ollama(prompt, temperature=0.35, model=self.model)
            data = _parse_json(raw)
            return {
                "question":   data.get("question", ""),
                "topic":      data.get("topic", q_type),
                "based_on":   data.get("based_on", ""),
                "type":       q_type,
                "phase":      2,
                "difficulty": int(data.get("difficulty", 3)),
            }
        except Exception as e:
            print(f"[NextQuestionAgent] Tech question failed: {e}")
            # Use asked_questions to pick a fallback that hasn't been used
            used = [h.get("question","") for h in history]
            tech_fallbacks = [
                ("How do you approach debugging a production issue under time pressure?", "debugging"),
                ("Explain the difference between SQL and NoSQL databases and when to use each.", "databases"),
                ("How would you secure a REST API endpoint that handles sensitive data?", "security"),
                ("Describe how you would implement caching in a web application.", "caching"),
                ("What is your approach to writing testable code?", "testing"),
            ]
            exp_fallbacks = [
                ("Describe the most technically challenging bug you have fixed and how you solved it.", "problem solving"),
                ("Walk me through how you have handled a disagreement with a teammate on a technical decision.", "collaboration"),
                ("Tell me about a time you had to learn a new technology quickly to deliver a project.", "adaptability"),
                ("Describe a project where you had to balance quality with a tight deadline.", "delivery"),
                ("What is the project you are most proud of and what was your specific contribution?", "ownership"),
            ]
            pool = tech_fallbacks if q_type == "technical" else exp_fallbacks
            # Pick first fallback not already used
            chosen_q, chosen_topic = pool[0]
            for q_text_fb, topic_fb in pool:
                if not any(q_text_fb.lower()[:30] in u.lower() for u in used):
                    chosen_q, chosen_topic = q_text_fb, topic_fb
                    break
            return {
                "question":   chosen_q,
                "topic":      chosen_topic,
                "based_on":   "JD requirements",
                "type":       q_type,
                "phase":      2,
                "difficulty": 3,
            }


# ─────────────────────────────────────────────────────────
#  SCORING
# ─────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
#  ANSWER EVALUATION AGENT
#  Multi-dimensional scoring across 5 criteria.
#  Two small focused LLM calls so llama3:latest stays reliable.
# ─────────────────────────────────────────────────────────

# Weights per question type
_EVAL_WEIGHTS = {
    "technical":  {"correctness": 0.35, "depth": 0.30, "clarity": 0.15, "practical": 0.15, "completeness": 0.05},
    "experience": {"correctness": 0.20, "depth": 0.25, "clarity": 0.20, "practical": 0.30, "completeness": 0.05},
    "personal":   {"correctness": 0.10, "depth": 0.20, "clarity": 0.35, "practical": 0.25, "completeness": 0.10},
    "followup":   {"correctness": 0.30, "depth": 0.35, "clarity": 0.20, "practical": 0.10, "completeness": 0.05},
}

_SCORE_PROMPT = """You are a senior interviewer evaluating a candidate answer.

QUESTION TYPE: {q_type}
QUESTION: {question}
CANDIDATE ANSWER: {answer}

Rate each dimension from 1-10:
Correctness: how factually accurate is the answer
Depth: how thorough and detailed is the explanation
Clarity: how clear and well-structured is the communication
Practical: how well they show real-world application
Completeness: how fully they addressed the question

Output ONLY these 5 lines, nothing else:
Correctness: X
Depth: X
Clarity: X
Practical: X
Completeness: X"""

_INSIGHT_PROMPT = """You are a senior interviewer. A candidate answered this question:

QUESTION: {question}
ANSWER: {answer}
SCORES: correctness={correctness}, depth={depth}, clarity={clarity}

Write ONE sentence for each:
Strength: what the candidate did well
Weakness: what was missing or weak"""


class AnswerEvaluationAgent:
    """
    Multi-dimensional answer evaluation.
    Returns scores across 5 criteria + weighted final score + strength/weakness.
    """

    def __init__(self, model: str = "llama3:latest"):
        self.model = model

    def evaluate(
        self,
        question:  str,
        answer:    str,
        q_type:    str = "technical",
        topic:     str = "",
    ) -> Dict:
        """
        Returns full evaluation dict:
        {
            correctness, depth, clarity, practical, completeness,  # 1-10 each
            final_score,      # weighted int 1-10
            explanation,      # one sentence summary
            strength,         # what candidate did well
            weakness,         # what was missing
        }
        """
        print(f"[EvalAgent] Evaluating {q_type} answer ({len(answer.split())} words)...")

        # ── Call 1: dimension scores ──────────────────────
        dims = self._get_scores(question, answer, q_type)

        # ── Call 2: strength + weakness insight ──────────
        insights = self._get_insights(question, answer, dims)

        # ── Weighted final score ──────────────────────────
        weights  = _EVAL_WEIGHTS.get(q_type, _EVAL_WEIGHTS["technical"])
        raw      = (
            dims["correctness"]  * weights["correctness"]  +
            dims["depth"]        * weights["depth"]        +
            dims["clarity"]      * weights["clarity"]      +
            dims["practical"]    * weights["practical"]    +
            dims["completeness"] * weights["completeness"]
        )
        final = max(1, min(10, round(raw)))

        # ── Build explanation ─────────────────────────────
        explanation = insights.get("strength", "") or (
            f"Score {final}/10 — correctness:{dims['correctness']} "
            f"depth:{dims['depth']} clarity:{dims['clarity']}"
        )

        print(f"[EvalAgent] Final={final} | "
              f"C={dims['correctness']} D={dims['depth']} "
              f"Cl={dims['clarity']} P={dims['practical']} Co={dims['completeness']}")

        return {
            "correctness":  dims["correctness"],
            "depth":        dims["depth"],
            "clarity":      dims["clarity"],
            "practical":    dims["practical"],
            "completeness": dims["completeness"],
            "final_score":  final,
            "explanation":  explanation,
            "strength":     insights.get("strength", ""),
            "weakness":     insights.get("weakness", ""),
        }

    def _get_scores(self, question: str, answer: str, q_type: str) -> Dict:
        """Call 1 — get 5 dimension scores."""
        prompt = _SCORE_PROMPT.format(
            q_type   = q_type.capitalize(),
            question = question[:300],
            answer   = answer[:500],
        )
        defaults = {"correctness": 5, "depth": 5, "clarity": 5,
                    "practical": 5, "completeness": 5}
        try:
            resp = ollama.generate(
                model   = self.model,
                prompt  = prompt,
                options = {"temperature": 0.2, "num_predict": 80},
            )
            raw  = resp["response"].strip()
            dims = dict(defaults)
            key_map = {
                "correctness":  "correctness",
                "depth":        "depth",
                "clarity":      "clarity",
                "practical":    "practical",
                "completeness": "completeness",
            }
            for line in raw.split("\n"):
                line = line.strip()
                for key, field in key_map.items():
                    if line.lower().startswith(key + ":"):
                        m = re.search(r"(\d+)", line)
                        if m:
                            dims[field] = max(1, min(10, int(m.group(1))))
            return dims
        except Exception as e:
            print(f"[EvalAgent] Score call failed: {e}")
            # fallback: rough score from answer length + keywords
            words = len(answer.strip().split())
            base  = min(7, max(3, words // 20))
            return {k: base for k in defaults}

    def _get_insights(self, question: str, answer: str, dims: Dict) -> Dict:
        """Call 2 — get strength and weakness sentence."""
        prompt = _INSIGHT_PROMPT.format(
            question    = question[:250],
            answer      = answer[:400],
            correctness = dims["correctness"],
            depth       = dims["depth"],
            clarity     = dims["clarity"],
        )
        try:
            resp = ollama.generate(
                model   = self.model,
                prompt  = prompt,
                options = {"temperature": 0.4, "num_predict": 100},
            )
            raw      = resp["response"].strip()
            strength = ""
            weakness = ""
            for line in raw.split("\n"):
                ls = line.strip()
                if ls.lower().startswith("strength:"):
                    strength = ls.split(":", 1)[1].strip()
                elif ls.lower().startswith("weakness:"):
                    weakness = ls.split(":", 1)[1].strip()
            return {"strength": strength, "weakness": weakness}
        except Exception as e:
            print(f"[EvalAgent] Insight call failed: {e}")
            return {"strength": "", "weakness": ""}


# ── Backward-compatible wrapper ───────────────────────────
_eval_agent = AnswerEvaluationAgent()

def score_answer(
    question: str,
    answer:   str,
    q_type:   str = "technical",
    topic:    str = "",
    model:    str = "llama3:latest",
) -> Tuple[int, str, Dict]:
    """
    Returns (final_score, explanation, full_eval_dict).
    Callers that only unpack 2 values still work if they ignore the 3rd.
    """
    result = _eval_agent.evaluate(question, answer, q_type, topic)
    return result["final_score"], result["explanation"], result


# ─────────────────────────────────────────────────────────
#  LEGACY STUBS
# ─────────────────────────────────────────────────────────

def generate_first_question(jd_text, resume_text, model="llama3:latest"):
    agent = PersonalInfoAgent(model=model)
    info  = agent.extract_info(resume_text)
    return agent.generate_first_question(info).get("question", "Tell me about yourself.")

def generate_next_question(jd_text, resume_text, history, model="llama3:latest"):
    return ""

def generate_interview_questions(jd_text, resume_text, num_questions=5, model="llama3:latest"):
    return []