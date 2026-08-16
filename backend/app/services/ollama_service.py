"""
ollama_service.py — 7-Section Interview Engine
================================================

INTERVIEW STRUCTURE (fixed order, every interview)
====================================================
  Section 1 — Opening / Introduction         (1 question  — always Q1)
  Section 2 — Resume-Based Questions         (2-3 questions — specific resume items)
  Section 3 — Technical Questions            (2-3 questions — JD tech skills)
  Section 4 — JD-Based Questions             (1-2 questions — role expectations)
  Section 5 — Behavioral / HR Questions      (2 questions  — STAR format)
  Section 6 — Situational Questions          (1-2 questions — what would you do if)
  Section 7 — Closing / Candidate Questions  (1 question  — fixed warm closer)

  Follow-ups (max 3 total) — injected anywhere score < 5 or answer < 20 words

STOP ALGORITHM
==============
  Hard cap     : 15 questions answered
  Early stop   : avg >= 8.0 after 6+ answers AND sections 1-3 complete
  Struggle stop: 3 consecutive scores < 4 after section 2 started
  Normal end   : all 7 sections complete (closing question always fires last)

ARCHITECTURE NOTE
=================
  One NextQuestionAgent, seven private generator methods.
  One _build_interview_plan() call cached per session.
  The plan now produces: resume_topics, tech_skills, jd_requirements,
  behavioral_scenarios, situational_scenarios — one LLM call at the start.
"""

import re
import json
import os
import random
from typing import List, Dict, Tuple, Optional
from groq import Groq


# ─────────────────────────────────────────────────────────
#  GROQ CLIENT  (lazy init so .env is loaded first)
# ─────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"
_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        from .config import settings
        key = settings.groq_api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        _client = Groq(api_key=key)
    return _client


def _call_groq(prompt: str, temperature: float = 0.3,
               max_tokens: int = 1024) -> str:
    response = _get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _parse_json(raw: str) -> dict:
    """Strip markdown fences then parse JSON from LLM response."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    raise ValueError(f"No valid JSON in response:\n{raw[:300]}")


# ─────────────────────────────────────────────────────────
#  STT CONTAMINATION GUARD
# ─────────────────────────────────────────────────────────

def _is_contaminated(question: str, answer: str) -> bool:
    """
    Returns True when the answer is essentially the question echoed back —
    which happens when the browser mic picks up TTS audio.
    Method: if >60% of answer words appear in the question → contaminated.
    """
    if not answer or not question:
        return False
    q_words = set(re.sub(r"[^a-z0-9\s]", "", question.lower()).split())
    a_words = set(re.sub(r"[^a-z0-9\s]", "", answer.lower()).split())
    if not a_words:
        return False
    return (len(q_words & a_words) / len(a_words)) > 0.60


# ─────────────────────────────────────────────────────────
#  KEYWORD SCREENING  (kept for screening_agent.py import)
# ─────────────────────────────────────────────────────────

_TECH_KW = {
    "python", "java", "javascript", "typescript", "react", "nodejs", "node",
    "fastapi", "django", "flask", "sql", "postgresql", "mysql", "mongodb",
    "redis", "docker", "kubernetes", "aws", "gcp", "azure", "git", "rest",
    "api", "html", "css", "bootstrap", "tailwind", "machine learning",
    "deep learning", "nlp", "tensorflow", "pytorch", "pandas", "numpy",
    "scikit", "llm", "langchain", "crewai", "microservices", "linux",
    "bash", "spring", "selenium", "pytest", "ollama", "vector", "embedding",
    "rag", "openai", "graphql", "kafka", "rabbitmq", "elasticsearch",
    "c++", "c#", ".net", "kotlin", "swift", "ruby", "php", "android",
    "ios", "flutter", "groq", "pydantic", "sqlalchemy",
}


def _kw_extract(text: str) -> set:
    n = re.sub(r"[^a-z0-9#+.\s]", " ", text.lower())
    return {kw for kw in _TECH_KW if kw in n}


def ollama_semantic_score(resume_text: str, jd_text: str) -> Tuple[float, Dict]:
    """Kept for backward compat. Real screening lives in screening_agent.py."""
    if not resume_text or not jd_text:
        return 0.0, {"error": "empty input"}
    try:
        prompt = (
            f"Rate how well this resume matches the job description.\n"
            f"JOB DESCRIPTION:\n{jd_text[:1000]}\n"
            f"RESUME:\n{resume_text[:1000]}\n"
            f"Reply with ONLY a single integer 0-100."
        )
        raw  = _call_groq(prompt, temperature=0.1, max_tokens=10)
        nums = re.findall(r"\b(\d{1,3})\b", raw)
        if nums:
            return min(100.0, max(0.0, float(nums[0]))), {"method": "groq_llm_rating"}
    except Exception as e:
        print(f"[Screening] Groq failed: {e}")
    jd_kw   = _kw_extract(jd_text)
    res_kw  = _kw_extract(resume_text)
    matched = jd_kw & res_kw
    score   = round(
        min(100, (len(matched) / max(1, len(jd_kw))) * 100 + min(10, len(matched) * 1.5)), 2
    )
    return score, {"method": "keyword_overlap"}


def get_embedding(text: str, model: str = "") -> List[float]:
    return []


def cosine_similarity(a: List[float], b: List[float]) -> float:
    return -1.0


# ─────────────────────────────────────────────────────────
#  RESUME PARSER  — runs ONCE on upload
# ─────────────────────────────────────────────────────────

_PARSE_PROMPT = """Read this resume carefully and extract structured information.

RESUME:
{resume_text}

Return ONLY valid JSON with no markdown fences:
{{
  "name":                   "",
  "location":               "",
  "email":                  "",
  "phone":                  "",
  "education": {{
    "degree":               "",
    "field":                "",
    "college":              "",
    "graduation_year":      "",
    "gpa_or_percentage":    ""
  }},
  "total_experience_years": "",
  "current_role":           "",
  "previous_companies":     [],
  "key_skills":             [],
  "certifications":         [],
  "projects": [
    {{
      "name":        "",
      "description": "",
      "tech_used":   [],
      "outcome":     ""
    }}
  ],
  "achievements":              [],
  "career_goal_clues":         "",
  "hobbies_interests":         [],
  "resume_sections_available": []
}}

Use empty string or [] when a field is not found. Return ONLY the JSON object."""


class PersonalInfoAgent:
    """
    Runs once on upload.
    Extracts structured info from resume, generates Section-1 (intro) question.
    """

    # 8 different "tell me about yourself" opener styles — one chosen randomly
    _Q1_VARIANTS = [
        "Ask the candidate to walk you through their background — education, key skills, and what they have been working on recently. Address them by name. Max 2 sentences.",
        "Ask the candidate to give a brief overview of their journey so far — where they started and where they are now. Address them by name. Max 2 sentences.",
        "Ask the candidate to introduce themselves — who they are, what they study or do, and what they are most proud of so far. Address them by name. Max 2 sentences.",
        "Ask the candidate to tell their story so far — the experiences and decisions that brought them to where they are today. Address them by name. Max 2 sentences.",
        "Ask the candidate to give a quick snapshot of themselves — their background, main skills, and what kind of work they genuinely enjoy. Address them by name. Max 2 sentences.",
        "Ask the candidate to describe themselves in a nutshell — academic background, what they have built or worked on, and what drives them. Address them by name. Max 2 sentences.",
        "Ask the candidate to share the parts of their background that best represent who they are professionally. Address them by name. Max 2 sentences.",
        "Ask the candidate what you should know about them before the interview begins — what defines them professionally. Address them by name. Max 2 sentences.",
    ]

    def extract_info(self, resume_text: str) -> Dict:
        print("[PersonalInfoAgent] Parsing resume...")
        try:
            raw  = _call_groq(
                _PARSE_PROMPT.format(resume_text=resume_text[:3500]),
                temperature=0.1, max_tokens=1000,
            )
            info = _parse_json(raw)
            print(
                f"[PersonalInfoAgent] name={info.get('name')} "
                f"skills={info.get('key_skills', [])[:3]} "
                f"projects={len(info.get('projects', []))}"
            )
            return info
        except Exception as e:
            print(f"[PersonalInfoAgent] Parse failed: {e}")
            return {
                "name": "", "location": "", "education": {},
                "total_experience_years": "", "current_role": "",
                "previous_companies": [], "key_skills": [],
                "certifications": [], "projects": [],
                "achievements": [], "career_goal_clues": "",
                "hobbies_interests": [], "resume_sections_available": [],
            }

    def generate_first_question(self, personal_info: Dict,
                                candidate_name: str = "") -> Dict:
        """
        Section 1 — Opening / Introduction.
        Always a 'tell me about yourself' style opener.
        Wording varies each interview (one of 8 variants chosen randomly).
        Personalised with candidate name + one concrete resume detail.
        """
        # Priority: form name → personal_info["form_name"] → LLM-extracted name
        name = (
            candidate_name.strip()
            or personal_info.get("form_name", "").strip()
            or personal_info.get("name", "").strip()
        )
        # Store form_name so later agents can access it
        if candidate_name.strip():
            personal_info["form_name"] = candidate_name.strip()

        greeting = f"Hi {name}," if name else "Hi,"
        print(f"[PersonalInfoAgent] Generating Q1 (Section 1 — Opening) for: {name or '(no name)'}...")

        # Pick one resume detail to weave in for personalisation
        edu      = personal_info.get("education", {})
        projects = personal_info.get("projects", [])
        skills   = personal_info.get("key_skills", [])

        resume_hint = ""
        if isinstance(edu, dict) and edu.get("field"):
            college     = edu.get("college", "")
            resume_hint = f"{edu['field']} at {college}" if college else edu["field"]
        elif projects:
            p           = projects[0]
            resume_hint = p.get("name", "") if isinstance(p, dict) else str(p)
        elif skills:
            resume_hint = ", ".join(str(s) for s in skills[:2])

        instruction = random.choice(self._Q1_VARIANTS)

        prompt = f"""You are a warm, professional interviewer opening a job interview.

CANDIDATE NAME  : {name if name else "the candidate"}
RESUME DETAIL   : {resume_hint if resume_hint else "not available"}
GREETING TO USE : "{greeting}"
INSTRUCTION     : {instruction}

RULES:
- The question MUST be a "tell me about yourself" / introduction style opener.
- Start the question with exactly "{greeting}"
- Weave in the resume detail naturally to make it feel personalised.
- Do NOT ask about just one specific project — keep it broad (who they are overall).
- Phrasing must follow the instruction above exactly.
- Max 2 sentences. Warm and conversational tone.

Return ONLY valid JSON (no markdown):
{{"question": "...", "topic": "introduction", "based_on": "self-introduction"}}"""

        try:
            raw  = _call_groq(prompt, temperature=0.8, max_tokens=200)
            data = _parse_json(raw)
            q    = data.get("question", "").strip()
            if not q or len(q) < 15:
                raise ValueError("Too short")
            print(f"[PersonalInfoAgent] Q1: {q[:80]}...")
            return {
                "question": q,     "type":       "opening",
                "topic":    "introduction",
                "based_on": "self-introduction",
                "phase":    1,     "difficulty": 1,
            }
        except Exception as e:
            print(f"[PersonalInfoAgent] Q1 LLM failed ({e}) — using fallback")
            return self._fallback_q1(personal_info, name)

    def _fallback_q1(self, info: Dict, name: str = "") -> Dict:
        n        = (name or info.get("name", "")).strip()
        greeting = f"Hi {n}," if n else "Hi,"
        options  = [
            f"{greeting} could you start by walking me through your background — your education, the skills you have developed, and what you have been working on recently?",
            f"{greeting} before we dive in, could you introduce yourself — who you are, what you have studied, and what kind of work you enjoy most?",
            f"{greeting} let us start with you — tell me your story so far, the experiences that shaped you, and where you are right now professionally.",
            f"{greeting} could you introduce yourself — your academic background, the projects or roles you have been involved in, and what you are most proud of so far?",
        ]
        return {
            "question": random.choice(options), "type": "opening",
            "topic":    "introduction",
            "based_on": "self-introduction",
            "phase":    1, "difficulty": 1,
        }


# ─────────────────────────────────────────────────────────
#  INTERVIEW PLAN  — built once, cached per session
# ─────────────────────────────────────────────────────────

_PLAN_PROMPT = """You are a senior technical interviewer. Analyze this resume and job description.

JOB DESCRIPTION:
{jd_text}

RESUME:
{resume_text}

Create a comprehensive 7-section interview plan. Return ONLY valid JSON (no markdown):
{{
  "candidate_level": "junior|mid|senior",
  "matched_skills":  [],
  "missing_skills":  [],

  "resume_topics": [
    {{
      "section":      "project|work_experience|skill|certification|achievement",
      "specific_ref": "EXACT name/detail from the resume — must be a real item, not generic",
      "probe_angle":  "what to explore: technical challenge / design decision / outcome / tech used"
    }}
  ],

  "tech_skills": [
    {{
      "skill":         "specific technology or concept from the JD",
      "depth":         "conceptual|applied|architectural",
      "candidate_has": true
    }}
  ],

  "jd_requirements": [
    {{
      "requirement": "a specific role expectation or responsibility from the JD",
      "angle":       "how to probe it — experience with this / expectation alignment / process preference"
    }}
  ],

  "behavioral_scenarios": [
    {{
      "theme":   "teamwork|conflict|leadership|failure|pressure|initiative|communication",
      "context": "relate to the candidate's background or JD role if possible"
    }}
  ],

  "situational_scenarios": [
    {{
      "scenario":    "a realistic what-would-you-do-if situation for this specific role",
      "skill_tested": "decision making|prioritisation|communication|technical judgment"
    }}
  ],

  "gap_skills": [],
  "summary":    ""
}}

STRICT RULES:
- resume_topics   : 3-4 items. specific_ref MUST be an exact project name, company, skill, or cert from the resume.
- tech_skills     : 3-4 items. candidate_has=true only if the resume clearly mentions this skill.
- jd_requirements : 2-3 items. Role-level expectations only — NOT tech skills (those go in tech_skills).
- behavioral_scenarios : exactly 2, with different themes.
- situational_scenarios: exactly 2, realistic for this specific role and industry.
- gap_skills: 2-3 skills the JD clearly requires that the resume clearly lacks.

Return ONLY the JSON object, nothing else."""

_plan_cache: Dict[str, Dict] = {}


def _plan_is_usable(plan: Dict) -> bool:
    """
    A plan is usable only when every section list has at least one entry.
    An empty/failed plan must NEVER be cached — it causes all section routing
    blocks to fall through and fires the closing question after just one answer.
    """
    return (
        len(plan.get("resume_topics",         [])) >= 1
        and len(plan.get("tech_skills",       [])) >= 1
        and len(plan.get("jd_requirements",   [])) >= 1
        and len(plan.get("behavioral_scenarios",  [])) >= 1
        and len(plan.get("situational_scenarios", [])) >= 1
    )


def _build_fallback_plan(resume_text: str, jd_text: str) -> Dict:
    """
    Minimal keyword-derived plan used when the LLM fails all retries.
    Guarantees every section list has at least one entry so section routing
    never falls through to closing after a single question.
    """
    skills = list(_kw_extract(resume_text) | _kw_extract(jd_text))[:6] or ["Python", "problem solving"]

    return {
        "candidate_level": "mid",
        "matched_skills":  skills,
        "missing_skills":  [],
        "resume_topics": [
            {"section": "project",
             "specific_ref": "your most recent project",
             "probe_angle": "technical challenge and outcome"},
            {"section": "skill",
             "specific_ref": skills[0] if skills else "your main skill",
             "probe_angle": "how you applied it in practice"},
        ],
        "tech_skills": [
            {"skill": s, "depth": "applied", "candidate_has": True}
            for s in skills[:3]
        ] or [{"skill": "Python", "depth": "applied", "candidate_has": True}],
        "jd_requirements": [
            {"requirement": "delivering quality work on time",
             "angle": "how they manage deadlines and priorities"},
        ],
        "behavioral_scenarios": [
            {"theme": "teamwork", "context": "working with a team on a challenging project"},
            {"theme": "pressure", "context": "delivering work under a tight deadline"},
        ],
        "situational_scenarios": [
            {"scenario": "You are midway through a project and the requirements change significantly.",
             "skill_tested": "decision making"},
            {"scenario": "A teammate is not contributing and the deadline is approaching.",
             "skill_tested": "communication"},
        ],
        "gap_skills": [],
        "summary": "Fallback plan — LLM plan generation failed after all retries.",
    }


def _build_interview_plan(resume_text: str, jd_text: str, session_key: str) -> Dict:
    global _plan_cache

    # Only return cached plan if it was stored (meaning it passed usability check).
    if session_key and session_key in _plan_cache:
        return _plan_cache[session_key]

    print("[InterviewPlan] Building 7-section plan from resume + JD...")

    # Retry up to 3 times — transient LLM failures are common.
    for attempt in range(1, 4):
        try:
            raw  = _call_groq(
                _PLAN_PROMPT.format(
                    jd_text     = jd_text[:2000],
                    resume_text = resume_text[:2500],
                ),
                temperature=0.15, max_tokens=1400,
            )
            plan = _parse_json(raw)
            print(
                f"[InterviewPlan] attempt={attempt} level={plan.get('candidate_level')} | "
                f"resume_topics={len(plan.get('resume_topics', []))} "
                f"tech_skills={len(plan.get('tech_skills', []))} "
                f"jd_req={len(plan.get('jd_requirements', []))} "
                f"behavioral={len(plan.get('behavioral_scenarios', []))} "
                f"situational={len(plan.get('situational_scenarios', []))}"
            )

            if _plan_is_usable(plan):
                # Cache ONLY a usable plan — never cache empty fallbacks.
                if session_key:
                    _plan_cache[session_key] = plan
                return plan

            print(f"[InterviewPlan] attempt={attempt} — plan unusable, retrying…")

        except Exception as e:
            print(f"[InterviewPlan] attempt={attempt} failed: {e}")

    # All retries exhausted — use keyword-derived fallback.
    print("[InterviewPlan] All retries failed — using keyword-derived fallback plan")
    fallback = _build_fallback_plan(resume_text, jd_text)
    if session_key:
        _plan_cache[session_key] = fallback
    return fallback


# ─────────────────────────────────────────────────────────
#  STOP ALGORITHM
# ─────────────────────────────────────────────────────────

MAX_INTERVIEW_QUESTIONS = 15

SECTION_MINIMUMS = {
    "opening":     1,
    "resume":      2,
    "technical":   2,
    "jd":          1,
    "behavioral":  2,
    "situational": 1,
    "closing":     1,
}


def _core_sections_complete(
    s1: int, s2: int, s3: int, s4: int, s5: int, s6: int,
    plan: Dict,
) -> bool:
    """
    Core interview coverage means Sections 1–6 are fully exhausted.

    For sections 2 and 3 we consider them complete only when ALL planned
    topics / skills have been asked (not just the minimum count).  This
    prevents the engine from jumping to closing while there are still
    unused resume topics or tech skills in the plan.
    """
    # Section 1 — always exactly 1
    if s1 < SECTION_MINIMUMS["opening"]:
        return False

    # Section 2 — must have reached the minimum AND exhausted all planned topics
    planned_resume  = len(plan.get("resume_topics", []))
    s2_target       = max(SECTION_MINIMUMS["resume"], planned_resume)
    if s2 < s2_target:
        return False

    # Section 3 — must have reached the minimum AND exhausted all planned skills
    planned_tech    = len(plan.get("tech_skills", []))
    s3_target       = max(SECTION_MINIMUMS["technical"], planned_tech)
    if s3 < s3_target:
        return False

    # Section 4 — minimum 1; exhaust all planned JD requirements
    planned_jd      = len(plan.get("jd_requirements", []))
    s4_target       = max(SECTION_MINIMUMS["jd"], planned_jd)
    if s4 < s4_target:
        return False

    # Section 5 — minimum 2; exhaust all planned behavioral scenarios
    planned_beh     = len(plan.get("behavioral_scenarios", []))
    s5_target       = max(SECTION_MINIMUMS["behavioral"], planned_beh)
    if s5 < s5_target:
        return False

    # Section 6 — minimum 1; exhaust all planned situational scenarios
    planned_sit     = len(plan.get("situational_scenarios", []))
    s6_target       = max(SECTION_MINIMUMS["situational"], planned_sit)
    if s6 < s6_target:
        return False

    return True


def _all_sections_complete(
    s1: int, s2: int, s3: int, s4: int, s5: int, s6: int, s7: int,
    plan: Dict,
) -> bool:
    """
    Full interview completion — Sections 1–7 all exhausted.
    """
    return (
        _core_sections_complete(s1, s2, s3, s4, s5, s6, plan)
        and s7 >= SECTION_MINIMUMS["closing"]
    )


def _compute_stop_decision(
    scores: List[int], q_count: int,
    s1: int, s2: int, s3: int, s4: int, s5: int, s6: int, s7: int,
    plan: Dict,
) -> Tuple[bool, str]:
    """
    Coverage-first stop decision.

    KEY RULE: The interview NEVER stops because of a high average score or
    because of weak scores.  It stops only after every planned topic in
    every section has been covered and the closing question has been asked.

    The hard cap (15 questions) allows an early exit ONLY when Sections 1–6
    are already exhausted; in that case the next action is to ask closing,
    not to end immediately.
    """

    # Normal completion — all 7 sections fully exhausted.
    if _all_sections_complete(s1, s2, s3, s4, s5, s6, s7, plan):
        return True, "all_7_sections_complete"

    # Opening must always happen first.
    if s1 < SECTION_MINIMUMS["opening"]:
        return False, "section1_not_done"

    # Hard cap reached — only allow stopping if Sections 1–6 are done.
    if q_count >= MAX_INTERVIEW_QUESTIONS:
        if _core_sections_complete(s1, s2, s3, s4, s5, s6, plan) and s7 == 0:
            # Signal caller to ask closing then end.
            return True, "hard_cap_reached_closing_required"

        # Some sections still pending — log but do not stop.
        return False, f"hard_cap_reached_sections_still_pending"

    # Diagnostic only — weak streak is tracked but NEVER stops the interview.
    weak_streak = 0
    for s in reversed(scores or []):
        if isinstance(s, (int, float)) and s < 4:
            weak_streak += 1
        else:
            break

    if weak_streak >= 3:
        return False, f"weak_streak_{weak_streak}_continue_until_all_sections_done"

    return False, "continue"


# ─────────────────────────────────────────────────────────
#  SECTION PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────

# ── Section 2 — Resume-Based ──────────────────────────────────────────────────
_S2_PROMPT = """You are a sharp interviewer probing a specific part of the candidate's resume.

RESUME EXCERPT:
{resume_text}

TOPIC TO PROBE:
  Section        : {section}
  Specific Item  : {specific_ref}
  Probe Angle    : {probe_angle}
  Candidate Level: {level}

QUESTIONS ALREADY ASKED — DO NOT REPEAT OR REPHRASE ANY:
{asked}

CANDIDATE'S RECENT ANSWERS:
{recent}

Generate ONE specific question about "{specific_ref}".

Requirements:
- Reference the exact item named above — not a generic resume question.
- Explore the probe angle (e.g. challenge faced, design decision, outcome, tech stack chosen).
- Difficulty: junior=conceptual understanding, mid=applied real example, senior=architectural decision.
- Must be answerable from the candidate's own experience — not hypothetical.
- Cannot be similar to anything already asked.
- 1-2 sentences.

Return ONLY valid JSON:
{{"question": "...", "topic": "{section}", "based_on": "{specific_ref}", "difficulty": 3}}"""


# ── Section 3 — Technical ─────────────────────────────────────────────────────
_S3_PROMPT = """You are a technical interviewer testing depth of knowledge.

CANDIDATE LEVEL        : {level}
TECHNICAL SKILL        : {skill}
CANDIDATE HAS IN RESUME: {candidate_has}
DEPTH EXPECTED         : {depth}

QUESTIONS ALREADY ASKED — DO NOT REPEAT OR REPHRASE ANY:
{asked}

Generate ONE technical question about "{skill}".

Rules:
- candidate_has=true  → ask an APPLIED or ARCHITECTURAL question (how they used it, problems they solved with it, trade-offs).
- candidate_has=false → ask a CONCEPTUAL AWARENESS question (what it is, when you would use it, basic difference vs alternatives).
- Difficulty by level: junior=what/why, mid=how/applied, senior=trade-offs/architecture.
- Do NOT ask a question already asked.
- Do NOT ask "what is {skill}" to a mid/senior — too basic.
- 1-2 sentences, clear and direct.

Return ONLY valid JSON:
{{"question": "...", "topic": "technical — {skill}", "based_on": "tech:{skill}", "difficulty": 3}}"""


# ── Section 4 — JD-Based ─────────────────────────────────────────────────────
_S4_PROMPT = """You are an interviewer checking alignment between the candidate and the job requirements.

JOB DESCRIPTION EXCERPT:
{jd_text}

REQUIREMENT TO PROBE:
  Requirement : {requirement}
  Probe Angle : {angle}

CANDIDATE BACKGROUND:
{candidate_summary}

QUESTIONS ALREADY ASKED — DO NOT REPEAT OR REPHRASE ANY:
{asked}

Generate ONE question that:
- Probes how well the candidate aligns with or understands this specific JD requirement.
- Uses the probe angle (experience with this / preference / how they approach it).
- Is specific to THIS role — not a generic "why do you want to work here" question.
- Cannot repeat or rephrase anything already asked.
- 1-2 sentences.

Return ONLY valid JSON:
{{"question": "...", "topic": "JD — {requirement}", "based_on": "JD:{requirement}"}}"""


# ── Section 5 — Behavioral / HR ──────────────────────────────────────────────
_S5_PROMPT = """You are an HR interviewer using the STAR behavioral method.

BEHAVIORAL THEME : {theme}
CONTEXT HINT     : {context}
CANDIDATE NAME   : {name}
CANDIDATE LEVEL  : {level}

QUESTIONS ALREADY ASKED — DO NOT REPEAT OR REPHRASE ANY:
{asked}

Generate ONE behavioral question using STAR structure (Situation, Task, Action, Result).

Rules:
- Open with "Tell me about a time when..." OR "Can you describe a situation where..." or similar STAR opener.
- The question must explore the theme: {theme}.
- If the context_hint mentions something from the resume, weave it in naturally.
- Address the candidate by name ({name}) if it feels natural.
- Must NOT repeat or rephrase any already-asked question.
- 1-2 sentences.

Return ONLY valid JSON:
{{"question": "...", "topic": "behavioral — {theme}", "based_on": "behavioral:{theme}"}}"""


# ── Section 6 — Situational ───────────────────────────────────────────────────
_S6_PROMPT = """You are an interviewer presenting a realistic work scenario.

SCENARIO     : {scenario}
SKILL TESTED : {skill_tested}
CANDIDATE LEVEL: {level}

QUESTIONS ALREADY ASKED — DO NOT REPEAT OR REPHRASE ANY:
{asked}

Generate ONE situational question based on the scenario above.

Rules:
- Frame it as "What would you do if..." OR "Imagine you are..." OR "How would you handle..."
- The situation must be realistic and directly relevant to the role described.
- The question should reveal the candidate's judgment, process, or communication approach.
- Must NOT repeat or rephrase anything already asked.
- 1-2 sentences, clear and specific.

Return ONLY valid JSON:
{{"question": "...", "topic": "situational — {skill_tested}", "based_on": "situational:{skill_tested}"}}"""


# ── Section 7 — Closing (random pick from 5 variants) ────────────────────────
_CLOSING_VARIANTS = [
    "We are nearing the end of our interview — do you have any questions for me about the role, the team, or what a typical day looks like here?",
    "Before we wrap up, is there anything you would like to ask about the position, the company culture, or the next steps in the process?",
    "We have covered a lot of ground today — is there anything you would like to know more about, either about the role itself or the team you would be joining?",
    "As we close, do you have any questions for me — about the work, the team, growth opportunities, or anything else on your mind?",
    "Before I let you go, I would love to hear any questions you have — about the day-to-day of the role, the team dynamics, or what success looks like in this position.",
]


# ── Follow-up prompts ─────────────────────────────────────────────────────────
_FOLLOWUP_PROMPT = """You are a sharp interviewer. The candidate gave a weak or vague answer.

QUESTION ASKED : {question}
CANDIDATE SAID : {answer}
SCORE          : {score}/10
WHAT WAS WEAK  : {weakness}

Generate ONE targeted follow-up that:
- Directly addresses what was missing or unclear in their answer.
- References EXACTLY what they said (or failed to say).
- Asks for a concrete example, specific number, or deeper explanation.
- Is NOT generic — not "can you elaborate?" or "tell me more".
- 1-2 sentences.

Return ONLY valid JSON:
{{"question": "...", "topic": "follow-up", "based_on": "previous answer"}}"""


_CONTAMINATION_PROMPT = """You are a friendly interviewer. The candidate may have had a mic issue and their
submitted text appears to be the question itself, not their actual answer.

ORIGINAL QUESTION: {question}

Write one short, warm sentence asking them to share their answer in their own words.

Return ONLY valid JSON:
{{"question": "...", "topic": "clarification", "based_on": "technical issue"}}"""


# ─────────────────────────────────────────────────────────
#  NEXT QUESTION AGENT  — 7-section orchestrator
# ─────────────────────────────────────────────────────────

class NextQuestionAgent:
    """
    Called after every submitted answer.
    Routes through all 7 sections in fixed order.

    Sections 2–6 are driven entirely by the interview plan — every planned
    topic / skill / requirement / scenario is asked before the engine moves
    to the next section.  Score does NOT influence progression.

    Returns: (next_question_dict | None, plan_dict | None)
    None as first value = interview is complete.
    """

    # ── Public API used by applicant.py ───────────────────────────────────────

    def should_end(self, scores: List[int], answered_count: int) -> Tuple[bool, str]:
        """
        Lightweight pre-check used by applicant.py before calling generate_next.

        This method must not stop the interview because it does not know
        section coverage. The real stop decision happens inside generate_next()
        using section counters.
        """
        return False, "section_aware_stop_check_required"


    def generate_next(
        self,
        resume_text:    str,
        jd_text:        str,
        personal_info:  Dict,
        history:        List[Dict],   # {question, answer, score, type, topic, based_on, explanation}
        scores:         List[int],
        followup_count: int,
        session_key:    str = "",
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Main routing method.
        Returns (question_dict | None, plan_dict | None).
        """
        q_count = len(history)

        # ── Build / fetch the interview plan ──────────────────────────────────
        plan  = _build_interview_plan(resume_text, jd_text, session_key)
        level = plan.get("candidate_level", "mid")

        # ── Count completed questions per section ──────────────────────────────
        s1 = sum(1 for h in history if h.get("type") == "opening")
        s2 = sum(1 for h in history if h.get("type") == "resume")
        s3 = sum(1 for h in history if h.get("type") == "technical")
        s4 = sum(1 for h in history if h.get("type") == "jd")
        s5 = sum(1 for h in history if h.get("type") == "behavioral")
        s6 = sum(1 for h in history if h.get("type") == "situational")
        s7 = sum(1 for h in history if h.get("type") == "closing")

        # ── Compute per-section targets from the plan ─────────────────────────
        # These are the number of questions we WANT to ask for each section.
        # Minimum floors are enforced; plan length is the ceiling.
        s2_target = max(SECTION_MINIMUMS["resume"],      len(plan.get("resume_topics", [])))
        s3_target = max(SECTION_MINIMUMS["technical"],   len(plan.get("tech_skills", [])))
        s4_target = max(SECTION_MINIMUMS["jd"],          len(plan.get("jd_requirements", [])))
        s5_target = max(SECTION_MINIMUMS["behavioral"],  len(plan.get("behavioral_scenarios", [])))
        s6_target = max(SECTION_MINIMUMS["situational"], len(plan.get("situational_scenarios", [])))

        print(
            f"[NextQ] Progress → "
            f"S1:{s1}/1 S2:{s2}/{s2_target} S3:{s3}/{s3_target} "
            f"S4:{s4}/{s4_target} S5:{s5}/{s5_target} S6:{s6}/{s6_target} "
            f"S7:{s7}/1 | followups:{followup_count}"
        )

        # ── Full blacklist of already-asked questions (sent to every LLM call)
        asked = (
            "\n".join(f"- {h['question']}" for h in history if h.get("question"))
            or "None yet."
        )

        # ── Info about the last exchange ───────────────────────────────────────
        last          = history[-1] if history else {}
        last_q        = last.get("question", "")
        last_a        = last.get("answer", "")
        last_score    = last.get("score", 10)
        last_type     = last.get("type", "opening")
        last_weakness = last.get("explanation", "")

        # ─────────────────────────────────────────────────────────────────────
        # PRIORITY 1: STT contamination check
        # ─────────────────────────────────────────────────────────────────────
        if q_count >= 1 and _is_contaminated(last_q, last_a):
            print("[NextQ] STT contamination detected — injecting re-answer request")
            return self._gen_contamination_followup(last_q), plan

        # ─────────────────────────────────────────────────────────────────────
        # PRIORITY 2: Follow-up injection
        #   Only when the last answer was weak or too short.
        #   Never after closing. Never when follow-up budget (3) is exhausted.
        #   NOTE: follow-ups do NOT advance the section counter — they are
        #   injected in-place so that the section target is still met.
        # ─────────────────────────────────────────────────────────────────────
        word_count = len(last_a.strip().split())
        if (
            followup_count < 3
            and q_count >= 1
            and last_type not in ("followup", "closing")
            and (last_score < 5 or word_count < 20)
        ):
            print(f"[NextQ] Weak answer (score={last_score}, words={word_count}) — injecting follow-up")
            return self._gen_followup(last_q, last_a, last_score, last_weakness), plan

        # ─────────────────────────────────────────────────────────────────────
        # PRIORITY 3: Stop algorithm
        #   Passes the plan so section targets are plan-aware.
        # ─────────────────────────────────────────────────────────────────────
        stop, reason = _compute_stop_decision(
            scores, q_count, s1, s2, s3, s4, s5, s6, s7, plan
        )

        if stop:
            print(f"[NextQ] Stop condition met — {reason}")

            # If Sections 1–6 are exhausted but closing has not been asked,
            # ask the closing question before ending the interview.
            if s7 == 0:
                print("[NextQ] Injecting closing question before ending")
                return self._gen_closing(), plan

            # Closing already answered — interview is truly complete.
            return None, plan

        # ═════════════════════════════════════════════════════════════════════
        #  SECTION ROUTING  — strict fixed order: 1 → 2 → 3 → 4 → 5 → 6 → 7
        #
        #  Each section block checks whether its PLAN-AWARE target has been
        #  reached.  If not, it tries to find an unused planned topic.
        #  Only when ALL topics in a section are exhausted does routing fall
        #  through to the next section.
        # ═════════════════════════════════════════════════════════════════════

        # ── Section 1: Opening ────────────────────────────────────────────────
        # Q1 is always generated by PersonalInfoAgent at upload time.
        # This block is a safety net only.
        if s1 == 0:
            print("[NextQ] S1 safety-net — generating opening question")
            return self._gen_opening_fallback(personal_info), plan

        # ── Section 2: Resume-Based ───────────────────────────────────────────
        # Continue until all planned resume topics have been asked.
        if s2 < s2_target:
            asked_refs = {
                h.get("based_on", "").lower().strip()
                for h in history
                if h.get("type") == "resume" and h.get("based_on", "").strip()
            }
            next_topic = next(
                (t for t in plan.get("resume_topics", [])
                 if t.get("specific_ref", "").lower().strip() not in asked_refs),
                None,
            )
            if next_topic:
                print(f"[NextQ] S2 Resume — probing: {next_topic.get('specific_ref')} ({s2+1}/{s2_target})")
                return self._gen_resume_q(
                    resume_text, next_topic, level, asked, history[-3:]
                ), plan
            # All planned topics asked but count still below target — fall through.
            print(f"[NextQ] S2 — no more unused topics (asked {s2}/{s2_target}), moving on")

        # ── Section 3: Technical ──────────────────────────────────────────────
        # Continue until all planned tech skills have been asked.
        if s3 < s3_target:
            asked_tech = {
                h.get("based_on", "").lower().strip()
                for h in history
                if h.get("type") == "technical" and h.get("based_on", "").strip()
            }
            next_skill = next(
                (t for t in plan.get("tech_skills", [])
                 if f"tech:{t.get('skill','').lower().strip()}" not in asked_tech),
                None,
            )
            if next_skill:
                print(f"[NextQ] S3 Technical — skill: {next_skill.get('skill')} ({s3+1}/{s3_target})")
                return self._gen_technical_q(next_skill, level, asked), plan
            print(f"[NextQ] S3 — no more unused skills (asked {s3}/{s3_target}), moving on")

        # ── Section 4: JD-Based ───────────────────────────────────────────────
        # Continue until all planned JD requirements have been asked.
        if s4 < s4_target:
            asked_jd = {
                h.get("based_on", "").lower().strip()
                for h in history
                if h.get("type") == "jd" and h.get("based_on", "").strip()
            }
            next_req = next(
                (r for r in plan.get("jd_requirements", [])
                 if f"jd:{r.get('requirement','').lower().strip()}" not in asked_jd),
                None,
            )
            if next_req:
                print(f"[NextQ] S4 JD — requirement: {next_req.get('requirement')} ({s4+1}/{s4_target})")
                return self._gen_jd_q(next_req, jd_text, personal_info, asked), plan
            print(f"[NextQ] S4 — no more unused requirements (asked {s4}/{s4_target}), moving on")

        # ── Section 5: Behavioral / HR ────────────────────────────────────────
        # Continue until all planned behavioral scenarios have been asked.
        if s5 < s5_target:
            asked_beh = {
                h.get("based_on", "").lower().strip()
                for h in history
                if h.get("type") == "behavioral" and h.get("based_on", "").strip()
            }
            next_beh = next(
                (b for b in plan.get("behavioral_scenarios", [])
                 if f"behavioral:{b.get('theme','').lower().strip()}" not in asked_beh),
                None,
            )
            if next_beh:
                print(f"[NextQ] S5 Behavioral — theme: {next_beh.get('theme')} ({s5+1}/{s5_target})")
                return self._gen_behavioral_q(
                    next_beh, level, personal_info, asked
                ), plan
            print(f"[NextQ] S5 — no more unused scenarios (asked {s5}/{s5_target}), moving on")

        # ── Section 6: Situational ────────────────────────────────────────────
        # Continue until all planned situational scenarios have been asked.
        if s6 < s6_target:
            asked_sit = {
                h.get("based_on", "").lower().strip()
                for h in history
                if h.get("type") == "situational" and h.get("based_on", "").strip()
            }
            next_sit = next(
                (s for s in plan.get("situational_scenarios", [])
                 if f"situational:{s.get('skill_tested','').lower().strip()}" not in asked_sit),
                None,
            )
            if next_sit:
                print(f"[NextQ] S6 Situational — skill: {next_sit.get('skill_tested')} ({s6+1}/{s6_target})")
                return self._gen_situational_q(next_sit, level, asked), plan
            print(f"[NextQ] S6 — no more unused scenarios (asked {s6}/{s6_target}), moving on")

        # ── Section 7: Closing (always fires exactly once, always last) ────────
        if s7 == 0:
            print("[NextQ] S7 — Closing question")
            return self._gen_closing(), plan

        # All sections and all planned topics exhausted.
        print("[NextQ] All 7 sections + all planned topics complete — interview over")
        return None, plan

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTION GENERATOR METHODS
    # ═════════════════════════════════════════════════════════════════════════

    # ── Section 1 safety-net fallback ─────────────────────────────────────────
    def _gen_opening_fallback(self, personal_info: Dict) -> Dict:
        name     = (
            personal_info.get("form_name", "").strip()
            or personal_info.get("name", "").strip()
        )
        greeting = f"Hi {name}," if name else "Hi,"
        options  = [
            f"{greeting} could you start by walking me through your background — your education, the skills you have developed, and what you have been working on recently?",
            f"{greeting} before we dive in, could you introduce yourself — who you are, what you have studied, and what kind of work you enjoy most?",
        ]
        return {
            "question": random.choice(options), "type": "opening",
            "topic":    "introduction", "based_on": "self-introduction",
            "phase":    1, "difficulty": 1,
        }

    # ── Section 2: Resume-Based ───────────────────────────────────────────────
    def _gen_resume_q(
        self,
        resume_text:    str,
        topic:          Dict,
        level:          str,
        asked:          str,
        recent_history: List[Dict],
    ) -> Dict:
        specific_ref = topic.get("specific_ref", "")
        recent_text  = "\n".join(
            f"Q: {h['question']}\nA: {h.get('answer','')[:120]}"
            for h in recent_history
        ) or "No prior answers."

        try:
            raw  = _call_groq(
                _S2_PROMPT.format(
                    resume_text  = resume_text[:2000],
                    section      = topic.get("section", "experience"),
                    specific_ref = specific_ref,
                    probe_angle  = topic.get("probe_angle", "key challenge"),
                    level        = level,
                    asked        = asked,
                    recent       = recent_text,
                ),
                temperature=0.4, max_tokens=300,
            )
            data = _parse_json(raw)
            q    = data.get("question", "").strip()
            if not q or len(q) < 10:
                raise ValueError("Empty question")
            return {
                "question":   q,
                "type":       "resume",
                "topic":      topic.get("section", "resume"),
                "based_on":   specific_ref,
                "phase":      2,
                "difficulty": int(data.get("difficulty", 3)),
            }
        except Exception as e:
            print(f"[NextQ] S2 failed: {e}")
            ref = specific_ref or "your experience"
            return {
                "question":   f"Can you walk me through {ref} in detail — what was the biggest challenge and how did you overcome it?",
                "type":       "resume",
                "topic":      topic.get("section", "resume"),
                "based_on":   specific_ref,
                "phase":      2, "difficulty": 3,
            }

    # ── Section 3: Technical ─────────────────────────────────────────────────
    def _gen_technical_q(
        self,
        skill_entry: Dict,
        level:       str,
        asked:       str,
    ) -> Dict:
        skill         = skill_entry.get("skill", "")
        candidate_has = skill_entry.get("candidate_has", True)
        depth         = skill_entry.get("depth", "applied")

        try:
            raw  = _call_groq(
                _S3_PROMPT.format(
                    level         = level,
                    skill         = skill,
                    candidate_has = candidate_has,
                    depth         = depth,
                    asked         = asked,
                ),
                temperature=0.4, max_tokens=280,
            )
            data = _parse_json(raw)
            q    = data.get("question", "").strip()
            if not q or len(q) < 10:
                raise ValueError("Empty question")
            return {
                "question":   q,
                "type":       "technical",
                "topic":      f"technical — {skill}",
                "based_on":   f"tech:{skill}",
                "phase":      3,
                "difficulty": int(data.get("difficulty", 3)),
            }
        except Exception as e:
            print(f"[NextQ] S3 failed: {e}")
            if candidate_has:
                q = f"Can you walk me through a project where you used {skill} — what did you build and what was the hardest problem you solved with it?"
            else:
                q = f"How familiar are you with {skill}? Could you explain what it is and describe a scenario where you would use it?"
            return {
                "question":   q,
                "type":       "technical",
                "topic":      f"technical — {skill}",
                "based_on":   f"tech:{skill}",
                "phase":      3, "difficulty": 3,
            }

    # ── Section 4: JD-Based ──────────────────────────────────────────────────
    def _gen_jd_q(
        self,
        req_entry:     Dict,
        jd_text:       str,
        personal_info: Dict,
        asked:         str,
    ) -> Dict:
        requirement = req_entry.get("requirement", "")
        angle       = req_entry.get("angle", "")

        edu      = personal_info.get("education", {})
        skills   = personal_info.get("key_skills", [])
        projects = personal_info.get("projects", [])
        candidate_summary = (
            f"Education: {edu.get('field','')} at {edu.get('college','')}. "
            f"Skills: {', '.join(str(s) for s in skills[:4])}. "
            f"Projects: {', '.join(p.get('name','') if isinstance(p,dict) else str(p) for p in projects[:2])}."
        )

        try:
            raw  = _call_groq(
                _S4_PROMPT.format(
                    jd_text           = jd_text[:800],
                    requirement       = requirement,
                    angle             = angle,
                    candidate_summary = candidate_summary[:400],
                    asked             = asked,
                ),
                temperature=0.45, max_tokens=280,
            )
            data = _parse_json(raw)
            q    = data.get("question", "").strip()
            if not q or len(q) < 10:
                raise ValueError("Empty question")
            return {
                "question":   q,
                "type":       "jd",
                "topic":      f"JD — {requirement}",
                "based_on":   f"JD:{requirement}",
                "phase":      4, "difficulty": 3,
            }
        except Exception as e:
            print(f"[NextQ] S4 failed: {e}")
            return {
                "question":   f"The role mentions {requirement} — how does that align with your experience or working style so far?",
                "type":       "jd",
                "topic":      f"JD — {requirement}",
                "based_on":   f"JD:{requirement}",
                "phase":      4, "difficulty": 2,
            }

    # ── Section 5: Behavioral / HR ───────────────────────────────────────────
    def _gen_behavioral_q(
        self,
        scenario:      Dict,
        level:         str,
        personal_info: Dict,
        asked:         str,
    ) -> Dict:
        theme   = scenario.get("theme", "teamwork")
        context = scenario.get("context", "")
        name    = (
            personal_info.get("form_name", "").strip()
            or personal_info.get("name", "").strip()
        )

        try:
            raw  = _call_groq(
                _S5_PROMPT.format(
                    theme   = theme,
                    context = context,
                    name    = name or "the candidate",
                    level   = level,
                    asked   = asked,
                ),
                temperature=0.5, max_tokens=280,
            )
            data = _parse_json(raw)
            q    = data.get("question", "").strip()
            if not q or len(q) < 10:
                raise ValueError("Empty question")
            return {
                "question":   q,
                "type":       "behavioral",
                "topic":      f"behavioral — {theme}",
                "based_on":   f"behavioral:{theme}",
                "phase":      5, "difficulty": 3,
            }
        except Exception as e:
            print(f"[NextQ] S5 failed: {e}")
            fallbacks = {
                "teamwork":      "Tell me about a time you worked in a team on a challenging project — what was your role and how did you handle any disagreements within the team?",
                "conflict":      "Can you describe a situation where you disagreed with a teammate or instructor — how did you handle it and what was the outcome?",
                "leadership":    "Tell me about a time you took the lead on something, even without being formally asked — what drove you to step up and what happened?",
                "failure":       "Can you share an experience where something did not go as planned — what went wrong, how did you respond, and what did you take from it?",
                "pressure":      "Tell me about a time you had to deliver something under tight deadline pressure — how did you prioritise and what was the result?",
                "initiative":    "Can you describe a time you went beyond what was expected of you in a project, internship, or course — what was the impact?",
                "communication": "Tell me about a time you had to explain something technical to someone non-technical — how did you approach it?",
            }
            q = fallbacks.get(theme, f"Tell me about a time that tested your {theme} — what was the situation and how did you handle it?")
            return {
                "question":   q,
                "type":       "behavioral",
                "topic":      f"behavioral — {theme}",
                "based_on":   f"behavioral:{theme}",
                "phase":      5, "difficulty": 3,
            }

    # ── Section 6: Situational ───────────────────────────────────────────────
    def _gen_situational_q(
        self,
        scenario: Dict,
        level:    str,
        asked:    str,
    ) -> Dict:
        situation    = scenario.get("scenario", "")
        skill_tested = scenario.get("skill_tested", "judgment")

        try:
            raw  = _call_groq(
                _S6_PROMPT.format(
                    scenario     = situation,
                    skill_tested = skill_tested,
                    level        = level,
                    asked        = asked,
                ),
                temperature=0.5, max_tokens=280,
            )
            data = _parse_json(raw)
            q    = data.get("question", "").strip()
            if not q or len(q) < 10:
                raise ValueError("Empty question")
            return {
                "question":   q,
                "type":       "situational",
                "topic":      f"situational — {skill_tested}",
                "based_on":   f"situational:{skill_tested}",
                "phase":      6, "difficulty": 3,
            }
        except Exception as e:
            print(f"[NextQ] S6 failed: {e}")
            fallbacks = {
                "decision making":    "Imagine you are midway through a project and realise the approach you chose will not meet the deadline — what would you do?",
                "prioritisation":     "What would you do if you had three urgent tasks due at the same time and could only complete two of them fully before the deadline?",
                "communication":      "Imagine a stakeholder is unhappy with your work but you believe your approach is correct — how would you handle that conversation?",
                "technical judgment": "How would you approach debugging a critical bug in production that you have never encountered before, with no documentation available?",
            }
            q = fallbacks.get(
                skill_tested.lower(),
                f"How would you handle a situation where you had to demonstrate {skill_tested} under pressure with limited time and information?",
            )
            return {
                "question":   q,
                "type":       "situational",
                "topic":      f"situational — {skill_tested}",
                "based_on":   f"situational:{skill_tested}",
                "phase":      6, "difficulty": 3,
            }

    # ── Section 7: Closing ───────────────────────────────────────────────────
    def _gen_closing(self) -> Dict:
        """
        Always fires as the last question before the interview ends.
        """
        return {
            "question":   random.choice(_CLOSING_VARIANTS),
            "type":       "closing",
            "topic":      "candidate questions",
            "based_on":   "closing",
            "phase":      7, "difficulty": 1,
        }

    # ── Follow-up generators ─────────────────────────────────────────────────
    def _gen_followup(
        self, question: str, answer: str, score: int, weakness: str
    ) -> Dict:
        try:
            raw  = _call_groq(
                _FOLLOWUP_PROMPT.format(
                    question = question,
                    answer   = answer[:400],
                    score    = score,
                    weakness = weakness or "answer was too brief or vague",
                ),
                temperature=0.4, max_tokens=250,
            )
            data = _parse_json(raw)
            q    = data.get("question", "").strip()
            if not q or len(q) < 10:
                raise ValueError("Empty question")
            return {
                "question":   q,
                "type":       "followup",
                "topic":      "follow-up",
                "based_on":   "previous answer",
                "phase":      0, "difficulty": 3,
            }
        except Exception as e:
            print(f"[NextQ] Follow-up failed: {e}")
            return {
                "question":   "Could you give me a concrete, specific example that demonstrates exactly what you just described?",
                "type":       "followup",
                "topic":      "clarification",
                "based_on":   "previous answer",
                "phase":      0, "difficulty": 3,
            }

    def _gen_contamination_followup(self, original_question: str) -> Dict:
        try:
            raw  = _call_groq(
                _CONTAMINATION_PROMPT.format(question=original_question),
                temperature=0.3, max_tokens=120,
            )
            data = _parse_json(raw)
            q    = data.get("question", "").strip()
            if not q or len(q) < 10:
                raise ValueError("Empty question")
        except Exception:
            q = "It looks like there may have been a technical issue — could you please share your answer to that question in your own words?"
        return {
            "question":   q,
            "type":       "followup",
            "topic":      "clarification",
            "based_on":   "technical issue — re-answer requested",
            "phase":      0, "difficulty": 1,
        }


# ─────────────────────────────────────────────────────────
#  ANSWER EVALUATION AGENT
# ─────────────────────────────────────────────────────────

_EVAL_WEIGHTS = {
    "opening":     {"correctness": 0.05, "depth": 0.20, "clarity": 0.40, "practical": 0.25, "completeness": 0.10},
    "resume":      {"correctness": 0.25, "depth": 0.30, "clarity": 0.20, "practical": 0.20, "completeness": 0.05},
    "technical":   {"correctness": 0.40, "depth": 0.30, "clarity": 0.15, "practical": 0.10, "completeness": 0.05},
    "jd":          {"correctness": 0.15, "depth": 0.20, "clarity": 0.30, "practical": 0.25, "completeness": 0.10},
    "behavioral":  {"correctness": 0.10, "depth": 0.25, "clarity": 0.25, "practical": 0.35, "completeness": 0.05},
    "situational": {"correctness": 0.15, "depth": 0.25, "clarity": 0.25, "practical": 0.30, "completeness": 0.05},
    "closing":     {"correctness": 0.05, "depth": 0.15, "clarity": 0.40, "practical": 0.30, "completeness": 0.10},
    "followup":    {"correctness": 0.30, "depth": 0.35, "clarity": 0.20, "practical": 0.10, "completeness": 0.05},
    "personal":    {"correctness": 0.10, "depth": 0.20, "clarity": 0.35, "practical": 0.25, "completeness": 0.10},
    "gap":         {"correctness": 0.20, "depth": 0.25, "clarity": 0.25, "practical": 0.25, "completeness": 0.05},
    "experience":  {"correctness": 0.20, "depth": 0.25, "clarity": 0.20, "practical": 0.30, "completeness": 0.05},
}

_SCORE_PROMPT = """You are a senior interviewer evaluating a candidate answer.

QUESTION TYPE   : {q_type}
QUESTION        : {question}
CANDIDATE ANSWER: {answer}

Rate each dimension from 1-10:
Correctness  : factual accuracy and relevance to the question
Depth        : thoroughness and level of detail
Clarity      : how clear and well-structured the answer is
Practical    : real-world examples or application shown
Completeness : how fully the question was addressed

Output ONLY these 5 lines, nothing else:
Correctness: X
Depth: X
Clarity: X
Practical: X
Completeness: X"""

_INSIGHT_PROMPT = """You are a senior interviewer reviewing a candidate answer.

QUESTION : {question}
ANSWER   : {answer}
SCORES   : correctness={correctness}, depth={depth}, clarity={clarity}

Write exactly ONE specific sentence for each:
Strength: what the candidate did well in this answer
Weakness: what was missing or could be stronger"""


class AnswerEvaluationAgent:

    def evaluate(self, question: str, answer: str,
                 q_type: str = "resume", topic: str = "") -> Dict:

        if _is_contaminated(question, answer):
            print("[EvalAgent] Contaminated answer detected — scoring 1/10")
            return {
                "correctness": 1, "depth": 1, "clarity": 1,
                "practical": 1, "completeness": 1,
                "final_score": 1,
                "explanation": "Answer appeared to echo the question — possible microphone issue.",
                "strength": "", "weakness": "Answer did not address the question.",
            }

        print(f"[EvalAgent] Evaluating type={q_type} | words={len(answer.split())}")
        dims     = self._score_dimensions(question, answer, q_type)
        insights = self._get_insights(question, answer, dims)
        weights  = _EVAL_WEIGHTS.get(q_type, _EVAL_WEIGHTS["resume"])
        raw_s    = sum(dims[k] * weights[k] for k in weights)
        final    = max(1, min(10, round(raw_s)))
        explanation = insights.get("strength", "") or f"Score {final}/10"
        print(
            f"[EvalAgent] Final={final} | "
            f"Corr={dims['correctness']} Depth={dims['depth']} "
            f"Clarity={dims['clarity']} Prac={dims['practical']} Comp={dims['completeness']}"
        )
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

    def _score_dimensions(self, question: str, answer: str, q_type: str) -> Dict:
        defaults = {"correctness": 5, "depth": 5, "clarity": 5,
                    "practical": 5, "completeness": 5}
        try:
            raw  = _call_groq(
                _SCORE_PROMPT.format(
                    q_type   = q_type,
                    question = question[:300],
                    answer   = answer[:500],
                ),
                temperature=0.2, max_tokens=80,
            )
            dims = dict(defaults)
            for line in raw.split("\n"):
                line = line.strip()
                for key in defaults:
                    if line.lower().startswith(key + ":"):
                        m = re.search(r"(\d+)", line)
                        if m:
                            dims[key] = max(1, min(10, int(m.group(1))))
            return dims
        except Exception as e:
            print(f"[EvalAgent] Score call failed: {e}")
            base = min(7, max(3, len(answer.strip().split()) // 20))
            return {k: base for k in defaults}

    def _get_insights(self, question: str, answer: str, dims: Dict) -> Dict:
        try:
            raw      = _call_groq(
                _INSIGHT_PROMPT.format(
                    question    = question[:250],
                    answer      = answer[:400],
                    correctness = dims["correctness"],
                    depth       = dims["depth"],
                    clarity     = dims["clarity"],
                ),
                temperature=0.4, max_tokens=120,
            )
            strength = weakness = ""
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


_eval_agent = AnswerEvaluationAgent()


def score_answer(
    question: str, answer: str,
    q_type: str = "resume", topic: str = "", model: str = "",
) -> Tuple[int, str, Dict]:
    result = _eval_agent.evaluate(question, answer, q_type, topic)
    return result["final_score"], result["explanation"], result


# ─────────────────────────────────────────────────────────
#  LEGACY STUBS
# ─────────────────────────────────────────────────────────

def generate_first_question(jd_text, resume_text, model=""):
    agent = PersonalInfoAgent()
    info  = agent.extract_info(resume_text)
    return agent.generate_first_question(info).get("question", "Tell me about yourself.")

def generate_next_question(jd_text, resume_text, history, model=""):
    return ""

def generate_interview_questions(jd_text, resume_text, num_questions=5, model=""):
    return []