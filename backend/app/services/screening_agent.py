"""
screening_agent.py — ResumeScreeningAgent
==========================================
Multi-dimensional resume screener using Groq LLM.

Dimensions scored (each 0-100):
  1. skills_match       — How many JD skills the resume has
  2. experience_level   — Candidate level vs required level
  3. education_match    — Degree field relevance + level
  4. project_relevance  — Projects relevant to JD domain
  5. role_continuity    — Career trajectory fit for the role

Final score = weighted average of all 5 dimensions.
Verdict     = PROCEED / REVIEW / REJECT based on thresholds.
"""

import re
import json
from typing import Dict, List, Tuple

# ── Import Groq caller from ollama_service ──────────────────────────────────────
# We reuse _call_groq and _parse_json so there's one LLM client everywhere.
try:
    from .ollama_service import _call_groq, _parse_json, _kw_extract
except ImportError:
    from backend.app.services.ollama_service import _call_groq, _parse_json, _kw_extract


# ─────────────────────────────────────────────────────────────────────────────
#  DIMENSION WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
_WEIGHTS = {
    "skills_match":     0.35,   # most important
    "experience_level": 0.25,
    "education_match":  0.15,
    "project_relevance":0.15,
    "role_continuity":  0.10,
}

# ─────────────────────────────────────────────────────────────────────────────
#  LLM PROMPT — single call, all 5 dimensions at once
# ─────────────────────────────────────────────────────────────────────────────
_SCREENING_PROMPT = """You are a strict ATS resume screener. Analyze the resume against the job description and score each dimension.

JOB DESCRIPTION:
{jd_text}

RESUME:
{resume_text}

Score each dimension from 0 to 100:

1. skills_match: How many required technical skills from the JD are present in the resume?
   - 80-100: Has 80%+ of required skills
   - 60-79: Has 60-80% of required skills  
   - 40-59: Has 40-60% of required skills
   - 0-39: Missing most required skills

2. experience_level: Does the candidate's experience match the required level?
   - Check years of experience, seniority of roles, complexity of work
   - 80-100: Perfect match (senior role = senior candidate)
   - 60-79: One level off (mid role, slightly junior candidate)
   - 40-59: Two levels off
   - 0-39: Major mismatch

3. education_match: Is the candidate's education relevant?
   - Check degree field (CS/IT/Engineering preferred for tech roles)
   - Check degree level (Bachelor/Master/PhD)
   - 80-100: CS/IT/Engineering degree, relevant field
   - 60-79: Related field (Math, Physics, MCA)
   - 40-59: Unrelated field but has tech certifications
   - 0-39: No relevant education

4. project_relevance: Are the candidate's projects relevant to the JD?
   - Check if projects use technologies mentioned in JD
   - Check if projects solve problems similar to the role
   - 80-100: Projects directly match JD domain and tech stack
   - 60-79: Projects partially relevant
   - 40-59: Some tangential relevance
   - 0-39: No relevant projects

5. role_continuity: Does the candidate's career trajectory fit this role?
   - Is this a natural next step in their career?
   - Are previous roles in the same domain?
   - 80-100: Clear progression toward this exact role
   - 60-79: Some relevant experience, minor pivot
   - 40-59: Career change, some transferable skills
   - 0-39: Completely unrelated career path

Also extract:
- matched_skills: list of skills the candidate HAS that the JD requires (max 10)
- missing_skills: list of skills the JD requires that the candidate LACKS (max 8)
- candidate_level: junior|mid|senior
- required_level: junior|mid|senior (infer from JD)
- education_field: the candidate's degree field (e.g. "Computer Science", "Electronics")
- key_projects: list of 3 most relevant project names from resume
- notes: one specific note per dimension explaining the score

Return ONLY valid JSON, no markdown fences:
{{
  "skills_match": 0,
  "experience_level": 0,
  "education_match": 0,
  "project_relevance": 0,
  "role_continuity": 0,
  "matched_skills": [],
  "missing_skills": [],
  "candidate_level": "junior",
  "required_level": "mid",
  "education_field": "",
  "key_projects": [],
  "notes": {{
    "skills_match": "",
    "experience_level": "",
    "education_match": "",
    "project_relevance": "",
    "role_continuity": ""
  }},
  "summary": ""
}}"""


# ─────────────────────────────────────────────────────────────────────────────
#  KEYWORD FALLBACK — used when LLM fails
# ─────────────────────────────────────────────────────────────────────────────
_EDU_KW = {
    "computer science", "cs", "information technology", "it", "software engineering",
    "electronics", "electrical", "mca", "bca", "btech", "b.tech", "m.tech",
    "mtech", "bsc", "msc", "data science", "artificial intelligence", "ai",
    "machine learning", "mathematics", "physics", "engineering",
}

_EXP_PATTERNS = [
    r'(\d+)\s*\+?\s*years?\s+of\s+experience',
    r'(\d+)\s*\+?\s*yrs?\s+experience',
    r'experience\s+of\s+(\d+)\s*\+?\s*years?',
]

def _extract_years(text: str) -> float:
    """Best-effort extraction of experience years from text."""
    for pat in _EXP_PATTERNS:
        m = re.search(pat, text.lower())
        if m:
            return float(m.group(1))
    return 0.0

def _keyword_fallback(resume_text: str, jd_text: str) -> Dict:
    """Pure keyword-based scoring when LLM is unavailable."""
    jd_kw  = _kw_extract(jd_text)
    res_kw = _kw_extract(resume_text)

    matched  = list(jd_kw & res_kw)
    missing  = list(jd_kw - res_kw)
    sm_score = round(min(100, (len(matched) / max(1, len(jd_kw))) * 100 + min(10, len(matched) * 1.5)), 1)

    # Education
    res_lower = resume_text.lower()
    edu_score = 50
    for kw in _EDU_KW:
        if kw in res_lower:
            edu_score = 70
            break
    if "computer science" in res_lower or "software" in res_lower:
        edu_score = 85

    # Experience
    years     = _extract_years(resume_text)
    jd_years  = _extract_years(jd_text)
    if jd_years == 0:
        exp_score = 60
    elif years >= jd_years:
        exp_score = min(90, 60 + int((years - jd_years) * 5))
    else:
        exp_score = max(30, int((years / max(1, jd_years)) * 70))

    return {
        "skills_match":      sm_score,
        "experience_level":  exp_score,
        "education_match":   edu_score,
        "project_relevance": sm_score * 0.8,   # rough proxy
        "role_continuity":   55.0,
        "matched_skills":    matched[:10],
        "missing_skills":    missing[:8],
        "candidate_level":   "junior" if years < 2 else ("senior" if years > 5 else "mid"),
        "required_level":    "junior" if jd_years < 2 else ("senior" if jd_years > 5 else "mid"),
        "education_field":   "Not detected",
        "key_projects":      [],
        "notes": {
            "skills_match":      f"Keyword match: {len(matched)}/{len(jd_kw)} skills",
            "experience_level":  f"Detected ~{years} years experience",
            "education_match":   "Keyword-based education detection",
            "project_relevance": "Estimated from skill overlap",
            "role_continuity":   "Unable to assess without LLM",
        },
        "summary":  f"Keyword screening: {len(matched)} of {len(jd_kw)} required skills matched.",
        "_method":  "keyword_fallback",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN AGENT CLASS
# ─────────────────────────────────────────────────────────────────────────────
class ResumeScreeningAgent:
    """
    Screens a resume against a JD.

    Returns a dict with:
      final_score      — 0-100 weighted score
      verdict          — PROCEED | REVIEW | REJECT
      verdict_color    — green | amber | red
      dimension_scores — {skills_match, experience_level, ...} each 0-100
      dimension_notes  — one sentence per dimension
      matched_skills   — list of matched skill strings
      missing_skills   — list of missing skill strings
      candidate_level  — junior | mid | senior
      required_level   — junior | mid | senior
      education_field  — degree field string
      key_projects     — list of project name strings
      screening_notes  — list of human-readable bullets
      summary          — 1-2 sentence overall summary
    """

    def screen(self, resume_text: str, jd_text: str) -> Dict:
        print("[ScreeningAgent] Starting multi-dimensional screening...")

        # ── Try LLM scoring ────────────────────────────────────────────────────
        raw_scores = None
        try:
            prompt = _SCREENING_PROMPT.format(
                jd_text     = jd_text[:2000],
                resume_text = resume_text[:2500],
            )
            raw        = _call_groq(prompt, temperature=0.1, max_tokens=900)
            raw_scores = _parse_json(raw)
            print(f"[ScreeningAgent] LLM scored: "
                  f"skills={raw_scores.get('skills_match')} "
                  f"exp={raw_scores.get('experience_level')} "
                  f"edu={raw_scores.get('education_match')} "
                  f"proj={raw_scores.get('project_relevance')} "
                  f"cont={raw_scores.get('role_continuity')}")
        except Exception as e:
            print(f"[ScreeningAgent] LLM failed: {e} → keyword fallback")
            raw_scores = _keyword_fallback(resume_text, jd_text)

        # ── Validate and clamp all dimension scores ────────────────────────────
        dims = {}
        for key in _WEIGHTS:
            val = raw_scores.get(key, 0)
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = 0.0
            dims[key] = round(max(0.0, min(100.0, val)), 1)

        # ── Weighted final score ───────────────────────────────────────────────
        final_score = round(
            sum(dims[k] * _WEIGHTS[k] for k in _WEIGHTS), 1
        )

        # ── Verdict ────────────────────────────────────────────────────────────
        if final_score >= 65:
            verdict       = "PROCEED"
            verdict_color = "green"
        elif final_score >= 45:
            verdict       = "REVIEW"
            verdict_color = "amber"
        else:
            verdict       = "REJECT"
            verdict_color = "red"

        # ── Extract lists safely ───────────────────────────────────────────────
        matched_skills = self._safe_list(raw_scores.get("matched_skills", []))
        missing_skills = self._safe_list(raw_scores.get("missing_skills", []))
        key_projects   = self._safe_list(raw_scores.get("key_projects",   []))

        # ── Dimension notes ────────────────────────────────────────────────────
        raw_notes = raw_scores.get("notes", {})
        if not isinstance(raw_notes, dict):
            raw_notes = {}
        dimension_notes = {
            k: str(raw_notes.get(k, self._default_note(k, dims[k])))
            for k in _WEIGHTS
        }

        # ── Human-readable screening notes ────────────────────────────────────
        screening_notes = self._build_notes(
            dims, matched_skills, missing_skills,
            raw_scores.get("candidate_level", ""),
            raw_scores.get("required_level", ""),
        )

        summary = str(raw_scores.get("summary", "")) or (
            f"Overall score {final_score}/100. "
            f"Matched {len(matched_skills)} required skills. "
            f"Verdict: {verdict}."
        )

        print(f"[ScreeningAgent] Final={final_score} Verdict={verdict} | "
              f"Dims: {dims}")

        return {
            "final_score":      final_score,
            "verdict":          verdict,
            "verdict_color":    verdict_color,
            "dimension_scores": dims,
            "dimension_notes":  dimension_notes,
            "matched_skills":   matched_skills[:10],
            "missing_skills":   missing_skills[:8],
            "candidate_level":  str(raw_scores.get("candidate_level", "junior")),
            "required_level":   str(raw_scores.get("required_level",  "mid")),
            "education_field":  str(raw_scores.get("education_field", "")),
            "key_projects":     key_projects[:5],
            "screening_notes":  screening_notes,
            "summary":          summary,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _safe_list(self, value) -> List[str]:
        """Convert whatever the LLM returned into a clean list of strings."""
        if not value:
            return []
        if isinstance(value, str):
            # Sometimes LLM returns a comma-separated string
            return [v.strip() for v in value.split(",") if v.strip()]
        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, dict):
                    # e.g. {"skill": "Python"} → "Python"
                    first_val = next(
                        (str(v) for v in item.values() if v), str(item)
                    )
                    result.append(first_val)
                elif item:
                    result.append(str(item))
            return result
        return []

    def _default_note(self, dimension: str, score: float) -> str:
        """Generate a default note when LLM didn't provide one."""
        label = {
            "skills_match":      "Technical skills",
            "experience_level":  "Experience",
            "education_match":   "Education",
            "project_relevance": "Projects",
            "role_continuity":   "Career continuity",
        }.get(dimension, dimension)
        if score >= 80:
            return f"{label}: Strong match ({score:.0f}/100)"
        elif score >= 60:
            return f"{label}: Moderate match ({score:.0f}/100)"
        elif score >= 40:
            return f"{label}: Partial match ({score:.0f}/100)"
        else:
            return f"{label}: Weak match ({score:.0f}/100)"

    def _build_notes(
        self,
        dims: Dict,
        matched: List,
        missing: List,
        candidate_level: str,
        required_level: str,
    ) -> List[str]:
        """Build a list of human-readable screening bullets."""
        notes = []

        # Skills
        if matched:
            notes.append(f"✅ Matched {len(matched)} required skills: {', '.join(matched[:5])}")
        if missing:
            notes.append(f"⚠️ Missing {len(missing)} skills: {', '.join(missing[:4])}")

        # Experience
        if candidate_level and required_level:
            if candidate_level == required_level:
                notes.append(f"✅ Experience level matches: {candidate_level}")
            else:
                notes.append(
                    f"⚠️ Level mismatch: candidate is {candidate_level}, role needs {required_level}"
                )

        # Dimension highlights
        if dims.get("education_match", 0) >= 75:
            notes.append("✅ Education background is relevant to this role")
        elif dims.get("education_match", 0) < 50:
            notes.append("⚠️ Education may not align with role requirements")

        if dims.get("project_relevance", 0) >= 70:
            notes.append("✅ Projects show relevant domain experience")
        elif dims.get("project_relevance", 0) < 45:
            notes.append("⚠️ Projects lack relevance to the JD domain")

        if dims.get("role_continuity", 0) >= 70:
            notes.append("✅ Career path aligns well with this role")

        return notes