# backend/app/services/report_service.py
#
# Two responsibilities:
#   1. ReportAnalysisAgent  — calls Ollama to generate strength/weakness
#                             analysis + hire/reject recommendation
#   2. PDFReportBuilder     — builds a clean professional PDF from
#                             applicant data + analysis

import io
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple

import ollama

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY


# ─────────────────────────────────────────────────────────
#  COLOURS  (professional black & white + one accent)
# ─────────────────────────────────────────────────────────
BLACK      = colors.HexColor("#0d0d0d")
DARK_GRAY  = colors.HexColor("#2c2c2c")
MID_GRAY   = colors.HexColor("#555555")
LIGHT_GRAY = colors.HexColor("#f0f0f0")
RULE_GRAY  = colors.HexColor("#cccccc")
ACCENT     = colors.HexColor("#1a1a2e")   # near-black navy for headers
WHITE      = colors.white

PASS_COLOR   = colors.HexColor("#1a5c2e")
REJECT_COLOR = colors.HexColor("#7a1a1a")
HOLD_COLOR   = colors.HexColor("#5c4a00")

TYPE_LABELS = {
    "personal":   "Personal",
    "technical":  "Technical",
    "experience": "Experience",
    "followup":   "Follow-up",
}


# ─────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────

def _safe_join(items, sep=", "):
    if not items:
        return ""
    parts = []
    for item in items:
        if isinstance(item, dict):
            parts.append(next((str(v) for v in item.values() if v), str(item)))
        elif item:
            parts.append(str(item))
    return sep.join(parts)


def _strip_md(text):
    """Strip markdown formatting from candidate answers for clean PDF display."""
    if not text:
        return ""
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*",   r"\1", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
    text = re.sub(r"^\s*[-*+]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"-{3,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _call_llm(prompt: str, model: str = "llama3:latest", max_tokens: int = 512) -> str:
    resp = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": 0.3, "num_predict": max_tokens}
    )
    return resp["response"].strip()


def _parse_json_safe(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except:
                pass
    return {}


# ─────────────────────────────────────────────────────────
#  AGENT — REPORT ANALYSIS
#  Uses 4 small focused LLM calls instead of one big call.
#  llama3:latest handles small prompts reliably.
# ─────────────────────────────────────────────────────────

class ReportAnalysisAgent:

    def __init__(self, model: str = "llama3:latest"):
        self.model = model

    def analyse(
        self,
        name:             str,
        role:             str,
        match_score:      float,
        interview_score:  float,
        matched_skills:   List[str],
        skill_gaps:       List[str],
        questions:        List,
        answers:          List[str],
        scores:           List[int],
    ) -> Dict:

        # Build compact transcript — strip markdown from answers
        lines = []
        for i, q in enumerate(questions):
            q_text = q.get("question", str(q)) if isinstance(q, dict) else str(q)
            q_type = TYPE_LABELS.get(q.get("type",""), "Q") if isinstance(q, dict) else "Q"
            ans    = _strip_md(answers[i]) if i < len(answers) else "Not answered"
            sc     = scores[i] if i < len(scores) else "N/A"
            lines.append(f"[{q_type}] Q{i+1} (Score:{sc}/10): {q_text}\nAnswer: {ans[:200]}")

        transcript = "\n\n".join(lines)
        short_tx   = transcript[:2500]  # stay well within 1b context

        matched = _safe_join(matched_skills) or "Not specified"
        gaps    = _safe_join(skill_gaps)     or "None identified"

        # ── Call 1: Summary + skill/communication assessment ──
        summary_data = self._call_summary(name, role, match_score,
                                          interview_score, matched, gaps, short_tx)

        # ── Call 2: Strengths (3-5 points) ──
        strengths = self._call_strengths(name, interview_score, matched, short_tx)

        # ── Call 3: Weaknesses / areas for improvement (3-4 points) ──
        weaknesses = self._call_weaknesses(name, interview_score, gaps, short_tx)

        # ── Call 4: Recommendation + next steps ──
        rec_data = self._call_recommendation(name, role, match_score,
                                             interview_score, matched, gaps,
                                             strengths, weaknesses)

        return {
            "overall_summary":          summary_data.get("overall_summary",
                f"{name} completed a {len(questions)}-question interview scoring {interview_score}/10."),
            "strengths":               strengths,
            "weaknesses":              weaknesses,
            "skill_assessment":        summary_data.get("skill_assessment",
                f"Candidate matched {matched} and showed gaps in {gaps}."),
            "communication_assessment":summary_data.get("communication_assessment",
                "Communication was assessed across all interview responses."),
            "recommendation":          rec_data.get("recommendation",
                "HIRE" if interview_score >= 7 else "HOLD" if interview_score >= 5 else "REJECT"),
            "recommendation_reasoning":rec_data.get("recommendation_reasoning",
                f"Based on interview score of {interview_score}/10 and resume match of {match_score}/100."),
            "suggested_next_steps":    rec_data.get("suggested_next_steps",
                "Review transcript and proceed accordingly."),
        }

    # ── internal LLM calls ────────────────────────────────

    def _call_summary(self, name, role, match_score, interview_score,
                      matched, gaps, transcript) -> Dict:
        prompt = f"""You are a hiring manager. Write a brief evaluation of this candidate.

CANDIDATE: {name}  ROLE: {role}
RESUME MATCH: {match_score}/100  INTERVIEW SCORE: {interview_score}/10
MATCHED SKILLS: {matched}
SKILL GAPS: {gaps}

INTERVIEW HIGHLIGHTS:
{transcript[:1500]}

Return ONLY valid JSON:
{{
  "overall_summary": "3 sentences about this specific candidate based on their interview performance",
  "skill_assessment": "2 sentences on their technical skill level from what they demonstrated",
  "communication_assessment": "1 sentence on their communication clarity and structure"
}}"""
        print("[ReportAnalysisAgent] Call 1: summary...")
        try:
            raw = _call_llm(prompt, self.model, 400)
            return _parse_json_safe(raw)
        except Exception as e:
            print(f"[ReportAnalysisAgent] Summary call failed: {e}")
            return {}

    def _call_strengths(self, name, interview_score, matched, transcript) -> List[str]:
        prompt = f"""You are a hiring manager identifying a candidate's strengths.

CANDIDATE: {name}  INTERVIEW SCORE: {interview_score}/10
MATCHED SKILLS: {matched}

INTERVIEW HIGHLIGHTS:
{transcript[:1500]}

List 4 specific strengths this candidate showed. Each must reference something specific from the interview.

Return ONLY valid JSON:
{{"strengths": ["strength 1", "strength 2", "strength 3", "strength 4"]}}"""
        print("[ReportAnalysisAgent] Call 2: strengths...")
        try:
            raw  = _call_llm(prompt, self.model, 400)
            data = _parse_json_safe(raw)
            items = data.get("strengths", [])
            if items and isinstance(items, list):
                return [str(s) for s in items if s]
        except Exception as e:
            print(f"[ReportAnalysisAgent] Strengths call failed: {e}")
        # score-based fallback
        fallbacks = []
        if interview_score >= 7:
            fallbacks.append("Demonstrated strong technical knowledge with detailed, accurate answers")
        if matched:
            fallbacks.append(f"Proven experience with key required skills: {matched[:80]}")
        fallbacks.append("Completed all interview questions with consistent engagement")
        fallbacks.append("Provided structured responses showing organised thinking")
        return fallbacks

    def _call_weaknesses(self, name, interview_score, gaps, transcript) -> List[str]:
        prompt = f"""You are a hiring manager identifying areas for improvement in a candidate.

CANDIDATE: {name}  INTERVIEW SCORE: {interview_score}/10
SKILL GAPS: {gaps}

INTERVIEW HIGHLIGHTS:
{transcript[:1500]}

List 3 specific areas for improvement. Be honest and reference specific evidence from the interview.

Return ONLY valid JSON:
{{"weaknesses": ["area 1", "area 2", "area 3"]}}"""
        print("[ReportAnalysisAgent] Call 3: weaknesses...")
        try:
            raw  = _call_llm(prompt, self.model, 350)
            data = _parse_json_safe(raw)
            items = data.get("weaknesses", [])
            if items and isinstance(items, list):
                return [str(w) for w in items if w]
        except Exception as e:
            print(f"[ReportAnalysisAgent] Weaknesses call failed: {e}")
        fallbacks = []
        if gaps:
            fallbacks.append(f"Skill gaps identified in: {gaps[:100]} — not demonstrated in interview")
        if interview_score < 7:
            fallbacks.append("Some answers lacked depth or concrete real-world examples")
        fallbacks.append("Would benefit from more hands-on production experience to strengthen responses")
        return fallbacks

    def _call_recommendation(self, name, role, match_score, interview_score,
                              matched, gaps, strengths, weaknesses) -> Dict:
        str_summary = "; ".join(strengths[:3]) if strengths else "see transcript"
        wk_summary  = "; ".join(weaknesses[:2]) if weaknesses else "see transcript"
        prompt = f"""You are a hiring manager making a final hiring decision.

CANDIDATE: {name}  ROLE: {role}
RESUME MATCH: {match_score}/100  INTERVIEW SCORE: {interview_score}/10
STRENGTHS: {str_summary}
CONCERNS: {wk_summary}
SKILL GAPS: {gaps}

Make a hiring recommendation. Be specific about why.

Return ONLY valid JSON:
{{
  "recommendation": "HIRE or HOLD or REJECT",
  "recommendation_reasoning": "3-4 sentences explaining the decision with specific evidence",
  "suggested_next_steps": "1-2 sentences on exactly what should happen next"
}}"""
        print("[ReportAnalysisAgent] Call 4: recommendation...")
        try:
            raw = _call_llm(prompt, self.model, 400)
            return _parse_json_safe(raw)
        except Exception as e:
            print(f"[ReportAnalysisAgent] Recommendation call failed: {e}")
            rec = "HIRE" if interview_score >= 7 else "HOLD" if interview_score >= 5 else "REJECT"
            return {
                "recommendation": rec,
                "recommendation_reasoning": f"{name} scored {interview_score}/10 with a resume match of {match_score}/100. Strengths include {str_summary[:100]}. Concerns include {wk_summary[:100]}.",
                "suggested_next_steps": "Proceed with next round interview" if rec == "HOLD" else ("Extend offer" if rec == "HIRE" else "Send rejection with feedback"),
            }


# ─────────────────────────────────────────────────────────
#  PDF REPORT BUILDER
# ─────────────────────────────────────────────────────────

class PDFReportBuilder:
    """
    Builds a clean, professional A4/Letter PDF interview report.
    Returns bytes — FastAPI streams it directly to the browser.
    """

    def __init__(self):
        self.styles  = getSampleStyleSheet()
        self._build_styles()

    def _build_styles(self):
        """Define all custom paragraph styles."""
        self.s = {}

        self.s["doc_title"] = ParagraphStyle(
            "doc_title", parent=self.styles["Normal"],
            fontSize=22, fontName="Helvetica-Bold",
            textColor=ACCENT, alignment=TA_LEFT,
            spaceAfter=4,
        )
        self.s["doc_subtitle"] = ParagraphStyle(
            "doc_subtitle", parent=self.styles["Normal"],
            fontSize=10, fontName="Helvetica",
            textColor=MID_GRAY, alignment=TA_LEFT,
            spaceAfter=2,
        )
        self.s["section_heading"] = ParagraphStyle(
            "section_heading", parent=self.styles["Normal"],
            fontSize=11, fontName="Helvetica-Bold",
            textColor=ACCENT, spaceBefore=18, spaceAfter=6,
        )
        self.s["body"] = ParagraphStyle(
            "body", parent=self.styles["Normal"],
            fontSize=9.5, fontName="Helvetica",
            textColor=DARK_GRAY, leading=15,
            alignment=TA_JUSTIFY, spaceAfter=6,
        )
        self.s["body_bold"] = ParagraphStyle(
            "body_bold", parent=self.styles["Normal"],
            fontSize=9.5, fontName="Helvetica-Bold",
            textColor=DARK_GRAY, leading=15,
        )
        self.s["small"] = ParagraphStyle(
            "small", parent=self.styles["Normal"],
            fontSize=8.5, fontName="Helvetica",
            textColor=MID_GRAY, leading=13,
        )
        self.s["bullet"] = ParagraphStyle(
            "bullet", parent=self.styles["Normal"],
            fontSize=9.5, fontName="Helvetica",
            textColor=DARK_GRAY, leading=15,
            leftIndent=14, spaceAfter=3,
        )
        self.s["q_text"] = ParagraphStyle(
            "q_text", parent=self.styles["Normal"],
            fontSize=9, fontName="Helvetica-Bold",
            textColor=DARK_GRAY, leading=14,
            spaceAfter=3,
        )
        self.s["a_text"] = ParagraphStyle(
            "a_text", parent=self.styles["Normal"],
            fontSize=9, fontName="Helvetica",
            textColor=MID_GRAY, leading=14,
            leftIndent=10, spaceAfter=2,
            alignment=TA_JUSTIFY,
        )
        self.s["footer"] = ParagraphStyle(
            "footer", parent=self.styles["Normal"],
            fontSize=7.5, fontName="Helvetica",
            textColor=MID_GRAY, alignment=TA_CENTER,
        )

    # ── public entry point ────────────────────────────────
    def build(
        self,
        applicant_id:    int,
        name:            str,
        email:           str,
        created_at,
        match_score:     float,
        interview_score: float,
        status:          str,
        matched_skills:  List[str],
        skill_gaps:      List[str],
        questions:       List,
        answers:         List[str],
        scores:          List[int],
        analysis:        Dict,
        explanations:    List[str] = None,
    ) -> bytes:
        """Build and return PDF as bytes."""
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=letter,
            leftMargin=0.85*inch, rightMargin=0.85*inch,
            topMargin=0.9*inch,   bottomMargin=0.9*inch,
        )
        story = []
        explanations = explanations or []
        story += self._header(name, email, applicant_id, created_at)
        story += self._candidate_overview(match_score, interview_score,
                                          status, matched_skills, skill_gaps)
        story += self._executive_summary(analysis)
        story += self._strengths_weaknesses(analysis)
        story += self._recommendation(analysis, interview_score)
        story += self._transcript(questions, answers, scores, explanations)
        story += self._footer_note()

        doc.build(story, onFirstPage=self._page_template,
                  onLaterPages=self._page_template)
        return buf.getvalue()

    # ── page template (header/footer on every page) ───────
    def _page_template(self, canvas, doc):
        canvas.saveState()
        w, h = letter

        # top rule
        canvas.setStrokeColor(ACCENT)
        canvas.setLineWidth(2)
        canvas.line(0.85*inch, h - 0.5*inch, w - 0.85*inch, h - 0.5*inch)

        # "CONFIDENTIAL" top right
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MID_GRAY)
        canvas.drawRightString(w - 0.85*inch, h - 0.38*inch, "CONFIDENTIAL — ATS AI INTERVIEW REPORT")

        # bottom rule + page number
        canvas.setStrokeColor(RULE_GRAY)
        canvas.setLineWidth(0.5)
        canvas.line(0.85*inch, 0.6*inch, w - 0.85*inch, 0.6*inch)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MID_GRAY)
        canvas.drawCentredString(w/2, 0.4*inch, f"Page {doc.page}")

        canvas.restoreState()

    # ── sections ──────────────────────────────────────────

    def _header(self, name, email, applicant_id, created_at):
        date_str = created_at.strftime("%B %d, %Y") if hasattr(created_at, "strftime") \
                   else str(created_at)[:10]
        story = [
            Spacer(1, 0.15*inch),
            Paragraph("Interview Evaluation Report", self.s["doc_title"]),
            Paragraph(
                f"Candidate: <b>{name}</b> &nbsp;·&nbsp; "
                f"{email} &nbsp;·&nbsp; "
                f"ID #{applicant_id} &nbsp;·&nbsp; "
                f"{date_str}",
                self.s["doc_subtitle"]
            ),
            HRFlowable(width="100%", thickness=0.5, color=RULE_GRAY,
                       spaceAfter=12, spaceBefore=6),
        ]
        return story

    def _candidate_overview(self, match_score, interview_score,
                             status, matched_skills, skill_gaps):
        story = [Paragraph("Candidate Overview", self.s["section_heading"])]

        # score table
        score_color  = PASS_COLOR   if interview_score >= 7 else \
                       HOLD_COLOR   if interview_score >= 5 else REJECT_COLOR
        match_color  = PASS_COLOR   if match_score >= 60    else \
                       HOLD_COLOR   if match_score >= 40    else REJECT_COLOR
        status_clean = status.replace("_", " ").title()

        data = [
            [
                Paragraph("Resume Match", self.s["body_bold"]),
                Paragraph("Interview Score", self.s["body_bold"]),
                Paragraph("Screening Status", self.s["body_bold"]),
            ],
            [
                Paragraph(f'<font color="#{match_color.hexval()[2:]}">'
                          f'<b>{match_score}/100</b></font>', self.s["body"]),
                Paragraph(f'<font color="#{score_color.hexval()[2:]}">'
                          f'<b>{interview_score}/10</b></font>', self.s["body"]),
                Paragraph(f'<b>{status_clean}</b>', self.s["body"]),
            ],
        ]
        col_w = [(letter[0] - 1.7*inch) / 3] * 3
        t = Table(data, colWidths=col_w)
        t.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0), LIGHT_GRAY),
            ("GRID",         (0,0), (-1,-1), 0.4, RULE_GRAY),
            ("TOPPADDING",   (0,0), (-1,-1), 7),
            ("BOTTOMPADDING",(0,0), (-1,-1), 7),
            ("LEFTPADDING",  (0,0), (-1,-1), 10),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

        # skills row
        if matched_skills or skill_gaps:
            skills_data = []
            if matched_skills:
                skills_data.append([
                    Paragraph("<b>Matched Skills</b>", self.s["small"]),
                    Paragraph(", ".join(matched_skills), self.s["small"]),
                ])
            if skill_gaps:
                skills_data.append([
                    Paragraph("<b>Skill Gaps</b>", self.s["small"]),
                    Paragraph(", ".join(skill_gaps), self.s["small"]),
                ])
            st = Table(skills_data, colWidths=[1.3*inch, None])
            st.setStyle(TableStyle([
                ("VALIGN",      (0,0), (-1,-1), "TOP"),
                ("TOPPADDING",  (0,0), (-1,-1), 4),
                ("BOTTOMPADDING",(0,0),(-1,-1), 4),
                ("LEFTPADDING", (0,0), (-1,-1), 0),
            ]))
            story.append(st)

        return story

    def _executive_summary(self, analysis: Dict):
        story = [Paragraph("Executive Summary", self.s["section_heading"])]
        summary = analysis.get("overall_summary", "No summary available.")
        story.append(Paragraph(summary, self.s["body"]))

        # skill + communication assessments
        skill_assess = analysis.get("skill_assessment", "")
        comm_assess  = analysis.get("communication_assessment", "")
        if skill_assess:
            story.append(Paragraph(f"<b>Technical Skills:</b> {skill_assess}", self.s["body"]))
        if comm_assess:
            story.append(Paragraph(f"<b>Communication:</b> {comm_assess}", self.s["body"]))

        return story

    def _strengths_weaknesses(self, analysis: Dict):
        story = [Paragraph("Strengths &amp; Areas for Improvement", self.s["section_heading"])]

        strengths  = analysis.get("strengths", [])
        weaknesses = analysis.get("weaknesses", [])

        col_w = (letter[0] - 1.7*inch) / 2 - 4

        def bullet_list(items, label, bg):
            rows = [[Paragraph(f"<b>{label}</b>", self.s["body_bold"])]]
            for item in items:
                rows.append([Paragraph(f"• {item}", self.s["bullet"])])
            t = Table(rows, colWidths=[col_w])
            t.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,0), bg),
                ("GRID",          (0,0), (-1,-1), 0.4, RULE_GRAY),
                ("TOPPADDING",    (0,0), (-1,-1), 6),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                ("LEFTPADDING",   (0,0), (-1,-1), 10),
                ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ]))
            return t

        s_table = bullet_list(strengths or ["None noted"], "Strengths (from interview evidence)", LIGHT_GRAY)
        w_table = bullet_list(weaknesses or ["None noted"], "Areas for Improvement (with evidence)", LIGHT_GRAY)

        combined = Table([[s_table, w_table]], colWidths=[col_w + 4, col_w + 4])
        combined.setStyle(TableStyle([
            ("VALIGN",      (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 0),
            ("RIGHTPADDING",(0,0), (-1,-1), 0),
            ("TOPPADDING",  (0,0), (-1,-1), 0),
        ]))
        story.append(combined)
        return story

    def _recommendation(self, analysis: Dict, interview_score: float):
        rec      = analysis.get("recommendation", "HOLD").upper()
        reason   = analysis.get("recommendation_reasoning", "")
        next_s   = analysis.get("suggested_next_steps", "")

        rec_bg = PASS_COLOR if rec == "HIRE" else \
                 REJECT_COLOR if rec == "REJECT" else HOLD_COLOR

        story = [Paragraph("Hiring Recommendation", self.s["section_heading"])]

        # recommendation badge row
        badge_data = [[
            Paragraph(f"<b>{rec}</b>", ParagraphStyle(
                "rec_badge", parent=self.styles["Normal"],
                fontSize=14, fontName="Helvetica-Bold",
                textColor=WHITE, alignment=TA_CENTER,
            )),
            Paragraph(reason, self.s["body"]),
        ]]
        badge_col_w = [1.0*inch, letter[0] - 1.7*inch - 1.0*inch - 8]
        bt = Table(badge_data, colWidths=badge_col_w)
        bt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (0,0), rec_bg),
            ("BACKGROUND",    (1,0), (1,0), LIGHT_GRAY),
            ("GRID",          (0,0), (-1,-1), 0.4, RULE_GRAY),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ]))
        story.append(bt)

        if next_s:
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"<b>Suggested Next Steps:</b> {next_s}", self.s["body"]))

        return story

    def _transcript(self, questions, answers, scores, explanations=None):
        explanations = explanations or []
        story = [
            Paragraph("Interview Transcript", self.s["section_heading"]),
            Paragraph(
                "Full record of questions asked, candidate responses, per-answer scores, and evaluator notes.",
                self.s["small"]
            ),
            Spacer(1, 6),
        ]

        for i, q in enumerate(questions):
            q_text  = q.get("question", str(q)) if isinstance(q, dict) else str(q)
            q_type  = TYPE_LABELS.get(q.get("type",""), "Q") if isinstance(q, dict) else "Q"
            q_topic = q.get("topic", "")  if isinstance(q, dict) else ""
            answer  = _strip_md(answers[i]) if i < len(answers) else "Not answered"
            score   = scores[i]       if i < len(scores)       else None
            expl    = explanations[i] if i < len(explanations) else ""

            score_str   = f"{score}/10" if score is not None else "N/A"
            score_color = PASS_COLOR if (score or 0) >= 7 else \
                          HOLD_COLOR if (score or 0) >= 4 else REJECT_COLOR

            topic_str = f" · {q_topic}" if q_topic else ""

            block_rows = [
                Table([[
                    Paragraph(f"<b>Q{i+1}</b> &nbsp; [{q_type}{topic_str}]", self.s["small"]),
                    Paragraph(
                        f'<font color="#{score_color.hexval()[2:]}"><b>{score_str}</b></font>',
                        ParagraphStyle("sc_right", parent=self.styles["Normal"],
                                       fontSize=8.5, fontName="Helvetica-Bold",
                                       alignment=TA_RIGHT)
                    ),
                ]], colWidths=[None, 0.6*inch],
                   style=TableStyle([
                       ("BACKGROUND",    (0,0), (-1,-1), LIGHT_GRAY),
                       ("TOPPADDING",    (0,0), (-1,-1), 5),
                       ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                       ("LEFTPADDING",   (0,0), (-1,-1), 8),
                       ("RIGHTPADDING",  (0,0), (-1,-1), 8),
                       ("GRID",          (0,0), (-1,-1), 0, WHITE),
                   ])),
                Paragraph(q_text, self.s["q_text"]),
                Paragraph(answer[:600] + ("…" if len(answer) > 600 else ""), self.s["a_text"]),
            ]

            if expl:
                block_rows.append(
                    Paragraph(f"<i>Evaluator note: {expl}</i>", self.s["small"])
                )

            block_rows.append(
                HRFlowable(width="100%", thickness=0.4, color=RULE_GRAY,
                           spaceBefore=6, spaceAfter=6)
            )
            story.append(KeepTogether(block_rows))

        return story

    def _footer_note(self):
        return [
            Spacer(1, 0.2*inch),
            HRFlowable(width="100%", thickness=0.5, color=RULE_GRAY, spaceAfter=6),
            Paragraph(
                f"Generated by ATS AI Interview Agent · "
                f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC · "
                "This report is confidential and intended solely for the hiring team.",
                self.s["footer"]
            ),
        ]