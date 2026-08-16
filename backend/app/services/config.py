"""
config.py — Central configuration for all ATS AI services
==========================================================
Place this file at:  backend/app/services/config.py

All service files (ollama_service.py, screening_agent.py,
report_service.py) import from here instead of hardcoding
settings inline.

To change the model or URL, edit ONLY this file.
"""

# ── Ollama (local LLM) ────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:latest"          # change to llama3.2:1b if low RAM

# ── Ollama request defaults ────────────────────────────────────────────────────
OLLAMA_TIMEOUT     = 120               # seconds per request
OLLAMA_NUM_PREDICT = 600               # max tokens per response

# ── Groq (cloud fallback — optional) ─────────────────────────────────────────
# Leave GROQ_API_KEY empty ("") to disable Groq and always use Ollama.
# If you have a free Groq key, paste it here for faster responses.
GROQ_API_KEY   = ""                    # e.g. "gsk_xxxx..."
GROQ_MODEL     = "llama3-8b-8192"      # free-tier Groq model
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT   = 30

# ── Embedding model (for semantic resume scoring) ─────────────────────────────
EMBED_MODEL = "nomic-embed-text"       # pulled via: ollama pull nomic-embed-text

# ── Screening thresholds ──────────────────────────────────────────────────────
SCREENING_STRONG  = 75    # score >= 75  → STRONG  (proceed immediately)
SCREENING_AVERAGE = 50    # score >= 50  → AVERAGE (proceed with caution)
SCREENING_WEAK    = 30    # score >= 30  → WEAK    (hold)
                          # score <  30  → REJECT

# ── Interview settings ────────────────────────────────────────────────────────
MAX_QUESTIONS    = 10     # hard ceiling on interview length
MAX_FOLLOWUPS    = 3      # max follow-up injections per interview
FOLLOWUP_SCORE   = 5      # inject follow-up if score <= this value
FOLLOWUP_WORDS   = 20     # inject follow-up if answer has fewer words
END_AVG_SCORE    = 7      # end interview early if avg score >= this
END_MIN_ANSWERED = 3      # minimum questions before early-end check