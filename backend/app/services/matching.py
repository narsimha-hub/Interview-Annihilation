# backend/app/services/matching.py

import re
from typing import Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def clean_text(text: str) -> str:
    """Basic cleaning: lowercase, remove punctuation, normalize spaces"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tfidf_match(resume_text: str, jd_text: str) -> Tuple[float, Dict[str, float]]:
    """
    Compute similarity using TF-IDF + cosine similarity.
    Returns:
      - overall similarity score (0–100)
      - breakdown (very simple for now: just overall)
    """
    if not resume_text or not jd_text:
        return 0.0, {"overall": 0.0}

    # Clean both documents
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    if not resume_clean or not jd_clean:
        return 0.0, {"overall": 0.0}

    # Use TfidfVectorizer with reasonable settings for resumes/JDs
    vectorizer = TfidfVectorizer(
        stop_words='english',           # built-in stop word removal
        max_df=0.85,                    # ignore words that appear in >85% of docs
        min_df=1,                       # keep words that appear at least once
        ngram_range=(1, 2),             # include unigrams + bigrams (e.g. "machine learning")
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z0-9+-/#]*\b'  # keep tech terms like C++, .NET
    )

    # Fit & transform both texts together
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
    except ValueError as e:
        # Rare case: no valid terms found
        return 0.0, {"overall": 0.0, "error": str(e)}

    # Cosine similarity between resume (row 0) and JD (row 1)
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    # Convert to percentage (0–100)
    score = float(similarity_matrix[0][0]) * 100
    score = round(score, 1)

    # Very basic breakdown (can be extended later)
    breakdown = {"overall": score}

    return score, breakdown