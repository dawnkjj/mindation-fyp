# evidence-bounded support checking for generated study notes
from __future__ import annotations

# standard library
import re
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mindation.types import EvidenceChunk, SupportCheck

#split generated text into short claims suitable for support checking
def split_claims(text: str) -> list[str]:

    # split the text into sentences or claims, clean them, and filter out short or irrelevant ones
    raw = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    claims: list[str] = []

    #clean each cliam by removing unwanted characters and filtering out irrelevant
    for item in raw:
        cleaned = re.sub(r"^[\-•\d\.\)\s]+", "", item).strip()
        if len(cleaned) >= 35 and not cleaned.lower().startswith("nil"):
            claims.append(cleaned[:450])
    return claims[:10]

# find the best matching evidence chunk for a given claim using TF-IDF and cosine similarity
def _best_support(claim: str, evidence: list[EvidenceChunk]) -> tuple[float, EvidenceChunk | None]:
    if not claim.strip() or not evidence:
        return 0.0, None
    evidence_texts = [item.text for item in evidence]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=6000)
        matrix = vectorizer.fit_transform(evidence_texts + [claim])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    except Exception:
        return 0.0, None
    best_idx = int(scores.argmax())
    return float(scores[best_idx]), evidence[best_idx]

# check generated claims are supported by retrieved evidence
def check_summary_support(summary: str, evidence: list[EvidenceChunk]) -> list[SupportCheck]:

    checks: list[SupportCheck] = []

    # split the summary into individual claims and evaluate each against the evidence
    for claim in split_claims(summary):
        score, source = _best_support(claim, evidence)
        if score >= 0.18:
            status = "supported"
        elif score >= 0.08:
            status = "partly supported"
        else:
            status = "needs review"

        # create a object for each claim
        checks.append(
            SupportCheck(
                claim=claim,
                status=status,
                support_score=score,
                evidence_title=source.title if source else "Nil",
                evidence_snippet=(source.text[:320] if source else "Nil"),
            )
        )
    return checks
