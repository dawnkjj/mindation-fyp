# flashcard and quiz generation from submitted evidence
from __future__ import annotations

# library
import re
from collections import Counter

from mindation.types import EvidenceChunk

# stop words library
STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "have", "has", "are", "was", "were", "for",
    "you", "your", "about", "into", "using", "used", "can", "will", "not", "but", "they", "their",
    "lecture", "student", "students", "notes", "summary", "class",
}

# split into sentence and filter
def _sentences(evidence: list[EvidenceChunk]) -> list[str]:
    text = " ".join(item.text for item in evidence)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]

# extract keywords and ignore stopwords
def _keywords(evidence: list[EvidenceChunk], limit: int = 8) -> list[str]:
    text = " ".join(item.text.lower() for item in evidence)
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", text)
    counts = Counter(w for w in words if w not in STOPWORDS)
    return [word for word, _ in counts.most_common(limit)]

# create simple flashcards grounded in the submitted evidence
def build_flashcards(evidence: list[EvidenceChunk], limit: int = 6) -> list[dict[str, str]]:

    if not evidence:
        return []
    sentences = _sentences(evidence)
    keywords = _keywords(evidence, limit)
    cards: list[dict[str, str]] = []
    for keyword in keywords:
        answer = next((s for s in sentences if keyword.lower() in s.lower()), "")
        if not answer:
            continue
        cards.append(
            {
                "question": f"What does the lecture material say about '{keyword}'?",
                "answer": answer[:350],
            }
        )
        if len(cards) >= limit:
            break
    return cards

# create short-answer quiz questions from evidence
def build_quiz(evidence: list[EvidenceChunk], limit: int = 5) -> list[dict[str, str]]:

    cards = build_flashcards(evidence, limit)
    quiz: list[dict[str, str]] = []
    for index, card in enumerate(cards, start=1):
        quiz.append(
            {
                "question": f"Q{index}. {card['question']}",
                "expected_answer": card["answer"],
            }
        )
    return quiz
