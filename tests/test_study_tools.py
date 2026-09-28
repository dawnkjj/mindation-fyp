# test file for study_tools.py
from mindation.models.study_tools import build_flashcards, build_quiz
from mindation.types import EvidenceChunk

# test that flashcards and quiz are generated correctly from evidence
def test_flashcards_and_quiz_are_grounded():
    evidence = [
        EvidenceChunk("1", "typed_notes", "Notes", "Retrieval uses relevant documents to ground generated summaries and reduce unsupported claims.", 0.9)
    ]

    # build flashcards and quiz from evidence
    cards = build_flashcards(evidence)
    quiz = build_quiz(evidence)
    assert cards
    assert quiz
    assert "Retrieval" in cards[0]["question"] or "retrieval" in cards[0]["answer"].lower()
