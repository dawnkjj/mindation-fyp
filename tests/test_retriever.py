# test file for retriever.py
from mindation.config import AppLimits, ModelSettings
from mindation.models.retriever import EvidenceRetriever, chunk_text
from mindation.types import SourceDocument

# test that chunk_text splits text into chunks of the specified size with overlap
def test_chunk_text_returns_chunks():
    text = "Sentence one. " * 120
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(chunks)

# test that EvidenceRetriever retrieves relevant evidence from source documents
def test_retriever_finds_relevant_chunk():
    docs = [
        SourceDocument("1", "typed_notes", "AI notes", "Retrieval augmented generation reduces hallucination by grounding answers in source documents."),
        SourceDocument("2", "typed_notes", "Cooking", "Pasta needs boiling water and salt."),
    ]

    # create an EvidenceRetriever with sentence transformer disabled
    retriever = EvidenceRetriever(ModelSettings(use_sentence_transformer=False), AppLimits())

    ## test that retriever retrieves relevant evidence for a query
    evidence = retriever.retrieve("How does retrieval reduce hallucination?", docs, top_k=1)
    assert evidence
    assert "hallucination" in evidence[0].text.lower()
