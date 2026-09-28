# retrieval over user-submitted lecture material
from __future__ import annotations

# standard library
import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mindation.config import AppLimits, ModelSettings
from mindation.types import EvidenceChunk, SourceDocument


@dataclass(slots=True)
# for text chunks from materials
class _Chunk:
    source_id: str
    source_type: str
    title: str
    text: str

# split text into overlapping chunks
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:

    # clean the text by normalizing whitespace and removing null bytes
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        window = text[start:end]
        if end < len(text):
            # Try to end around a sentence boundary.
            boundary = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if boundary > chunk_size * 0.45:
                end = start + boundary + 1
                window = text[start:end]
        chunks.append(window.strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]

# retrieve relevant chunks from submitted inputs.
class EvidenceRetriever:

    # initialize the retriever with model settings and application limits
    def __init__(self, settings: ModelSettings, limits: AppLimits) -> None:
        self.settings = settings
        self.limits = limits
        self._st_model = None
        self.last_method = "not-run"

    # build overlapping text chunks from source documents
    def _build_chunks(self, documents: Iterable[SourceDocument]) -> list[_Chunk]:
        chunks: list[_Chunk] = []
        # for each document split the text
        for doc in documents:
            for text in chunk_text(doc.text, self.limits.chunk_size, self.limits.chunk_overlap):
                chunks.append(_Chunk(doc.source_id, doc.source_type, doc.title, text))
        return chunks

    # load sentence transformer
    def _try_sentence_transformer(self):

        if not self.settings.use_sentence_transformer:

            return None

        if self._st_model is not None:
            return self._st_model

        # return none if fail
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(
                self.settings.embedding_model
            )

            return self._st_model

        except Exception as exc:

            return None
    # return top evidence chunks for a query
    def retrieve(self, query: str, documents: list[SourceDocument], top_k: int | None = None) -> list[EvidenceChunk]:

        chunks = self._build_chunks(documents)
        if not chunks:
            self.last_method = "none"
            return []

        top_k = top_k or self.settings.top_k_evidence
        query = query.strip() or "summarise the submitted lecture material into useful student notes"
        texts = [chunk.text for chunk in chunks]

        st_model = self._try_sentence_transformer()

        # if available, use sentence transformer if not fall back to tf-idf
        if st_model is not None:
            self.last_method = "sentence-transformer"

            doc_emb = st_model.encode(
                texts,
                normalize_embeddings=True,
            )

            query_emb = st_model.encode(
                [query],
                normalize_embeddings=True,
            )

            scores = np.dot(
                doc_emb,
                query_emb[0],
            )
        else:
            self.last_method = "tf-idf"

            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=8000,
            )

            matrix = vectorizer.fit_transform(
                texts + [query]
            )

            scores = cosine_similarity(
                matrix[-1],
                matrix[:-1],
            ).flatten()

        # rank the evidence chunks by score and return the top k
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)[:top_k]
        evidence: list[EvidenceChunk] = []
        for idx, score in ranked:
            if math.isnan(float(score)):
                continue
            chunk = chunks[idx]
            evidence.append(
                EvidenceChunk(
                    source_id=chunk.source_id,
                    source_type=chunk.source_type,
                    title=chunk.title,
                    text=chunk.text,
                    score=float(score),
                )
            )
        return evidence
