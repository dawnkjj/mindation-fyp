# Mindation configuration settings
from __future__ import annotations

from dataclasses import dataclass

# this is for the mindation model configuration settings
@dataclass(slots=True)

# class for model settings, including which models to use for various tasks
class ModelSettings:
    use_whisper: bool = True
    whisper_model: str = "base"
    use_sentence_transformer: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_blip_captioning: bool = False
    image_caption_model: str = "Salesforce/blip-image-captioning-base"
    use_bart_router: bool = True
    router_model: str = "facebook/bart-large-mnli"
    use_ollama: bool = False
    ollama_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434/api/generate"
    max_summary_words: int = 450
    top_k_evidence: int = 6

# class for application limits, including maximum text and transcript sizes, chunk size, and overlap
@dataclass(slots=True)
class AppLimits:

    max_text_chars: int = 120_000
    max_transcript_chars: int = 120_000
    chunk_size: int = 900
    chunk_overlap: int = 120
