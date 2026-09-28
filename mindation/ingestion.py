# input for ingestion
# this is for the mindation ingestion module

from __future__ import annotations

# standard library imports
import re
from pathlib import Path

from pypdf import PdfReader

from .config import AppLimits
from .types import SourceDocument

# for clean text get rid of null bytes, normalise whitespace, and remove excessive newlines
def clean_text(text: str) -> str:

    # clean text by removing null bytes, normalizing whitespace, and removing excessive newlines
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# read the file and return the text
def read_text_file(path: Path, limits: AppLimits | None = None) -> str:

    # read a text file and return the cleaned text, limited to max_text_chars
    limits = limits or AppLimits()
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return clean_text(raw[: limits.max_text_chars])

# extract text from a PDF file using pypdf, limited to max_text_chars
def read_pdf_file(path: Path, limits: AppLimits | None = None) -> str:

    limits = limits or AppLimits()
    reader = PdfReader(str(path))
    pages: list[str] = []
    total_chars = 0
    for page in reader.pages:
        page_text = page.extract_text() or ""

        # skip empty pages
        if not page_text.strip():
            continue
        pages.append(page_text)
        total_chars += len(page_text)

        # stop if we reach the maximum character limit
        if total_chars >= limits.max_text_chars:
            break

    return clean_text("\n\n".join(pages)[: limits.max_text_chars])

# create a SourceDocument from a user-uploaded file
def source_from_file(path: Path, source_id: str, limits: AppLimits | None = None) -> SourceDocument:

    suffix = path.suffix.lower()

    # for pdf files
    if suffix == ".pdf":
        text = read_pdf_file(path, limits)
        source_type = "uploaded_pdf"

    # for text files
    elif suffix in {".txt", ".md"}:
        text = read_text_file(path, limits)
        source_type = "uploaded_notes"
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # return a SourceDocument with the extracted text and metadata
    return SourceDocument(
        source_id=source_id,
        source_type=source_type,
        title=path.name,
        text=text,
    )

# create a source from typed text
def typed_source(text: str, source_id: str, title: str, source_type: str) -> SourceDocument | None:

    cleaned = clean_text(text)
    if not cleaned:
        return None
    return SourceDocument(source_id=source_id, source_type=source_type, title=title, text=cleaned)
