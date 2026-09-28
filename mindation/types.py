# shared data classes for the Mindation pipeline
# these classes are used to represent the inputs and outputs of the pipeline, as well as intermediate results such as transcripts, routing results, and evidence chunks
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# text source submitted by the student.
@dataclass(slots=True)
class SourceDocument:

    source_id: str
    source_type: str
    title: str
    text: str

#from speech-to-text transcription.
@dataclass(slots=True)
class TranscriptResult:

    text: str = ""
    language: str | None = None
    duration_seconds: float | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    # property to check if the transcript is valid
    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and self.error is None

# result for the current student task
@dataclass(slots=True)
class RouteResult:

    label: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    method: str = "keyword"

# retrieved evidence chunk from submitted input
@dataclass(slots=True)
class EvidenceChunk:

    source_id: str
    source_type: str
    title: str
    text: str
    score: float

# evidence support result for a generated claim
@dataclass(slots=True)
class SupportCheck:

    claim: str
    status: str
    support_score: float
    evidence_title: str
    evidence_snippet: str

# generated study output
@dataclass(slots=True)
class StudyOutput:

    summary: str
    key_points: list[str]
    revision_actions: list[str]
    wellbeing_guidance: list[str]
    flashcards: list[dict[str, str]]
    quiz: list[dict[str, str]]
    route: RouteResult
    evidence: list[EvidenceChunk]
    support_checks: list[SupportCheck]
    transcript: TranscriptResult
    submitted_sources: list[SourceDocument]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

# input submitted
@dataclass(slots=True)
class PipelineInputs:

    text_documents: list[SourceDocument] = field(default_factory=list)
    audio_paths: list[Path] = field(default_factory=list)
    video_paths: list[Path] = field(default_factory=list)
    image_paths: list[Path] = field(default_factory=list)
    side_notes: str = ""
    student_task: str = ""
    wellbeing_checkin: dict[str, Any] = field(default_factory=dict)
