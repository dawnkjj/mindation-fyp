from __future__ import annotations

# standard library imports
import logging
import re
import tempfile
from pathlib import Path

# standard library imports
from .config import AppLimits, ModelSettings
from .ingestion import clean_text, typed_source
from .media import extract_audio_from_video
from .models.retriever import EvidenceRetriever
from .models.router import IntentRouter
from .models.study_tools import build_flashcards, build_quiz
from .models.summarizer import SummaryGenerator
from .models.transcriber import WhisperTranscriber
from .models.faithfulness import check_summary_support
from .models.captioner import ImageCaptioner
from .types import PipelineInputs, SourceDocument, StudyOutput, TranscriptResult

# Set up logging for the module
logger = logging.getLogger(__name__)

# Define a relevance threshold for evidence retrieval
RELEVANCE_THRESHOLD = 0.3376

# this is the study words that are considered generic and not topic-specific, used to identify generic study requests
GENERIC_STUDY_WORDS = {
    "a", "an", "the", "this", "these", "my", "uploaded",
    "lecture", "lectures", "material", "materials", "notes", "content",
    "summarise", "summarize", "summary", "give", "me", "concise",
    "main", "idea", "ideas", "important", "key", "point", "points",
    "concept", "concepts", "topic", "topics", "create", "make", "turn",
    "into", "from", "study", "revision", "structured", "flashcard",
    "flashcards", "quiz", "question", "questions", "test", "practice",
    "practise", "extract", "action", "actions", "item", "items",
    "what", "should", "i", "do", "next", "to", "revise", "effectively",
    "about", "of", "for", "using", "on", "some", "please", "prepare",
    "organise", "organize","recorded","recording","audio","video","transcript",
}


def _is_generic_study_request(text: str) -> bool:
    """Return True when the request only describes a study operation."""
    if not text.strip():
        return True

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    content_tokens = [
        token
        for token in tokens
        if token not in GENERIC_STUDY_WORDS
    ]

    return len(content_tokens) == 0


class MindationPipeline:
    """Orchestrates the models and processing components for Mindation."""

    def __init__(self, settings: ModelSettings | None = None, limits: AppLimits | None = None) -> None:
        self.settings = settings or ModelSettings()
        self.limits = limits or AppLimits()
        self.transcriber = WhisperTranscriber(self.settings)
        self.router = IntentRouter(self.settings)
        self.retriever = EvidenceRetriever(self.settings, self.limits)
        self.generator = SummaryGenerator(self.settings)
        self.captioner = ImageCaptioner(self.settings)

    def _transcribe_media(self, inputs: PipelineInputs, warnings: list[str]) -> TranscriptResult:
        """Transcribe submitted audio and video files."""

        transcript_parts: list[str] = []
        all_segments = []
        language = None
        duration = 0.0
        errors: list[str] = []

        audio_paths = list(inputs.audio_paths)
        with tempfile.TemporaryDirectory(prefix="mindation_video_audio_") as temp_dir:
            temp_path = Path(temp_dir)
            for video_path in inputs.video_paths:
                try:
                    audio_from_video = extract_audio_from_video(video_path, temp_path)
                    audio_paths.append(audio_from_video)
                except Exception as exc:
                    errors.append(f"Could not extract audio from video '{video_path.name}': {exc}")

            for audio_path in audio_paths:
                result = self.transcriber.transcribe(audio_path)
                if result.text.strip():
                    transcript_parts.append(result.text.strip())
                    all_segments.extend(result.segments)
                    language = language or result.language
                    duration += float(result.duration_seconds or 0.0)
                if result.error:
                    errors.append(f"{audio_path.name}: {result.error}")

        if errors:
            warnings.extend(errors)
        text = clean_text("\n\n".join(transcript_parts))[: self.limits.max_transcript_chars]
        return TranscriptResult(
            text=text,
            language=language,
            duration_seconds=duration or None,
            segments=all_segments,
            error="; ".join(errors) if errors and not text else None,
        )

    def _caption_images(self, inputs: PipelineInputs, warnings: list[str]) -> list[SourceDocument]:
        """Caption submitted screenshots/whiteboards and convert captions to evidence sources."""

        image_sources: list[SourceDocument] = []
        for index, image_path in enumerate(inputs.image_paths, start=1):
            caption, error = self.captioner.caption(image_path)
            if caption.strip():
                image_sources.append(
                    SourceDocument(
                        source_id=f"image_{index}",
                        source_type="image_caption",
                        title=f"Image caption: {image_path.name}",
                        text=caption.strip(),
                    )
                )
            if error:
                warnings.append(f"{image_path.name}: {error}")
        return image_sources

    def _assemble_sources(self, inputs: PipelineInputs, transcript: TranscriptResult, image_sources: list[SourceDocument]) -> list[SourceDocument]:
        """Combine all current-session sources into source documents."""

        sources: list[SourceDocument] = []
        sources.extend(doc for doc in inputs.text_documents if doc.text.strip())
        sources.extend(image_sources)

        if inputs.side_notes.strip():
            side = typed_source(inputs.side_notes, "side_notes", "Typed class side notes", "student_side_notes")
            if side is not None:
                sources.append(side)

        if transcript.text.strip():
            sources.append(
                SourceDocument(
                    source_id="transcript",
                    source_type="speech_transcript",
                    title="Recorded lecture transcript",
                    text=transcript.text,
                )
            )
        return sources


    def _make_wellbeing_guidance(self, inputs: PipelineInputs, route, evidence) -> list[str]:
        """Create wellbeing-aware study guidance without treating wellbeing as academic evidence.

        The check-in is intentionally lightweight: it adapts revision pacing and
        workload suggestions, but it does not diagnose health conditions.
        """

        checkin = inputs.wellbeing_checkin or {}
        stress = int(checkin.get("stress", 3) or 3)
        focus = int(checkin.get("focus", 3) or 3)
        energy = int(checkin.get("energy", 3) or 3)
        concern = str(checkin.get("concern", "") or "").strip()

        guidance: list[str] = []
        if stress >= 4:
            guidance.append("Use a lighter revision block first: 20 minutes of focused review, then a short break before attempting quiz questions.")
        else:
            guidance.append("Start with the summary, then use the flashcards for active recall while the material is still fresh.")

        if focus <= 2:
            guidance.append("Break the learning pack into small sections: transcript check, key points, then only 3-5 flashcards at a time.")
        else:
            guidance.append("Use the evidence trail to verify the summary before turning the notes into a revision checklist.")

        if energy <= 2:
            guidance.append("Prioritise recognition tasks first, such as reading the key points, before moving to harder recall tasks such as the quiz.")
        else:
            guidance.append("Attempt the quiz without looking at the expected answers, then revisit the evidence for weak areas.")

        if concern:
            guidance.append(f"Current study concern noted: {concern}. Mindation uses this only to adjust study pacing, not as academic evidence.")

        guidance.append("Mindation is a study-support prototype, not a medical or counselling tool. If stress or wellbeing concerns feel serious or persistent, use university or trusted wellbeing support.")
        return guidance

    def run(self, inputs: PipelineInputs) -> StudyOutput:
        """Run the full orchestration pipeline."""

        warnings: list[str] = []
        transcript = self._transcribe_media(inputs, warnings)
        image_sources = self._caption_images(inputs, warnings)
        sources = self._assemble_sources(inputs, transcript, image_sources)
        combined_text = "\n\n".join(doc.text for doc in sources)

        if not sources:
            route = self.router.route(inputs.student_task, "")
            empty_transcript = transcript
            return StudyOutput(
                summary="Nil - no usable user-submitted evidence was available for this run.",
                key_points=[],
                revision_actions=[],
                wellbeing_guidance=self._make_wellbeing_guidance(inputs, route, []),
                flashcards=[],
                quiz=[],
                route=route,
                evidence=[],
                support_checks=[],
                transcript=empty_transcript,
                submitted_sources=[],
                warnings=warnings,
                metadata={"input_policy": "strict_current_session_only"},
            )

        route = self.router.route(inputs.student_task, combined_text)
        student_query = inputs.student_task.strip()

        # Separate a generic study instruction from a topic-specific academic query.
        # Example generic request: "Summarise the lecture material."
        # Example topical request: "What is SQL used for?"
        generic_study_request = _is_generic_study_request(student_query)

        if generic_study_request:
            query = (
                "important concepts definitions examples and key points "
                "from the submitted lecture material"
            )
        else:
            query = student_query

        evidence = self.retriever.retrieve(
            query,
            sources,
            top_k=self.settings.top_k_evidence,
        )
        top_scores = [
            item.score
            for item in evidence[:3]
        ]

        top3_mean = (
            sum(top_scores) / len(top_scores)
            if top_scores
            else 0.0
        )

        relevance_gate_applied = (
            bool(student_query)
            and not generic_study_request
            and self.retriever.last_method == "sentence-transformer"
        )
        print(
            "[RELEVANCE DEBUG]",
            f"query={student_query!r}",
            f"generic_task={generic_study_request}",
            f"method={self.retriever.last_method}",
            f"top_scores={top_scores}",
            f"top3_mean={top3_mean:.4f}",
            f"threshold={RELEVANCE_THRESHOLD}",
            f"gate_applied={relevance_gate_applied}",
        )

        if (
            relevance_gate_applied
            and top3_mean < RELEVANCE_THRESHOLD
        ):
            warnings.append(
                "The submitted material did not contain sufficiently "
                "relevant evidence for the requested task."
            )

            return StudyOutput(
                summary=(
                    "Insufficient relevant evidence was found in the "
                    "submitted lecture material for this request."
                ),
                key_points=[],
                revision_actions=[],
                wellbeing_guidance=self._make_wellbeing_guidance(
                    inputs,
                    route,
                    [],
                ),
                flashcards=[],
                quiz=[],
                route=route,
                evidence=evidence,
                support_checks=[],
                transcript=transcript,
                submitted_sources=sources,
                warnings=warnings,
                metadata={
                    "input_policy": "strict_current_session_only",
                    "source_count": len(sources),
                    "evidence_count": len(evidence),

                    "retrieval_method": self.retriever.last_method,
                    "generic_study_request": generic_study_request,
                    "relevance_gate": "rejected",
                    "relevance_threshold": RELEVANCE_THRESHOLD,
                    "top3_mean_similarity": top3_mean,

                    "learning_trail": [
                        "1. Capture submitted lecture material",
                        "2. Transcribe audio/video if provided",
                        "3. Route the student task",
                        "4. Retrieve evidence from current-session inputs",
                        "5. Check query-to-evidence relevance",
                        "6. Generate study notes and revision tools",
                        "7. Check generated claims against retrieved evidence",
                        "8. Adapt revision pacing using the optional wellbeing check-in",
                    ],

                    "wellbeing_checkin": inputs.wellbeing_checkin,

                    "models": {
                        "speech_to_text": (
                            "Whisper"
                            if self.settings.use_whisper
                            else "Disabled"
                        ),
                        "image_to_text": (
                            "BLIP"
                            if self.settings.use_blip_captioning
                            else "Disabled"
                        ),
                        "retrieval": self.retriever.last_method,
                        "routing": (
                            "BART-MNLI"
                            if self.settings.use_bart_router
                            else "Keyword fallback"
                        ),
                        "generation": (
                            "Ollama"
                            if self.settings.use_ollama
                            else "Extractive fallback"
                        ),
                    },
                },
            )

        # Mindation produces a complete learning pack for an accepted request.
        # The router still influences route-aware generation, while all major
        # revision resources remain available in the final StudyOutput.
        summary = self.generator.generate_summary(
            evidence,
            route,
            inputs.student_task,
        )
        key_points = self.generator.make_key_points(evidence)
        revision_actions = self.generator.make_revision_actions(
            evidence,
            route,
        )
        wellbeing_guidance = self._make_wellbeing_guidance(
            inputs,
            route,
            evidence,
        )
        flashcards = build_flashcards(evidence)
        quiz = build_quiz(evidence)
        support_checks = check_summary_support(summary, evidence)

        print(
            "[ROUTE DEBUG]",
            f"label={route.label!r}",
            f"confidence={route.confidence:.4f}",
            f"method={route.method}",
        )

        return StudyOutput(
            summary=summary,
            key_points=key_points,
            revision_actions=revision_actions,
            wellbeing_guidance=wellbeing_guidance,
            flashcards=flashcards,
            quiz=quiz,
            route=route,
            evidence=evidence,
            support_checks=support_checks,
            transcript=transcript,
            submitted_sources=sources,
            warnings=warnings,
            metadata={
                "input_policy": "strict_current_session_only",
                "source_count": len(sources),
                "evidence_count": len(evidence),
                "retrieval_method": self.retriever.last_method,
                "route_label": route.label,
                "route_confidence": route.confidence,
                "route_method": route.method,
                "generic_study_request": generic_study_request,
                "relevance_gate": (
                    "accepted"
                    if relevance_gate_applied
                    else "not-applied-generic-task"
                    if generic_study_request
                    else "not-applied"
                ),
                "relevance_threshold": (
                    RELEVANCE_THRESHOLD
                    if relevance_gate_applied
                    else None
                ),
                "top3_mean_similarity": top3_mean,
                "learning_trail": [
                    "1. Capture submitted lecture material",
                    "2. Transcribe audio/video if provided",
                    "3. Route the student task",
                    "4. Retrieve evidence from current-session inputs",
                    "5. Generate study notes and revision tools",
                    "6. Check generated claims against retrieved evidence",
                    "7. Adapt revision pacing using the optional wellbeing check-in",
                ],
                "wellbeing_checkin": inputs.wellbeing_checkin,
                "models": {
                    "speech_to_text": "Whisper" if self.settings.use_whisper else "Disabled",
                    "image_to_text": "BLIP" if self.settings.use_blip_captioning else "Disabled",
                    "retrieval": self.retriever.last_method,
                    "routing": "BART-MNLI" if self.settings.use_bart_router else "Keyword fallback",
                    "generation": "Ollama" if self.settings.use_ollama else "Extractive fallback",
                },
            },
        )
