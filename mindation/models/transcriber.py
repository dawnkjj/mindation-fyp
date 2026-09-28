# speech-to-text model wrappers

from __future__ import annotations

# library
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from mindation.config import ModelSettings
from mindation.types import TranscriptResult

logger = logging.getLogger(__name__)

# whisper model loader
@lru_cache(maxsize=3)
def _load_whisper_model(model_name: str) -> Any:
    import whisper

    return whisper.load_model(model_name)

# Transcribe audio using OpenAI Whisper
class WhisperTranscriber:

    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings

    def transcribe(
    self,
    audio_path: Path,
    language: str | None = "en",
    ) -> TranscriptResult:
        # Transcribe an audio file.

        if not audio_path.exists():
            return TranscriptResult(error=f"Audio file not found: {audio_path}")

        if not self.settings.use_whisper:
            return TranscriptResult(error="Whisper is disabled in the sidebar settings.")

        try:
            model = _load_whisper_model(self.settings.whisper_model)

            transcribe_options: dict[str, Any] = {
                "fp16": False,
                "task": "transcribe",
            }

            if language:
                transcribe_options["language"] = language

            result: dict[str, Any] = model.transcribe(
                str(audio_path),
                **transcribe_options,
            )
        except ModuleNotFoundError:
            return TranscriptResult(
                error="Whisper is not installed. Run: python -m pip install -r requirements-ai.txt"
            )
        except FileNotFoundError as exc:
            return TranscriptResult(
                error=(
                    "Audio decoding failed because FFmpeg could not be found. "
                    "Install requirements again or check imageio-ffmpeg. Details: "
                    f"{exc}"
                )
            )
        except Exception as exc:  # pragma: no cover - external model errors vary
            logger.exception("Whisper transcription failed")
            return TranscriptResult(error=f"Whisper transcription failed: {exc}")

        text = (result.get("text") or "").strip()
        segments = result.get("segments") or []
        duration = None
        if segments:
            try:
                duration = float(segments[-1].get("end", 0.0))
            except Exception:
                duration = None
        return TranscriptResult(
            text=text,
            language=result.get("language"),
            duration_seconds=duration,
            segments=segments,
            error=None if text else "Whisper returned an empty transcript.",
        )
