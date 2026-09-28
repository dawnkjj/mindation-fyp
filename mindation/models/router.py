# intent routing for student note-taking tasks

from __future__ import annotations

# library
import re
from typing import Any

from mindation.config import ModelSettings
from mindation.types import RouteResult

ROUTE_LABELS = [
    "summarise lecture",
    "create revision notes",
    "create flashcards",
    "create quiz questions",
    "extract action items",
]

# route a student request to a study task
class IntentRouter:

    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings
        self._classifier: Any | None = None

    # load the BART-MNLI model for zero-shot classification if enabled
    def _load_classifier(self):
        print(
        "[ROUTER CONFIG DEBUG]",
        f"use_bart_router={self.settings.use_bart_router}",
        f"router_model={self.settings.router_model}",
        )
        if not self.settings.use_bart_router:
            return None
        if self._classifier is not None:
            return self._classifier
        try:
            from transformers import pipeline  # type: ignore

            self._classifier = pipeline("zero-shot-classification", model=self.settings.router_model)
            return self._classifier
        except Exception as exc:
            print(
                "[ROUTER DEBUG] Failed to load BART-MNLI:",
                repr(exc),
            )
            return None
    # return route label and confidence
    def route(self, student_task: str, text_context: str) -> RouteResult:

        task = student_task.strip() or "summarise lecture and create revision notes"
        classifier = self._load_classifier()
        if classifier is not None:
            try:
                result = classifier(task, candidate_labels=ROUTE_LABELS)
                labels = result["labels"]
                scores = result["scores"]
                score_map = {label: float(score) for label, score in zip(labels, scores)}
                return RouteResult(labels[0], float(scores[0]), score_map, method="bart-mnli")
            except Exception as exc:
                print(
                    "[ROUTER DEBUG] BART routing failed:",
                    repr(exc),
                )

        lowered = f"{task} {text_context[:2000]}".lower()
        scores = {label: 0.1 for label in ROUTE_LABELS}
        # go to keyword if bart fails
        keyword_map = {
            "summarise lecture": ["summary", "summarise", "summarize", "overview", "lecture"],
            "create revision notes": ["revise", "revision", "exam", "notes", "study"],
            "create flashcards": ["flashcard", "flashcards", "memorise", "remember"],
            "create quiz questions": ["quiz", "question", "test", "practice"],
            "extract action items": ["action", "deadline", "homework", "todo", "task"],
        }

        # each label based on keyword match
        for label, words in keyword_map.items():
            for word in words:
                if re.search(rf"\b{re.escape(word)}\b", lowered):
                    scores[label] += 0.25
        total = sum(scores.values()) or 1.0

        # normalise the score and return best value
        norm = {label: value / total for label, value in scores.items()}
        best = max(norm, key=norm.get)
        return RouteResult(best, float(norm[best]), norm, method="keyword-fallback")
