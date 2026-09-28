# summary generation model wrappers
from __future__ import annotations


#library
import json
import re
from typing import Any

import requests

from mindation.config import ModelSettings
from mindation.types import EvidenceChunk, RouteResult

#small sentence splitter suitable for fallback summarisation
def split_sentences(text: str) -> list[str]:

    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in pieces if len(p.strip()) > 20]

# generate grounded study notes
class SummaryGenerator:
    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings

    # oolama api calling
    def _ollama_generate(self, prompt: str) -> str | None:
        if not self.settings.use_ollama:
            return None
        payload = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        # handle exceptions
        try:
            response = requests.post(self.settings.ollama_url, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            return str(data.get("response", "")).strip() or None
        except Exception:
            return None

    #Generate a concise summary from retrieved evidence
    def generate_summary(self, evidence: list[EvidenceChunk], route: RouteResult, student_task: str) -> str:

        if not evidence:
            return "Nil - no usable user-submitted evidence was available for this run."

        evidence_text = "\n\n".join(
            f"[Source: {item.title} | {item.source_type} | score={item.score:.3f}]\n{item.text}"
            for item in evidence
        )

        # create a prompt for the LLM to generate a summary, key points, and revision actions
        prompt = f"""
You are Mindation, a student learning and wellbeing assistant.
Use ONLY the evidence below. Do not add external facts.
If the evidence is insufficient, write 'Nil' for that section.

Student task: {student_task or 'Create study notes from the submitted lecture material.'}
Detected route: {route.label}

Evidence:
{evidence_text}

Create:
1. A short lecture summary.
2. Important concepts.
3. Things the student should revise.
4. Any uncertainty or missing information.
""".strip()
        generated = self._ollama_generate(prompt)
        if generated:
            return generated[: self.settings.max_summary_words * 8]

        # Conservative extractive fallback.
        all_sentences: list[str] = []
        for item in evidence:
            all_sentences.extend(split_sentences(item.text))
        if not all_sentences:
            return evidence[0].text[:1500]
        selected = all_sentences[: min(8, len(all_sentences))]
        summary = " ".join(selected)
        return summary[: self.settings.max_summary_words * 8]
    #Extract key points from evidence
    def make_key_points(self, evidence: list[EvidenceChunk], max_points: int = 6) -> list[str]:

        if not evidence:
            return []
        sentences: list[str] = []
        for item in evidence:
            sentences.extend(split_sentences(item.text))
        # sentences with academic signal words
        signal = ["because", "therefore", "important", "model", "data", "evaluate", "method", "result", "should"]
        ranked = sorted(
            sentences,
            key=lambda s: sum(1 for word in signal if word in s.lower()),
            reverse=True,
        )
        points: list[str] = []
        seen: set[str] = set()
        for sentence in ranked:
            cleaned = sentence.strip(" -•")
            key = cleaned.lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            points.append(cleaned)
            if len(points) >= max_points:
                break
        return points

    #Create revision actions grounded in the available material
    def make_revision_actions(self, evidence: list[EvidenceChunk], route: RouteResult) -> list[str]:

        if not evidence:
            return []

        # create a list of revision actions based on the route label and evidence
        actions = [
            "Review the transcript and highlight terms that are repeated across the lecture.",
            "Compare the generated summary against the original submitted notes before using it for revision.",
            "Turn the flashcards into active recall practice and test without looking at the answers.",
        ]

        # add actions
        if route.label == "create quiz questions":
            actions.append("Use the quiz section as a short self-test after revising the notes.")

        if route.label == "extract action items":
            actions.append("Check whether the lecture mentioned coursework tasks, deadlines, or required readings.")
        return actions
