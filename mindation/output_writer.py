# export functions for Mindation outputs
from __future__ import annotations

# standard library imports
import json
from dataclasses import asdict

from .types import StudyOutput

# convert a StudyOutput to a student-friendly report
def to_markdown(result: StudyOutput) -> str:

    # create a list of lines for the markdown report
    lines: list[str] = []
    lines.append("# Your Mindation Notes")
    lines.append("")
    lines.append("## Summary")
    lines.append(result.summary or "Nil")
    lines.append("")

    lines.append("## Key Points")

    # if there are key points, add them to the report if not indicate "Nil"
    if result.key_points:
        for point in result.key_points:
            lines.append(f"- {point}")
    else:
        lines.append("Nil")
    lines.append("")


    # Revision plan
    lines.append("## Revision Actions")
    if result.revision_actions:
        for action in result.revision_actions:
            lines.append(f"- {action}")
    else:
        lines.append("Nil")
    lines.append("")

    # Wellbeing guidance
    lines.append("## Mindful Revision Plan")
    if result.wellbeing_guidance:
        for item in result.wellbeing_guidance:
            lines.append(f"- {item}")
    else:
        lines.append("Nil")
    lines.append("")

    # Flashcards
    lines.append("## Flashcards")
    if result.flashcards:
        for idx, card in enumerate(result.flashcards, start=1):
            lines.append(f"### Card {idx}")

            # add the question and answer for each flashcard, with a blank line in between
            lines.append(f"**Q:** {card['question']}")
            lines.append(f"**A:** {card['answer']}")
            lines.append("")
    else:
        lines.append("Nil")
    lines.append("")

    # Quiz
    lines.append("## Quiz")
    if result.quiz:
        for item in result.quiz:
            # add the question and expected answer for each quiz item, with a blank line in between
            lines.append(f"- **{item['question']}**")
            lines.append(f"  - Expected answer: {item['expected_answer']}")
    else:
        lines.append("Nil")
    lines.append("")

    # Transcript
    lines.append("## Transcript")
    lines.append(result.transcript.text or "Nil")
    lines.append("")

    # Evidence Support Check
    lines.append("## Evidence Support Check")
    # if there are support checks, add them to the report with their status, score, claim, and closest evidence title; otherwise indicate "Nil"
    if result.support_checks:
        for item in result.support_checks:
            lines.append(f"- **{item.status.upper()}** ({item.support_score:.3f}) - {item.claim}")
            lines.append(f"  - Closest evidence: {item.evidence_title}")
    else:
        lines.append("Nil")
    lines.append("")

    # Learning Trail
    lines.append("## Learning Trail")
    for step in result.metadata.get("learning_trail", []):
        lines.append(f"- {step}")
    if not result.metadata.get("learning_trail"):
        lines.append("Nil")
    lines.append("")

    # Evidence Used
    lines.append("## Evidence Used")
    if result.evidence:
        # for each evidence chunk, add its title, source type, similarity score, and text to the report
        for idx, ev in enumerate(result.evidence, start=1):
            lines.append(f"### Evidence {idx}: {ev.title}")
            lines.append(f"- Source type: {ev.source_type}")
            lines.append(f"- Similarity score: {ev.score:.3f}")
            lines.append("")
            lines.append(ev.text)
            lines.append("")
    else:
        lines.append("Nil")
    lines.append("")

    # Warnings
    lines.append("## Warnings!")
    if result.warnings:
        for warning in result.warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("Nil")

    return "\n".join(lines)

# convert a StudyOutput to formatted JSON
def to_json(result: StudyOutput) -> str:

    return json.dumps(asdict(result), indent=2, ensure_ascii=False)
