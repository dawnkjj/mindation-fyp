# Mindation

**Mindation** is my final-year Computer Science project for the CM3020 Artificial Intelligence project template:

**Project Idea 1: Orchestrating AI models to achieve a goal**

The project explores whether several pretrained AI models can work together to turn different types of university lecture material into useful revision resources.

The main idea came from a problem I regularly see when studying: lecture material is rarely stored in one clean place. A student might have PDF slides, typed notes, an audio recording, screenshots, a lecture video and small notes written during class. Mindation brings these different sources into one study session and uses them to create a structured Learning Pack.

The project is also designed around evidence. Rather than behaving like a general-purpose chatbot, Mindation tries to base academic outputs on the material submitted by the student during the current session.

---

## What Mindation can process

Mindation currently supports:

- TXT and Markdown notes
- PDF lecture notes and slides
- uploaded audio recordings
- browser microphone recordings
- lecture videos
- slide or whiteboard images
- typed class side notes

These different inputs are converted into textual evidence that can be searched and used by the rest of the pipeline.

---

## What Mindation creates

Depending on the student's request, the Learning Pack can contain:

- lecture summary
- key points
- revision actions
- study plan
- flashcards
- practice quiz questions
- audio/video transcript
- retrieved evidence
- Support Check results
- downloadable Markdown and JSON files

The final interface separates these results into different tabs so that the revision material, evidence and support information can be inspected individually.

---

## How the AI pipeline works

Mindation is built around several pretrained AI components rather than one model doing everything.
```text
Study materials
      |
      v
Modality processing
      |
      v
Evidence preparation and chunking
      |
      v
Evidence retrieval
      |
      v
Relevance check
      |
      v
Task routing
      |
      v
Revision-resource generation
      |
      v
Support Check
      |
      v
Learning Pack
``` 
## How to Start

Use Python **3.10, 3.11 or 3.12**.

cd path\to\mindation_project

py -3.12 -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel

python -m pip install -r requirements.txt

python -m streamlit run app.py
