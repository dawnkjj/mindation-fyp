## Main interface for Mindation
## this file is for the user-facing workflow
# the ai processing itself is kept inside the mindation.pipeline module
from __future__ import annotations

# standard library imports
import tempfile
from pathlib import Path
from typing import Any

# streamlit is used for the user interface and workflow
import streamlit as st

# this is the main interface for the Mindation app
from mindation.config import AppLimits, ModelSettings
from mindation.ingestion import source_from_file, typed_source
from mindation.output_writer import to_json, to_markdown
from mindation.pipeline import MindationPipeline
from mindation.types import PipelineInputs, SourceDocument, StudyOutput

# this is the main entry point for the Streamlit app
st.set_page_config(
    page_title="Mindation",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

#this is the main interface for styling
#separate from the python ui logic
APP_CSS_PATH = Path(__file__).with_name("app_styles.html")

with APP_CSS_PATH.open("r", encoding="utf-8") as css_file:
    APP_CSS = css_file.read()

st.markdown(APP_CSS, unsafe_allow_html=True)

# this function save an uploaded file to a temporary directory and returns the path
def save_uploaded_file(upload: Any, directory: Path) -> Path:
    #save a Streamlit UploadedFile to a temporary directory

    directory.mkdir(parents=True, exist_ok=True)

    # this prevents directory information from being included in the saved file name
    path = directory / Path(upload.name).name


    path.write_bytes(upload.getbuffer())
    return path

## renders the mindation branding and introduction of the project
def render_header() -> None:
    st.markdown(
        """
        <div class="topbar">
          <div class="brand-mark">
            <div class="brand-symbol">M</div>
            <div>
              <div class="brand-title">Mindation</div>
              <div class="brand-subtitle">Mind and education in one study workspace</div>
            </div>
          </div>
          <div class="session-pill">Session-based study support</div>
        </div>
        <div class="hero">
          <h1>Turn lecture material into a focused study pack.</h1>
          <p>Upload notes, recordings, videos or class side notes. Mindation uses only the material submitted in the current session to create summaries, flashcards, quizzes, an evidence trail and a manageable study plan.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

##this is the sidebar for the app
def render_sidebar() -> tuple[ModelSettings, AppLimits]:
    st.sidebar.success("AI study pipeline ready")

    ## these are the models used by the final pipeline
    # all automatically enabled
    settings = ModelSettings(
        use_whisper=True,
        whisper_model="base",
        use_blip_captioning=True,
        use_sentence_transformer=True,
        use_bart_router=True,
        use_ollama=True,
        ollama_model="llama3.2:3b",
        top_k_evidence=6,
    )

    return settings, AppLimits()

# the current study session and convert it into pipeline inputs for the MindationPipeline
def collect_inputs(temp_dir: Path, limits: AppLimits) -> PipelineInputs:

    # collect all the user inputs from the interface and convert them into a PipelineInputs object
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="step">01</span><div><h3 style="margin:0">Add lecture materials</h3><p class="subtle" style="margin:.15rem 0 0">Use notes, recordings, videos, images or side notes from this study session.</p></div></div>', unsafe_allow_html=True)

    # the columns for the file uploaders
    #depends on what the student wants to upload so have a few options
    col_a, col_b = st.columns(2)
    with col_a:
        # for each type of file, we allow multiple uploads and specify the accepted file types
        # this for text files, markdown files and pdfs
        note_uploads = st.file_uploader("Lecture notes or slides", type=["txt", "md", "pdf"], accept_multiple_files=True)

        # this for audio files, including mp3, wav, m4a, ogg and flac
        audio_uploads = st.file_uploader("Recorded audio", type=["mp3", "wav", "m4a", "ogg", "flac"], accept_multiple_files=True)

        # this for image files, including png, jpg, jpeg and webp
        image_uploads = st.file_uploader("Slide or whiteboard images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    with col_b:
        # this for video files, including mp4, mov, mkv and webm
        video_uploads = st.file_uploader("Lecture video", type=["mp4", "mov", "mkv", "webm"], accept_multiple_files=True)
        # this for microphone audio input
        microphone_audio = st.audio_input("Record audio here")

    # this is a text area for the student to add any side notes from the class, such as lecturer comments, examples, reminders or exam hints
    typed_notes = st.text_area(
        # this is the label for the text area
        "Class side notes",
        placeholder="Add lecturer comments, examples, reminders or exam hints from class.",
        height=135,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # for students to specify what they want to create
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="step">02</span><div><h3 style="margin:0">Choose your study output</h3><p class="subtle" style="margin:.15rem 0 0">Describe the study pack you want to create.</p></div></div>', unsafe_allow_html=True)
    student_task = st.text_input(
        "What would you like to create?",
        placeholder="Create a summary, key points, flashcards, a quiz and a study plan.",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    #this is the study check-in section for students to specify their current stress, focus and energy levels
    #pacing support and is not used as lecture evidence
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="step">03</span><div><h3 style="margin:0">Study check-in</h3><p class="subtle" style="margin:.15rem 0 0">Optional pacing support. These values are not used as lecture evidence.</p></div></div>', unsafe_allow_html=True)
    wb1, wb2, wb3 = st.columns(3)

    # sliders for stress, focus and energy levels, with a range from 1 to 5 and a default value of 3
    stress = wb1.slider("Stress today", min_value=1, max_value=5, value=3, help="1 = calm, 5 = very stressed")
    focus = wb2.slider("Focus level", min_value=1, max_value=5, value=3, help="1 = low focus, 5 = strong focus")
    energy = wb3.slider("Energy level", min_value=1, max_value=5, value=3, help="1 = low energy, 5 = high energy")
    wellbeing_concern = st.text_input("Study concerns", placeholder="Optional")
    st.markdown('</div>', unsafe_allow_html=True)

    # collecting all the uploaded files and typed notes and converting them into SourceDocument objects
    text_documents: list[SourceDocument] = []
    audio_paths: list[Path] = []
    video_paths: list[Path] = []
    image_paths: list[Path] = []
    # this is the temporary directory for saving uploaded files
    uploads_dir = temp_dir / "uploads"

    # for processing the uploaded files and typed notes
    if note_uploads:
        # for processing the uploaded text files and converting them into SourceDocument objects
        for idx, upload in enumerate(note_uploads, start=1):
            path = save_uploaded_file(upload, uploads_dir)
            try:
                doc = source_from_file(path, source_id=f"file_{idx}", limits=limits)
                if doc.text.strip():
                    text_documents.append(doc)
            # this is for any exceptions that may occur during the processing of the uploaded files
            except Exception as exc:
                st.warning(f"Could not read {upload.name}: {exc}")

    # for processing the typed notes and converting them into a SourceDocument object
    if typed_notes.strip():
        typed_doc = typed_source(typed_notes, "typed_notes", "Typed class notes", "typed_notes")
        if typed_doc is not None:
            text_documents.append(typed_doc)

# for processing the uploaded audio, video and image files and saving them to the temporary directory
    if audio_uploads:
        audio_paths.extend(save_uploaded_file(upload, uploads_dir) for upload in audio_uploads)
    if microphone_audio is not None:
        audio_paths.append(save_uploaded_file(microphone_audio, uploads_dir))
    if video_uploads:
        video_paths.extend(save_uploaded_file(upload, uploads_dir) for upload in video_uploads)
    if image_uploads:
        image_paths.extend(save_uploaded_file(upload, uploads_dir) for upload in image_uploads)

    # session review section for students to check what will be used before generating the learning pack
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="step">04</span><div><h3 style="margin:0">Session review</h3><p class="subtle" style="margin:.15rem 0 0">Check what will be used before generating the learning pack.</p></div></div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Text files", len(note_uploads or []))
    c2.metric("Audio", len(audio_paths))
    c3.metric("Videos", len(video_paths))
    c4.metric("Images", len(image_paths))
    c5.metric("Side notes", "Yes" if typed_notes.strip() else "No")
    c6.metric("Check-in", f"{stress}/{focus}/{energy}")
    st.markdown('</div>', unsafe_allow_html=True)

    # returning the collected inputs as a PipelineInputs object
    return PipelineInputs(
        text_documents=text_documents,
        audio_paths=audio_paths,
        video_paths=video_paths,
        image_paths=image_paths,
        side_notes="",
        student_task=student_task,
        wellbeing_checkin={"stress": stress, "focus": focus, "energy": energy, "concern": wellbeing_concern},
    )

# results of the MindationPipeline
def _support_badge(status: str) -> str:
    if status == "supported":
        return "Supported"
    if status == "partly supported":
        return "Partly supported"
    return "Needs review"

#results of the MindationPipeline in the Streamlit interface
def render_results(result: StudyOutput) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="step">05</span><div><h3 style="margin:0">Learning pack ready</h3><p class="subtle" style="margin:.15rem 0 0">Review the generated study materials below.</p></div></div>', unsafe_allow_html=True)

    # this is the metrics section for the generated learning pack
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sources used", len(result.submitted_sources))
    c2.metric("Evidence found", len(result.evidence))
    c3.metric("Flashcards", len(result.flashcards))
    c4.metric("Quiz questions", len(result.quiz))
    st.markdown('</div>', unsafe_allow_html=True)

    #
    # warnings section for any warnings that may have occurred
    if result.warnings:
        with st.expander("Setup notes", expanded=True):
            for warning in result.warnings:
                st.warning(warning)

# tabs section for the generated learning pack
    tabs = st.tabs(["Summary", "Transcript", "Study Plan", "Flashcards", "Quiz", "Evidence", "Support Check", "Downloads"])

    # summary section for any summary that may have been generated
    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Lecture Summary")
        st.write(result.summary or "Nil")
        st.subheader("Key Points")

        # this is the key points section for any key points that may have been generated
        if result.key_points:
            for point in result.key_points:
                st.markdown(f"- {point}")

        # this is the revision actions section for any revision actions that may have been generated
        else:
            st.info("Nil")
        st.subheader("Revision Actions")
        if result.revision_actions:
            for action in result.revision_actions:
                st.markdown(f"- {action}")
        else:
            st.info("Nil")
        st.markdown('</div>', unsafe_allow_html=True)

# transcript section for any transcript that may have been generated
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Transcript from audio or video")


        if result.transcript.text.strip():
            st.text_area("Transcript", result.transcript.text, height=330)
            if result.transcript.language:
                st.caption(f"Detected language: {result.transcript.language}")
            if result.transcript.duration_seconds:
                st.caption(f"Approximate duration processed: {result.transcript.duration_seconds:.1f} seconds")
        else:
            st.info("Nil - no transcript was produced. Record or upload audio and install Whisper if needed.")
        st.markdown('</div>', unsafe_allow_html=True)

# study plan section for any study plan that may have been generated
    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Study Plan")
        st.caption("This check-in is for pacing only. It is not medical advice!")

        if result.wellbeing_guidance:
            for item in result.wellbeing_guidance:
                st.markdown(f"- {item}")
        else:
            st.info("Nil")
        st.markdown('</div>', unsafe_allow_html=True)

# flashcards section for any flashcards that may have been generated
    with tabs[3]:
        # this is the flashcards section for any flashcards that may have been generated
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Flashcards")

        if result.flashcards:
            for idx, card in enumerate(result.flashcards, start=1):
                with st.expander(f"Card {idx}: {card['question']}"):
                    st.write(card["answer"])
        else:
            st.info("Nil")
        st.markdown('</div>', unsafe_allow_html=True)

# quiz section for any quiz questions that may have been generated
    with tabs[4]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Practice Quiz")
        if result.quiz:
            for item in result.quiz:
                st.markdown(f"**{item['question']}**")
                with st.expander("Expected answer"):
                    st.write(item["expected_answer"])
        else:
            st.info("Nil")
        st.markdown('</div>', unsafe_allow_html=True)

# evidence section for any evidence that may have been retrieved
    with tabs[5]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Evidence")
        if result.evidence:
            for idx, ev in enumerate(result.evidence, start=1):
                with st.expander(f"Evidence {idx}: {ev.title} | score {ev.score:.3f}"):
                    st.caption(f"Source type: {ev.source_type}")
                    st.write(ev.text)

        # this is for any evidence that may have been NOT retrieved
        else:
            st.info("Nil - no evidence retrieved.")
        st.markdown('</div>', unsafe_allow_html=True)
    # support check section for any support checks that may have been performed
    with tabs[6]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Support Check")
        st.caption("Claims are compared with retrieved evidence from this session.")
        if result.support_checks:
            for item in result.support_checks:
                with st.expander(f"{_support_badge(item.status)} | score {item.support_score:.3f} | {item.claim[:95]}"):
                    st.write(item.claim)
                    st.markdown(f"Closest evidence: {item.evidence_title}")
                    st.write(item.evidence_snippet)
        else:
            st.info("Nil")
        st.markdown('</div>', unsafe_allow_html=True)

# download section for the generated learning pack and full result
    with tabs[7]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Downloads")
        md = to_markdown(result)
        js = to_json(result)
        st.download_button("Download learning pack (.md)", md, file_name="mindation_learning_pack.md")
        st.download_button("Download full result (.json)", js, file_name="mindation_result.json")
        st.markdown('</div>', unsafe_allow_html=True)

# main function for the Streamlit app
def main() -> None:
    render_header()
    settings, limits = render_sidebar()

    # this is the temporary directory for saving uploaded files and processing them
    with tempfile.TemporaryDirectory(prefix="mindation_app_") as temp:
        inputs = collect_inputs(Path(temp), limits)
        run_clicked = st.button("Create Learning Pack", type="primary")

        # if the user clicks the "Create Learning Pack" button, run the MindationPipeline with the collected inputs and render the results
        if run_clicked:

            # this is a spinner to indicate that the learning pack is being created
            with st.spinner("Hold on, creating your learning pack..."):
                pipeline = MindationPipeline(settings=settings, limits=limits)
                result = pipeline.run(inputs)
            render_results(result)
        else:
            st.info("Add lecture material, then click Create Learning Pack!")

# entry point for the Streamlit app
if __name__ == "__main__":
    main()
