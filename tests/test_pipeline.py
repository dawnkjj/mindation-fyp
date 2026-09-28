# test file for pipeline.py
from mindation.config import ModelSettings
from mindation.pipeline import MindationPipeline
from mindation.types import PipelineInputs, SourceDocument

# test that the pipeline returns a summary and evidence when given typed input
def test_pipeline_returns_nil_without_input():

    # create a MindationPipeline with whisper disabled
    pipeline = MindationPipeline(settings=ModelSettings(use_whisper=False))
    result = pipeline.run(PipelineInputs())
    assert result.summary.startswith("Nil")
    assert result.evidence == []

# test that the pipeline uses typed input only and returns a summary and evidence
def test_pipeline_uses_typed_input_only():

    # create a MindationPipeline with whisper disabled
    pipeline = MindationPipeline(settings=ModelSettings(use_whisper=False))
    inputs = PipelineInputs(
        text_documents=[
            SourceDocument(
                "typed",
                "typed_notes",
                "Class notes",
                "The lecturer explained that retrieval evaluation can use Precision at K and mean reciprocal rank.",
            )
        ],
        student_task="Create revision notes about retrieval evaluation",
    )
    # run the pipeline with typed input
    result = pipeline.run(inputs)
    assert "retrieval" in result.summary.lower()
    assert result.evidence
    assert result.submitted_sources[0].title == "Class notes"
