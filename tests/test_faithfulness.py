# test file for faithfulness.py
from mindation.models.faithfulness import check_summary_support
from mindation.types import EvidenceChunk

# test that check_summary_support flags grounded claims correctly
def test_support_check_flags_grounded_claims():
    # create evidence with a relevant claim
    evidence = [
        EvidenceChunk(
            source_id="typed",
            source_type="typed_notes",
            title="Class notes",
            text="The lecturer explained that retrieval evaluation can use Precision at K and mean reciprocal rank.",
            score=0.8,
        )
    ]
    checks = check_summary_support(
        "Retrieval evaluation can use Precision at K and mean reciprocal rank.", evidence
    )
    assert checks
    assert checks[0].status in {"supported", "partly supported"}
