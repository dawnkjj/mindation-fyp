# evaluate the frozen relevance gate on unseen test queries

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# mannually labelled paths
CHUNKS_PATH = Path("evaluation/results/chunks.csv")
QUERIES_PATH = Path("evaluation/data/relevance_test_queries.csv")
OUTPUT_PATH = Path("evaluation/results/relevance_test_results.csv")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# using development data
RELEVANCE_THRESHOLD = 0.3376


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def find_text_column(
    rows: list[dict[str, str]],
) -> str:
    for column in ("text", "chunk_text", "content"):
        if column in rows[0]:
            return column

    raise ValueError(
        f"Cannot find text column. "
        f"Columns: {list(rows[0].keys())}"
    )


def main() -> None:
    chunks = read_csv(CHUNKS_PATH)
    queries = read_csv(QUERIES_PATH)

    text_column = find_text_column(chunks)

    texts = [
        row[text_column].strip()
        for row in chunks
        if row.get(text_column, "").strip()
    ]

    model = SentenceTransformer(
        MODEL_NAME,
        token=False,
    )

    print(
        f"Encoding {len(texts)} lecture chunks..."
    )

    document_embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    results: list[dict[str, object]] = []

    supported_total = 0
    supported_accepted = 0

    unsupported_total = 0
    unsupported_rejected = 0

    print(
        "\nFrozen relevance threshold: "
        f"{RELEVANCE_THRESHOLD:.4f}\n"
    )

    for row in queries:
        query_id = row["query_id"].strip()
        query = row["query"].strip()
        query_type = row["type"].strip().lower()

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = np.dot(
            document_embeddings,
            query_embedding,
        )

        ranked = np.argsort(scores)[::-1]

        top3 = ranked[:3]

        top1_score = float(
            scores[ranked[0]]
        )

        top3_mean = float(
            np.mean(scores[top3])
        )

        accepted = (
            top3_mean >= RELEVANCE_THRESHOLD
        )

        if query_type == "supported":
            supported_total += 1

            if accepted:
                supported_accepted += 1
                correct = True
            else:
                correct = False

        elif query_type == "unsupported":
            unsupported_total += 1

            if not accepted:
                unsupported_rejected += 1
                correct = True
            else:
                correct = False

        else:
            raise ValueError(
                f"Unknown query type: {query_type}"
            )

        decision = (
            "accept"
            if accepted
            else "reject"
        )

        print(
            f"{query_id:<4} "
            f"{query_type:<11} "
            f"top1={top1_score:.4f} "
            f"top3mean={top3_mean:.4f} "
            f"{decision:<6} "
            f"{'PASS' if correct else 'FAIL'}"
        )

        results.append(
            {
                "query_id": query_id,
                "query": query,
                "type": query_type,
                "top1_score": top1_score,
                "top3_mean": top3_mean,
                "threshold": RELEVANCE_THRESHOLD,
                "decision": decision,
                "correct": correct,
            }
        )

    supported_acceptance = (
        supported_accepted / supported_total
    )

    unsupported_rejection = (
        unsupported_rejected / unsupported_total
    )

    false_refusal_rate = (
        1.0 - supported_acceptance
    )

    balanced_accuracy = (
        supported_acceptance
        + unsupported_rejection
    ) / 2.0

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nFinal relevance-gate test")
    print("-------------------------")

    print(
        "Supported acceptance rate: "
        f"{supported_acceptance:.3f}"
    )

    print(
        "Unsupported rejection rate: "
        f"{unsupported_rejection:.3f}"
    )

    print(
        "False refusal rate: "
        f"{false_refusal_rate:.3f}"
    )

    print(
        "Balanced accuracy: "
        f"{balanced_accuracy:.3f}"
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()