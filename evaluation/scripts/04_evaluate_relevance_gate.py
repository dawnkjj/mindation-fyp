# evaluate query-to-evidence relevance and select a development threshold

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def read_csv(path: Path) -> list[dict[str, str]]:
    # read a CSV file into a list of dictionaries
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def find_text_column(rows: list[dict[str, str]]) -> str:
    #Locate the chunk-text column in chunks.csv
    if not rows:
        raise ValueError("chunks.csv contains no rows.")

    possible_names = (
        "text",
        "chunk_text",
        "content",
    )

    for name in possible_names:
        if name in rows[0]:
            return name

    raise ValueError(
        # Could not find chunk text column
        f"Available columns: {list(rows[0].keys())}"
    )


def find_chunk_id_column(rows: list[dict[str, str]]) -> str | None:
    #locate an optional chunk identifier column
    possible_names = (
        "chunk_id",
        "id",
        "evidence_id",
    )

    for name in possible_names:
        if name in rows[0]:
            return name

    return None


def calculate_threshold_metrics(
    rows: list[dict[str, object]],
    threshold: float,
) -> dict[str, float]:
    #calculate supported acceptance and unsupported rejection metrics

    supported = [
        row for row in rows
        if row["type"] == "supported"
    ]

    unsupported = [
        row for row in rows
        if row["type"] == "unsupported"
    ]

    supported_accepted = sum(
        float(row["top1_score"]) >= threshold
        for row in supported
    )

    unsupported_rejected = sum(
        float(row["top1_score"]) < threshold
        for row in unsupported
    )

    supported_accept_rate = (
        supported_accepted / len(supported)
        if supported else 0.0
    )

    unsupported_rejection_rate = (
        unsupported_rejected / len(unsupported)
        if unsupported else 0.0
    )

    false_refusal_rate = 1.0 - supported_accept_rate
    false_accept_rate = 1.0 - unsupported_rejection_rate

    balanced_accuracy = (
        supported_accept_rate + unsupported_rejection_rate
    ) / 2.0

    return {
        "threshold": threshold,
        "supported_accept_rate": supported_accept_rate,
        "unsupported_rejection_rate": unsupported_rejection_rate,
        "false_refusal_rate": false_refusal_rate,
        "false_accept_rate": false_accept_rate,
        "balanced_accuracy": balanced_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("evaluation/results/chunks.csv"),
        help="Path to chunks.csv",
    )

    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("evaluation/data/relevance_dev_queries.csv"),
        help="Path to development queries",
    )

    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
    )

    args = parser.parse_args()

    chunk_rows = read_csv(args.chunks)
    query_rows = read_csv(args.queries)

    text_column = find_text_column(chunk_rows)
    chunk_id_column = find_chunk_id_column(chunk_rows)

    chunk_texts = [
        row[text_column].strip()
        for row in chunk_rows
        if row.get(text_column, "").strip()
    ]

    if not chunk_texts:
        raise ValueError("No usable chunks found.")

    print(f"Loading SentenceTransformer: {args.model}")

    model = SentenceTransformer(
        args.model,
        token=False,
    )

    print(f"Encoding {len(chunk_texts)} lecture chunks...")

    chunk_embeddings = model.encode(
        chunk_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    result_rows: list[dict[str, object]] = []

    print(f"Evaluating {len(query_rows)} development queries...\n")

    for query_row in query_rows:
        query_id = query_row["query_id"].strip()
        query = query_row["query"].strip()
        query_type = query_row["type"].strip().lower()

        if query_type not in {"supported", "unsupported"}:
            raise ValueError(
                f"{query_id}: invalid type '{query_type}'"
            )

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = np.dot(
            chunk_embeddings,
            query_embedding,
        )

        ranked_indices = np.argsort(scores)[::-1]

        best_index = int(ranked_indices[0])
        top1_score = float(scores[best_index])

        top3_indices = ranked_indices[:3]
        top5_indices = ranked_indices[:5]

        top3_mean = float(
            np.mean(scores[top3_indices])
        )

        top5_mean = float(
            np.mean(scores[top5_indices])
        )

        if chunk_id_column:
            best_chunk_id = chunk_rows[best_index][chunk_id_column]
        else:
            best_chunk_id = str(best_index)

        result_rows.append(
            {
                "query_id": query_id,
                "query": query,
                "type": query_type,
                "top1_score": top1_score,
                "top3_mean": top3_mean,
                "top5_mean": top5_mean,
                "best_chunk_id": best_chunk_id,
                "best_chunk_text": chunk_texts[best_index][:300],
            }
        )

        print(
            f"{query_id:<4} "
            f"{query_type:<11} "
            f"top1={top1_score:.4f} "
            f"top3mean={top3_mean:.4f}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores_path = (
        args.output_dir
        / "relevance_dev_scores.csv"
    )

    with scores_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(result_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(result_rows)

    # candidate thresholds are placed between observed top-1 scores
    unique_scores = sorted(
        {
            float(row["top1_score"])
            for row in result_rows
        }
    )

    candidate_thresholds = [0.0]

    for left, right in zip(
        unique_scores,
        unique_scores[1:],
    ):
        candidate_thresholds.append(
            (left + right) / 2.0
        )

    candidate_thresholds.append(1.0)

    metric_rows = [
        calculate_threshold_metrics(
            result_rows,
            threshold,
        )
        for threshold in candidate_thresholds
    ]

    # selection rule, Highest balanced accuracy, lowest false-refusal rate , lower threshold.
    best = max(
        metric_rows,
        key=lambda row: (
            row["balanced_accuracy"],
            -row["false_refusal_rate"],
            -row["threshold"],
        ),
    )

    thresholds_path = (
        args.output_dir
        / "relevance_threshold_candidates.csv"
    )

    with thresholds_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(metric_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    print("\nDevelopment threshold selection")
    print("--------------------------------")
    print(
        f"Selected threshold: "
        f"{best['threshold']:.4f}"
    )
    print(
        "Supported acceptance rate: "
        f"{best['supported_accept_rate']:.3f}"
    )
    print(
        "Unsupported rejection rate: "
        f"{best['unsupported_rejection_rate']:.3f}"
    )
    print(
        "False refusal rate: "
        f"{best['false_refusal_rate']:.3f}"
    )
    print(
        "Balanced accuracy: "
        f"{best['balanced_accuracy']:.3f}"
    )

    print(f"\nSaved: {scores_path}")
    print(f"Saved: {thresholds_path}")


if __name__ == "__main__":
    main()