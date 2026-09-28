#evaluate support checker

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# make the project root importable when this script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mindation.models.faithfulness import check_summary_support


# paths manually labelled

INPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "data"
    / "support_checker_cases.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "figures"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "support_checker_predictions.csv"
)

METRICS_PATH = (
    RESULTS_DIR
    / "support_checker_metrics.csv"
)

CONFUSION_MATRIX_PATH = (
    FIGURES_DIR
    / "support_checker_confusion_matrix.png"
)


LABELS = [
    "supported",
    "partly supported",
    "needs review",
]

# minimal evidence object required by the support checker
class EvaluationEvidence:

    def __init__(self, text: str, title: str) -> None:
        self.text = text
        self.title = title

# load manually labelled support-checking cases
def load_cases(path: Path) -> list[dict[str, str]]:

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find evaluation dataset: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(
            "The support-checker evaluation dataset is empty."
        )

    required_columns = {
        "case_id",
        "claim",
        "evidence_text",
        "expected_label",
        "source",
        "manual_reason",
    }

    missing_columns = (
        required_columns - set(rows[0].keys())
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    return rows

# run every claim/evidence pair through the production checker
def evaluate_cases(
    rows: list[dict[str, str]],
) -> list[dict[str, object]]:

    results: list[dict[str, object]] = []

    for row in rows:
        case_id = row["case_id"].strip()
        claim = row["claim"].strip()
        evidence_text = row["evidence_text"].strip()

        expected_label = (
            row["expected_label"]
            .strip()
            .lower()
        )

        if expected_label not in LABELS:
            raise ValueError(
                f"{case_id}: invalid expected label "
                f"'{expected_label}'."
            )

        evidence = [
            EvaluationEvidence(
                text=evidence_text,
                title=row["source"].strip()
                or case_id,
            )
        ]

        # claim is passed as a one-claim summary
        checks = check_summary_support(
            claim,
            evidence,
        )

        if not checks:
            raise RuntimeError(
                f"{case_id}: support checker returned no result. "
                "Check whether the claim is shorter than the "
                "35-character minimum used by split_claims()."
            )

        check = checks[0]

        predicted_label = (
            check.status
            .strip()
            .lower()
        )

        score = float(
            check.support_score
        )

        correct = (
            predicted_label
            == expected_label
        )

        results.append(
            {
                "case_id": case_id,
                "claim": claim,
                "expected_label": expected_label,
                "predicted_label": predicted_label,
                "support_score": score,
                "correct": correct,
                "source": row["source"],
                "manual_reason": row[
                    "manual_reason"
                ],
                "matched_evidence_title": (
                    check.evidence_title
                ),
                "matched_evidence_snippet": (
                    check.evidence_snippet
                ),
            }
        )

        outcome = (
            "PASS"
            if correct
            else "FAIL"
        )

        print(
            f"{case_id:<5} "
            f"expected={expected_label:<18} "
            f"predicted={predicted_label:<18} "
            f"score={score:.4f} "
            f"{outcome}"
        )

    return results


def save_predictions(
    results: list[dict[str, object]],
    path: Path,
) -> None:
    #save case-level predictions

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "case_id",
        "claim",
        "expected_label",
        "predicted_label",
        "support_score",
        "correct",
        "source",
        "manual_reason",
        "matched_evidence_title",
        "matched_evidence_snippet",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def calculate_metrics(
    results: list[dict[str, object]],
) -> dict[str, float]:
    #calculate overall and per-class metrics

    y_true = [
        str(item["expected_label"])
        for item in results
    ]

    y_pred = [
        str(item["predicted_label"])
        for item in results
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
        )
    )

    macro_f1 = float(
        np.mean(f1)
    )

    metrics: dict[str, float] = {
        "overall_accuracy": float(
            accuracy
        ),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "macro_f1": macro_f1,
    }

    for index, label in enumerate(
        LABELS
    ):
        key = label.replace(
            " ",
            "_",
        )

        metrics[
            f"{key}_precision"
        ] = float(
            precision[index]
        )

        metrics[
            f"{key}_recall"
        ] = float(
            recall[index]
        )

        metrics[
            f"{key}_f1"
        ] = float(
            f1[index]
        )

        metrics[
            f"{key}_support"
        ] = float(
            support[index]
        )

    return metrics


def save_metrics(
    metrics: dict[str, float],
    path: Path,
) -> None:
    #save summary metrics

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "metric",
                "value",
            ]
        )

        for metric, value in (
            metrics.items()
        ):
            writer.writerow(
                [
                    metric,
                    value,
                ]
            )


def create_confusion_matrix(
    results: list[dict[str, object]],
    output_path: Path,
) -> None:
    #create and save confusion matrix figure

    y_true = [
        str(item["expected_label"])
        for item in results
    ]

    y_pred = [
        str(item["predicted_label"])
        for item in results
    ]

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    image = ax.imshow(matrix)

    fig.colorbar(
        image,
        ax=ax,
    )

    ax.set_xticks(
        range(len(LABELS))
    )

    ax.set_yticks(
        range(len(LABELS))
    )

    ax.set_xticklabels(
        LABELS,
        rotation=25,
        ha="right",
    )

    ax.set_yticklabels(
        LABELS
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "Manual label"
    )

    ax.set_title(
        "Mindation Support-Checker Confusion Matrix"
    )

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            matrix.shape[1]
        ):
            ax.text(
                column,
                row,
                str(
                    matrix[
                        row,
                        column,
                    ]
                ),
                ha="center",
                va="center",
            )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def print_summary(
    metrics: dict[str, float],
) -> None:
    #print evaluation summary

    print(
        "\nSupport-checker evaluation"
    )

    print(
        "Overall accuracy: "
        f"{metrics['overall_accuracy']:.3f}"
    )

    print(
        "Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.3f}"
    )

    print(
        "Macro F1: "
        f"{metrics['macro_f1']:.3f}"
    )

    print(
        "\nPer-class recall"
    )

    print(
        "Supported: "
        f"{metrics['supported_recall']:.3f}"
    )

    print(
        "Partly supported: "
        f"{metrics['partly_supported_recall']:.3f}"
    )

    print(
        "Needs review: "
        f"{metrics['needs_review_recall']:.3f}"
    )


def main() -> None:
    # run support-checker evaluation

    rows = load_cases(
        INPUT_PATH
    )

    print(
        f"Loaded {len(rows)} "
        "support-checker cases.\n"
    )

    results = evaluate_cases(
        rows
    )

    metrics = calculate_metrics(
        results
    )

    save_predictions(
        results,
        PREDICTIONS_PATH,
    )

    save_metrics(
        metrics,
        METRICS_PATH,
    )

    create_confusion_matrix(
        results,
        CONFUSION_MATRIX_PATH,
    )

    print_summary(
        metrics
    )

    print(
        "\nSaved predictions to:"
    )

    print(
        PREDICTIONS_PATH
    )

    print(
        "\nSaved metrics to:"
    )

    print(
        METRICS_PATH
    )

    print(
        "\nSaved confusion matrix to:"
    )

    print(
        CONFUSION_MATRIX_PATH
    )


if __name__ == "__main__":
    main()