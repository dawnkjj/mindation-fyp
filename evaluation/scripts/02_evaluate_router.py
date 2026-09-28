# library
from pathlib import Path
import csv
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import accuracy_score, f1_score
from mindation.config import ModelSettings
from mindation.models.router import IntentRouter, ROUTE_LABELS

# manually labelled prompts
DATA = PROJECT_ROOT / "evaluation" / "data" / "routing_queries.csv"

# stores the prediction fpr everyy routing prompts
OUT_PRED = PROJECT_ROOT / "evaluation" / "results" / "routing_predictions.csv"

# stores the final accuracy
OUT_METRICS = PROJECT_ROOT / "evaluation" / "results" / "routing_metrics.csv"

# load all manually labelled routing prompts
def load_rows():
    with DATA.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    #every expected label matches one of the task categories
    valid = set(ROUTE_LABELS)
    bad = [(r["prompt_id"], r["expected_label"]) for r in rows if r["expected_label"] not in valid]
    if bad:
        raise SystemExit(
            "Invalid expected router label(s): " + str(bad) +
            "\nValid labels are: " + ", ".join(ROUTE_LABELS)
        )
    return rows

#create the same IntentRouter
# changing use bart router allows the keyboard and bart approaches
def evaluate(name, use_bart, rows):
    router = IntentRouter(ModelSettings(use_bart_router=use_bart))
    y_true, y_pred = [], []
    outputs = []
    latencies = []
    methods_used = set()

    for row in rows:

        # measure the routing decision for one student request
        start = time.perf_counter()

        # routing from the request rather than the lecture evidence
        result = router.route(row["prompt"], "")
        latency = time.perf_counter() - start
        y_true.append(row["expected_label"])
        y_pred.append(result.label)
        latencies.append(latency)

        # record actual routing method
        methods_used.add(result.method)
        outputs.append([
            name, row["prompt_id"], row["prompt"], row["expected_label"],
            result.label, result.confidence, result.method, latency
        ])

    # accuracy measures the overall proportion of correct predictions
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=ROUTE_LABELS, average="macro", zero_division=0)
    mean_latency = sum(latencies) / len(latencies)
    return outputs, [name, accuracy, macro_f1, mean_latency, ";".join(sorted(methods_used))]

#
def main():
    #one labelled dataset for both routers so the comparison is fair
    rows = load_rows()
    all_predictions = []
    metrics = []

    #evaluate the lightweight keyword baseline
    pred, met = evaluate("Keyword", False, rows)
    all_predictions.extend(pred)
    metrics.append(met)

    pred, met = evaluate("BART requested", True, rows)
    all_predictions.extend(pred)
    metrics.append(met)

    # CREATE RESULT FOLDER
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    #save individual prediction
    with OUT_PRED.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment","prompt_id","prompt","expected_label","predicted_label",
                    "confidence","actual_method_used","latency_seconds"])
        w.writerows(all_predictions)

    # save summaries
    with OUT_METRICS.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment","accuracy","macro_f1","mean_latency_seconds","actual_methods_used"])
        w.writerows(metrics)

    print("\nRouting summary")
    for m in metrics:
        print(
            f"{m[0]:15s} accuracy={m[1]:.3f} macroF1={m[2]:.3f} "
            f"latency={m[3]:.4f}s actual_method={m[4]}"
        )
    print("\nIMPORTANT: If 'BART requested' says actual_method=keyword-fallback, "
          "BART did not load and must NOT be reported as a BART result.")

if __name__ == "__main__":
    main()
