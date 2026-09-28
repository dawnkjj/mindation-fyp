# library
from pathlib import Path
import csv
import sys
import time
import math
import numpy as np

# locate project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# manually labelled retrieval questions
QUERIES = PROJECT_ROOT / "evaluation" / "data" / "retrieval_queries.csv"
# evidence chunks generated
CHUNKS = PROJECT_ROOT / "evaluation" / "results" / "chunks.csv"
# stores the top ranked chunks
OUT_RANK = PROJECT_ROOT / "evaluation" / "results" / "retrieval_rankings.csv"
# stores the final average retrieval metrics
OUT_METRICS = PROJECT_ROOT / "evaluation" / "results" / "retrieval_metrics.csv"

# load evidence chunks
def load_chunks():
    with CHUNKS.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))

# load the manually labelled evaluation questions
def load_queries():
    with QUERIES.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # each query must have at least one relevant chunk
    missing = [r["query_id"] for r in rows if not r["gold_chunk_ids"].strip()]
    if missing:
        raise SystemExit(
            "Gold labels are missing for: " + ", ".join(missing) +
            "\nRun 00_export_chunks.py and manually fill gold_chunk_ids first."
        )
    return rows

# depending on the position the relevant result is ranked
def reciprocal_rank(ranked_ids, gold):
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0

# measures how much of the relevant evidence appears
def recall_at_k(ranked_ids, gold, k):
    if not gold:
        return 0.0
    return len(set(ranked_ids[:k]) & gold) / len(gold)

# build one tf-idf model for all the evidence chunks
def evaluate_tfidf(chunks, queries):
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)
    doc_matrix = vectorizer.fit_transform(texts)
    rows = []
    metric_rows = []
    for q in queries:
        # measures the query retrieval stage
        start = time.perf_counter()
        qv = vectorizer.transform([q["query"]])

        # tf-idf retrieval uses cosine similarity
        scores = cosine_similarity(qv, doc_matrix).flatten()
        latency = time.perf_counter() - start

        # sort the chunk indexes from highest to lowest
        order = np.argsort(scores)[::-1]
        ranked_ids = [chunks[i]["chunk_id"] for i in order]

        # multiple correct chunks can be stored using semi colon
        gold = {x.strip() for x in q["gold_chunk_ids"].split(";") if x.strip()}

        # store the top 5 chunks
        for rank, idx in enumerate(order[:5], start=1):
            rows.append([
                "TF-IDF", q["query_id"], rank, chunks[idx]["chunk_id"],
                float(scores[idx]), latency
            ])

            # calculate retrieval quality
        metric_rows.append([
            "TF-IDF", q["query_id"],
            recall_at_k(ranked_ids, gold, 1),
            recall_at_k(ranked_ids, gold, 3),
            recall_at_k(ranked_ids, gold, 5),
            reciprocal_rank(ranked_ids, gold),
            latency
        ])
    return rows, metric_rows

# sentence transformer
def evaluate_sbert(chunks, queries):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise SystemExit(
            "sentence-transformers is not available. Install requirements-ai.txt. "
            f"Details: {exc}"
        )

    # load the model used for retrieval
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name, token=False)
    texts = [c["text"] for c in chunks]
    # model load and document embedding are setup costs, not per-query latency.
    doc_emb = model.encode(texts, normalize_embeddings=True)

    rows = []
    metric_rows = []
    for q in queries:
        start = time.perf_counter()

        # encode the query using same embedding model
        q_emb = model.encode([q["query"]], normalize_embeddings=True)[0]

        # both sets of embedding are normalised
        scores = np.dot(doc_emb, q_emb)
        latency = time.perf_counter() - start

        #rank chunks from most to least similar
        order = np.argsort(scores)[::-1]
        ranked_ids = [chunks[i]["chunk_id"] for i in order]
        gold = {x.strip() for x in q["gold_chunk_ids"].split(";") if x.strip()}

        # save top 5 examine instead of relying
        for rank, idx in enumerate(order[:5], start=1):
            rows.append([
                "SentenceTransformer", q["query_id"], rank,
                chunks[idx]["chunk_id"], float(scores[idx]), latency
            ])
        metric_rows.append([
            "SentenceTransformer", q["query_id"],
            recall_at_k(ranked_ids, gold, 1),
            recall_at_k(ranked_ids, gold, 3),
            recall_at_k(ranked_ids, gold, 5),
            reciprocal_rank(ranked_ids, gold),
            latency
        ])
    return rows, metric_rows

# calculate average result
def summarise(metric_rows):
    methods = sorted({r[0] for r in metric_rows})
    summary = []
    for method in methods:
        subset = [r for r in metric_rows if r[0] == method]
        n = len(subset)

        # columns 2-6 contains recall
        means = [sum(float(r[i]) for r in subset) / n for i in range(2, 7)]
        summary.append([method, n, *means])
    return summary

# load same chunks and queries
def main():
    chunks = load_chunks()
    queries = load_queries()
    ranking_rows = []
    metric_rows = []

    for fn in [evaluate_tfidf, evaluate_sbert]:
        rr, mm = fn(chunks, queries)
        ranking_rows.extend(rr)
        metric_rows.extend(mm)

    #save detailed rankings for error analysis
    OUT_RANK.parent.mkdir(parents=True, exist_ok=True)
    with OUT_RANK.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method","query_id","rank","chunk_id","score","query_latency_seconds"])
        w.writerows(ranking_rows)

    #average the per-query metrics to produce the results reported
    summary = summarise(metric_rows)
    with OUT_METRICS.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method","n_queries","mean_recall_at_1","mean_recall_at_3",
                    "mean_recall_at_5","mrr","mean_query_latency_seconds"])
        w.writerows(summary)

    print("\nRetrieval summary")
    for row in summary:
        print(
            f"{row[0]:20s} n={row[1]} "
            f"R@1={row[2]:.3f} R@3={row[3]:.3f} "
            f"R@5={row[4]:.3f} MRR={row[5]:.3f} "
            f"latency={row[6]:.4f}s"
        )
    print(f"\nSaved {OUT_METRICS}")
    print(f"Saved {OUT_RANK}")

if __name__ == "__main__":
    main()
