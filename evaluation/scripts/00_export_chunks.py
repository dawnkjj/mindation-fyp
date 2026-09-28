from pathlib import Path
import csv
import sys

# root of the Mindation project so this evaluation script can import
# the same application modules used by the main system
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mindation.config import AppLimits
from mindation.ingestion import source_from_file
from mindation.models.retriever import chunk_text

# files used to build the retrieval evaluation dataset
SOURCE_DIR = PROJECT_ROOT / "evaluation" / "data" / "educational_sources"
OUT = PROJECT_ROOT / "evaluation" / "results" / "chunks.csv"

def main():

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # include source formats supported by the text-ingestion pipeline
    paths = sorted(
        p for p in SOURCE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"}
    )

     # stop early if the evaluation folder does not contain usable sources.\
    if not paths:
        raise SystemExit(f"No PDF/TXT/MD files found in {SOURCE_DIR}")

    limits = AppLimits()
    for path in paths:
        source_id = path.stem

        # Convert each source file into the common SourceDocument format
        doc = source_from_file(path, source_id=source_id, limits=limits)
        chunks = chunk_text(doc.text, limits.chunk_size, limits.chunk_overlap)

         # Give every chunk a stable ID so relevant chunks can be manually
        for i, text in enumerate(chunks, start=1):
            chunk_id = f"{source_id}::chunk_{i:03d}"
            rows.append([chunk_id, source_id, path.name, i, text])

# save the generated chunks in CSV format for manual labelling
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id","source_id","file_name","chunk_number","text"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} chunks to {OUT}")
    print("Next: manually label the correct chunk ID(s) in retrieval_queries.csv.")

if __name__ == "__main__":
    main()
