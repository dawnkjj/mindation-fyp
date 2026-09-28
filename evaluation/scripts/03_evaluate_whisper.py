# library
from pathlib import Path
import argparse
import csv
import re
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mindation.config import ModelSettings
from mindation.models.transcriber import WhisperTranscriber

# manually labelled
AUDIO_DIR = PROJECT_ROOT / "evaluation" / "data" / "audio"
RESULT_DIR = PROJECT_ROOT / "evaluation" / "results"

# clean texts
def normalise(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().split()

def edit_counts(ref, hyp):
    # stores tuple(total_distance, substitutions, deletions, insertions)
    n, m = len(ref), len(hyp)
    dp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0, 0)
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                sub_prev = dp[i-1][j-1]
                del_prev = dp[i-1][j]
                ins_prev = dp[i][j-1]
                candidates = [
                    (sub_prev[0]+1, sub_prev[1]+1, sub_prev[2], sub_prev[3]),
                    (del_prev[0]+1, del_prev[1], del_prev[2]+1, del_prev[3]),
                    (ins_prev[0]+1, ins_prev[1], ins_prev[2], ins_prev[3]+1),
                ]
                dp[i][j] = min(candidates, key=lambda x: x[0])
    return dp[n][m]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base")
    args = parser.parse_args()

    settings = ModelSettings(use_whisper=True, whisper_model=args.model)
    transcriber = WhisperTranscriber(settings)

    audio_files = sorted(
        p for p in AUDIO_DIR.iterdir()
        if p.suffix.lower() in {".wav",".mp3",".m4a",".flac",".ogg"}
    )
    if not audio_files:
        raise SystemExit(f"No audio files found in {AUDIO_DIR}")

    rows = []
    for audio in audio_files:
        reference_file = audio.with_suffix(".txt")
        if not reference_file.exists():
            print(f"Skipping {audio.name}: missing {reference_file.name}")
            continue

        reference = reference_file.read_text(encoding="utf-8")
        start = time.perf_counter()
        result = transcriber.transcribe(
            audio,
            # so that system can read audio simply
            language="en",
        )
        latency = time.perf_counter() - start

        if result.error:
            rows.append([audio.name,args.model,"",reference.strip(),"",0,0,0,0,"",latency,result.error])
            continue

        ref_tokens = normalise(reference)
        hyp_tokens = normalise(result.text)
        dist, subs, dels, ins = edit_counts(ref_tokens, hyp_tokens)
        wer = dist / len(ref_tokens) if ref_tokens else 0.0

        rows.append([
            audio.name,args.model,result.language or "",reference.strip(),result.text.strip(),
            len(ref_tokens),subs,dels,ins,wer,latency,""
        ])
        print(f"{audio.name}: WER={wer:.3f} latency={latency:.2f}s")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / f"whisper_{args.model}_english_results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["audio_file","model","language","reference","hypothesis","reference_words",
                    "substitutions","deletions","insertions","wer","processing_seconds","error"])
        w.writerows(rows)

    valid = [r for r in rows if r[9] != ""]
    if valid:
        mean_wer = sum(float(r[9]) for r in valid)/len(valid)
        mean_latency = sum(float(r[10]) for r in valid)/len(valid)
        print(f"\nMean WER: {mean_wer:.3f}")
        print(f"Mean processing time: {mean_latency:.2f}s")
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
