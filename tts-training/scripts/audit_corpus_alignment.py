"""Full-corpus transcript/audio alignment audit for the She corpus.

For every clip: transcribe the audio with faster-whisper large-v3 and compare
against the manifest text. Reports WER plus two corruption signatures that the
earlier foreign-character filter cannot see:

  * inflated manifest  - the manifest repeats a phrase the audio says once
  * missing manifest   - the audio contains speech the manifest omits

Run with the TTS service STOPPED so the GPU has room for whisper.
"""
import csv
import difflib
import json
import os
import re
import statistics
import sys
from collections import Counter

from faster_whisper import WhisperModel

MANIFEST = "/home/voiceuser/.hermes/cache/voices/she-xtts/manifest.csv"
CLIPS = "/home/voiceuser/.hermes/cache/voices/she-xtts/clips"
OUT = "/home/voiceuser/projects/gpt-sovits/reports/corpus-alignment-audit.json"
MODEL_DIR = "/home/voiceuser/projects/gpt-sovits/.cache/whisper-large-v3"


def norm(text):
    return [
        w.lower().strip(".,!?\"';:")
        for w in text.replace("—", " ").replace("-", " ").split()
        if w.strip(".,!?\"';:")
    ]


def wer(want, got):
    a, b = norm(want), norm(got)
    if not a:
        return 0.0 if not b else 1.0
    return 1 - difflib.SequenceMatcher(None, a, b).ratio()


def repeated_sentences(text, n=3):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return [(s, c) for s, c in Counter(sents).items() if c >= n]


def main():
    rows = list(csv.DictReader(open(MANIFEST)))
    print(f"auditing {len(rows)} clips", flush=True)
    model = WhisperModel(MODEL_DIR, device="cuda", compute_type="float16",
                         download_root="/home/voiceuser/projects/gpt-sovits/.cache/whisper")
    results = []
    for i, r in enumerate(rows):
        path = os.path.join(CLIPS, r["filename"])
        segs, _ = model.transcribe(path, beam_size=5)
        got = " ".join(s.text.strip() for s in segs).strip()
        w = wer(r["text"], got)
        reps = repeated_sentences(r["text"])
        # a manifest longer than the audio can hold is inflated
        cps = len(r["text"]) / float(r["duration_seconds"])
        results.append({
            "filename": r["filename"],
            "index": int(r["index"]),
            "duration_seconds": float(r["duration_seconds"]),
            "manifest_chars": len(r["text"]),
            "chars_per_second": round(cps, 2),
            "wer": round(w, 4),
            "repeated_sentences": [(s[:70], c) for s, c in reps],
            "manifest_text": r["text"],
            "asr_text": got,
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    wers = [x["wer"] for x in results]
    bad = [x for x in results if x["wer"] > 0.25]
    inflated = [x for x in results if x["repeated_sentences"]]
    print()
    print(f"  clips audited          : {len(results)}")
    print(f"  median WER             : {statistics.median(wers):.4f}")
    print(f"  mean WER               : {statistics.mean(wers):.4f}")
    print(f"  clips WER > 0.25       : {len(bad)} ({len(bad)/len(results)*100:.2f}%)")
    print(f"  clips WER > 0.50       : {len([x for x in results if x['wer'] > 0.5])}")
    print(f"  repeated-sentence clips: {len(inflated)}")
    print(f"  median cps             : {statistics.median([x['chars_per_second'] for x in results]):.2f}")

    json.dump({
        "clips_audited": len(results),
        "median_wer": statistics.median(wers),
        "mean_wer": statistics.mean(wers),
        "clips_wer_over_0_25": len(bad),
        "clips_wer_over_0_50": len([x for x in results if x["wer"] > 0.5]),
        "clips_with_repeated_sentences": len(inflated),
        "per_clip": results,
    }, open(OUT, "w"), indent=2)
    print(f"\n  wrote {OUT}")
    print("\n  worst 15 by WER:")
    for x in sorted(results, key=lambda z: -z["wer"])[:15]:
        print(f"    {x['wer']:.3f}  {x['filename']}  {x['asr_text'][:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
