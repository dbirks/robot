#!/usr/bin/env python
"""Fixed-corpus STT bakeoff harness (issue #25, EPIC Phase 5).

Corpus: a directory of WAVs + manifest.jsonl (one {"file", "text"} per line),
recorded once on the robot box from the real mic - quiet, off-axis, and TV
clips (the EPIC's noise corpus). Compare candidates on IDENTICAL audio:

    uv run python benchmarks/stt_corpus.py corpus/ --model nvidia/parakeet-tdt-0.6b-v2
    uv run python benchmarks/stt_corpus.py corpus/ --model <candidate>

Reports WER, per-clip latency p50/p95, and short-utterance WER separately
(Parakeet's known sub-second weakness: 'stop' -> 'The Stob').
RULE (ADR 0006): fp32 only on this CPU - int8 is a measured REGRESSION.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def normalize(s: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() or c == " " else " " for c in s).split()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]


def wer(pairs) -> float:
    ref_words = sum(len(r) for r, _ in pairs)
    if not ref_words:
        return float("nan")
    return sum(edit_distance(r, h) for r, h in pairs) / ref_words


def transcribe_all(model_id: str, clips: list[Path]) -> tuple[list[str], list[float]]:
    import onnx_asr  # the stack's current runtime

    model = onnx_asr.load_model(model_id)
    texts, times = [], []
    for clip in clips:
        t0 = time.perf_counter()
        texts.append(model.recognize(str(clip)))
        times.append(time.perf_counter() - t0)
    return texts, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=3, help="repeat cold-cache clips")
    args = ap.parse_args()

    manifest = [json.loads(l) for l in (args.corpus / "manifest.jsonl").read_text().splitlines() if l]
    clips, refs = [], []
    for row in manifest:
        p = args.corpus / row["file"]
        if not p.exists():
            raise SystemExit(f"missing clip {p}")
        clips.append(p)
        refs.append(normalize(row["text"]))

    best_texts = None
    best_times: list[float] = []
    for _ in range(args.reps):
        texts, times = transcribe_all(args.model, clips)
        if best_texts is None or sum(times) < sum(best_times):
            best_texts, best_times = texts, times  # warmest run wins

    pairs = list(zip(refs, [normalize(t) for t in best_texts]))
    short = [(r, h) for clip, (r, h) in zip(clips, pairs) if len(r) <= 2]
    n50 = statistics.median(best_times)
    n95 = sorted(best_times)[max(0, int(len(best_times) * 0.95) - 1)]
    print(f"model: {args.model}")
    print(
        f"clips: {len(clips)}  WER: {wer(pairs):.3f}  short-utterance WER: {wer(short) if short else float('nan'):.3f}"
    )
    print(f"latency (wall per clip): p50={n50 * 1000:.0f}ms p95={n95 * 1000:.0f}ms")
    for clip, ref, text in zip(clips, refs, (normalize(t) for t in best_texts)):
        if edit_distance(ref, text):
            print(f"  {clip.name}: {' '.join(ref)!r} -> {' '.join(text)!r}")


if __name__ == "__main__":
    main()
