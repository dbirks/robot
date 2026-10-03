"""Open-vocabulary keyword spotting via sherpa-onnx (ADR 0003 fast path).

"Reachy", "hey Reachy", "robot", "hey robot" - no LLM, no STT, sub-100ms
reaction class. Model/runtime pinned in docs/pins.yaml; install via
scripts/get_kws_model.sh. sherpa-onnx is an optional extra: this module
imports it lazily so the shell runs (without wake words) on a machine that
hasn't installed the extra yet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KeywordHit:
    keyword: str
    score: float


class KeywordSpotter:
    def __init__(
        self,
        model_dir: str,
        *,
        thresholds: dict[str, float] | None = None,
        default_threshold: float = 0.30,
        sample_rate: int = 16000,
    ) -> None:
        import sherpa_onnx  # ty: ignore[unresolved-import] lazy: optional extra

        self.sample_rate = sample_rate
        self._spotter = sherpa_onnx.KeywordSpotter(
            **_sherpa_config(model_dir, sample_rate),
            keywords_file=_keywords_file(model_dir, thresholds or {}, default_threshold),
        )
        self._stream = self._spotter.create_stream()

    def process(self, pcm: bytes) -> list[KeywordHit]:
        import numpy as np

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._stream.accept_waveform(self.sample_rate, samples)
        hits: list[KeywordHit] = []
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            key = self._spotter.get_result(self._stream)
            if key:
                # Thresholds are applied inside the decoder (per-keyword "#t"
                # in the keywords file); sherpa reports no score, so a hit
                # that surfaces has already cleared its bar.
                hits.append(KeywordHit(keyword=key.replace("_", " "), score=1.0))
                self._spotter.reset_stream(self._stream)
        return hits

    def reset(self) -> None:
        self._stream = self._spotter.create_stream()


def _sherpa_config(model_dir: str, sample_rate: int) -> dict:
    """sherpa-onnx transducer KWS paths (flat kwargs). The model dir is created
    by scripts/get_kws_model.sh and pinned in docs/pins.yaml. fp32, not the
    int8 files: int8 GEMMs regress on this AVX2-only CPU (ADR 0004)."""
    from pathlib import Path

    p = Path(model_dir)
    return {
        "encoder": str(p / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        "decoder": str(p / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        "joiner": str(p / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
        "tokens": str(p / "tokens.txt"),
        "num_threads": 1,  # 4-core box: KWS is latency-critical, not throughput
        "sample_rate": sample_rate,
        "keywords_score": 1.5,
        "keywords_threshold": 0.30,
        "max_active_paths": 8,
    }


def _keywords_file(model_dir: str, thresholds: dict[str, float], default_threshold: float) -> str:
    """Stamp the profile's per-keyword thresholds onto the tokenized keywords
    (from scripts/get_kws_model.sh) as sherpa's "#threshold" suffix."""
    from pathlib import Path

    src = Path(model_dir) / "reachy-keywords.txt"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing; run scripts/get_kws_model.sh")
    lines = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        label = line.rpartition("@")[2].replace("_", " ")
        lines.append(f"{line} #{thresholds.get(label, default_threshold)}")
    out = Path(model_dir) / "reachy-keywords.runtime.txt"
    out.write_text("\n".join(lines) + "\n")
    return str(out)
