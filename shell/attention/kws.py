"""Open-vocabulary keyword spotting via sherpa-onnx (ADR 0003 fast path).

"Reachy", "hey Reachy", "robot", "hey robot" - no LLM, no STT, sub-100ms
reaction class. Model/runtime pinned in docs/pins.yaml; install via
scripts/get_kws_model.sh. sherpa-onnx is an optional extra: this module
imports it lazily so the shell runs (without wake words) on a machine that
hasn't installed the extra yet.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_KEYWORDS = ["Reachy", "hey Reachy", "robot", "hey robot"]


@dataclass
class KeywordHit:
    keyword: str
    score: float


class KeywordSpotter:
    def __init__(
        self,
        model_dir: str,
        *,
        keywords: list[str] | None = None,
        thresholds: dict[str, float] | None = None,
        default_threshold: float = 0.30,
        sample_rate: int = 16000,
    ) -> None:
        import sherpa_onnx  # lazy: optional extra

        self.default_threshold = default_threshold
        self.thresholds = thresholds or {}
        self._stream = None
        self._spotter = sherpa_onnx.KeywordSpotter(
            **_sherpa_config(model_dir, sample_rate),
            keywords_file=_keywords_file(model_dir, keywords or DEFAULT_KEYWORDS),
        )

    def process(self, pcm: bytes) -> list[KeywordHit]:
        import numpy as np

        if self._stream is None:
            self._stream = self._spotter.create_stream()
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._stream.accept_waveform(16000, samples)
        hits: list[KeywordHit] = []
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            r = self._spotter.get_result(self._stream)
            key = (r.get("keyword") or str(r)).strip()
            if key:
                score = float(r.get("score", 1.0) if isinstance(r, dict) else 1.0)
                if score >= self.thresholds.get(key, self.default_threshold):
                    hits.append(KeywordHit(keyword=key, score=score))
                self._spotter.reset_stream(self._stream)
                self._stream = None
        return hits

    def reset(self) -> None:
        self._stream = None


def _sherpa_config(model_dir: str, sample_rate: int) -> dict:
    """Transducer KWS model set paths, per sherpa-onnx docs; the model dir
    layout is created by scripts/get_kws_model.sh and pinned in pins.yaml."""
    from pathlib import Path

    p = Path(model_dir)
    return {
        "transducer": {
            "encoder": str(p / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
            "decoder": str(p / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
            "joiner": str(p / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
        },
        "tokens": str(p / "tokens.txt"),
        "num_threads": 1,  # 4-core box: KWS is latency-critical, not throughput
        "sample_rate": sample_rate,
        "keywords_score": 1.5,
        "keywords_threshold": 0.30,
        "max_active_paths": 8,
    }


def _keywords_file(model_dir: str, keywords: list[str]) -> str:
    from pathlib import Path

    path = Path(model_dir) / "reachy-keywords.txt"
    if not path.exists():
        path.write_text("\n".join(keywords) + "\n")
    return str(path)
