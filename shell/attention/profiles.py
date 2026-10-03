"""Versioned, exportable attention profiles (ADR 0003).

Ship understandable profiles rather than a mystery sensitivity slider.
Every value is documented; Advanced UI may expose the raw numbers.
Profiles must round-trip so experiment results are reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

PROFILE_VERSION = 1


@dataclass(frozen=True)
class AttentionProfile:
    name: str
    version: int = PROFILE_VERSION

    # Keyword spotting: per-keyword score threshold (higher = requires more
    # confidence before firing). Per-keyword boosts override the default.
    kws_default_threshold: float = 0.30
    # "Reachy" is out-of-vocabulary for the BPE KWS model and scores lower
    # than real words: measured on synthesized "Hey Reachy" it fires at <=0.25
    # and misses at 0.30, with no false hits on a negative sentence. Re-measure
    # on real room audio before raising (and per profile for noisy rooms).
    kws_thresholds: dict = field(default_factory=lambda: {"Reachy": 0.25, "hey Reachy": 0.25})

    # Attention lease, seconds. Renewed on evidence; a single hard timer is
    # explicitly NOT how this works (ADR 0003) - these are its bounds.
    lease_base_s: float = 15.0
    lease_renew_s: float = 15.0
    lease_max_s: float = 120.0

    # Minimum attention confidence (0..1) to allow STATE-CHANGING tools.
    tool_confidence: float = 0.6

    # Ambient directedness: an utterance with no lease engages only when its
    # evidence score reaches this. Heuristics may only RAISE this bar, never
    # early-return "respond" (the structural bug behind 274-responds-0-ignores).
    ambient_directedness: float = 0.65

    # Rolling 60s speech duty cycle above which we believe the room has a TV
    # or crowd talking; all bars are multiplied by `duty_raise`.
    duty_cycle_limit: float = 0.70
    duty_raise: float = 1.5

    # Evidence weights (ParticipantResolver / manager fusion).
    w_face_known: float = 0.35  # recognized face oriented toward robot
    w_face_present: float = 0.15  # any face oriented toward robot
    w_doa_continuity: float = 0.25  # speech azimuth close to last participant
    w_recency: float = 0.25  # recent accepted dialogue
    recency_window_s: float = 20.0
    doa_tolerance_deg: float = 25.0  # linear array: broadside-ambiguous, keep loose

    # Ambiguous band: within +/- ask_band of the directedness bar, optionally
    # respond with "were you talking to me?" instead of silently ignoring.
    ask_band: float = 0.10
    ask_if_addressed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AttentionProfile":
        known = {f for f in cls.__dataclass_fields__}  # noqa: RUF012
        return cls(**{k: v for k, v in d.items() if k in known})


PROFILES: dict[str, AttentionProfile] = {
    "quiet": AttentionProfile(
        name="quiet",
        lease_base_s=20.0,
        tool_confidence=0.5,
        ambient_directedness=0.45,
        duty_cycle_limit=0.85,
    ),
    "home_tv": AttentionProfile(
        name="home_tv",
        kws_default_threshold=0.35,
        tool_confidence=0.7,
        ambient_directedness=0.70,
        duty_cycle_limit=0.60,
        ask_if_addressed=True,
    ),
    "event": AttentionProfile(
        name="event",
        kws_default_threshold=0.40,
        lease_base_s=10.0,
        tool_confidence=0.8,
        ambient_directedness=0.80,
        duty_cycle_limit=0.50,
        duty_raise=2.0,
    ),
}


def load_profile(name: str) -> AttentionProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown attention profile {name!r}; have {sorted(PROFILES)}") from None


def export_yaml(profile: AttentionProfile) -> str:
    import yaml

    return yaml.safe_dump(profile.to_dict(), sort_keys=False)
