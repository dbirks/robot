"""ParticipantResolver: who/where appears to be speaking?

Combines independent observations into a SCORED HYPOTHESIS, never false
certainty (EPIC). Identity signals are evidence, not security credentials;
a remembered voice must never authorize anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import journal as J
from .profiles import AttentionProfile

UNKNOWN = "unknown"


@dataclass
class Observation:
    doa_deg: float | None = None
    track_id: int | None = None
    face_identity: str | None = None  # from InsightFace
    face_confidence: float = 0.0
    face_oriented_to_robot: bool = False
    active_speaker_p: float | None = None  # from ASD (Phase 6); None = unknown
    voice_identity: str | None = None  # from WeSpeaker (Phase 7)
    voice_confidence: float = 0.0


@dataclass
class ResolvedParticipant:
    id: str
    confidence: float
    evidence: dict


class ParticipantResolver:
    def __init__(self, profile: AttentionProfile, journal, clock=None) -> None:
        self.profile = profile
        self.journal = journal
        self.last_doa_deg: float | None = None

    def resolve(self, obs: Observation) -> ResolvedParticipant:
        ev: dict = {}
        score = 0.0
        identity = UNKNOWN

        if obs.face_identity and obs.face_oriented_to_robot:
            identity = obs.face_identity
            score += self.profile.w_face_known * obs.face_confidence
            ev["face"] = f"{obs.face_identity}@{obs.face_confidence:.2f}"
        elif obs.face_oriented_to_robot:
            score += self.profile.w_face_present
            ev["face_present"] = True

        if obs.doa_deg is not None and self.last_doa_deg is not None:
            delta = abs(obs.doa_deg - self.last_doa_deg)
            delta = min(delta, 360.0 - delta)
            if delta <= self.profile.doa_tolerance_deg:
                score += self.profile.w_doa_continuity * (1.0 - delta / self.profile.doa_tolerance_deg)
                ev["doa_delta_deg"] = round(delta, 1)

        if obs.active_speaker_p is not None:
            score += self.profile.w_doa_continuity * obs.active_speaker_p
            ev["active_speaker_p"] = obs.active_speaker_p

        # Voice is supporting evidence only - never authorization.
        if obs.voice_identity:
            ev["voice"] = f"{obs.voice_identity}@{obs.voice_confidence:.2f}"
            if identity == UNKNOWN:
                identity = obs.voice_identity
            score += 0.10 * obs.voice_confidence

        if obs.track_id is not None:
            ev["track_id"] = obs.track_id

        resolved = ResolvedParticipant(id=identity, confidence=min(1.0, score), evidence=ev)
        if obs.doa_deg is not None:
            self.last_doa_deg = obs.doa_deg
        self.journal.write(
            J.PARTICIPANT_RESOLVED, id=resolved.id, confidence=round(resolved.confidence, 3), evidence=ev
        )
        return resolved
