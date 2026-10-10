import pytest

from shell.attention import (
    PROFILES,
    Action,
    AttentionLease,
    AttentionManager,
    Observation,
    ParticipantResolver,
    load_profile,
)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def profile():
    return PROFILES["quiet"]


@pytest.fixture
def lease(profile, journal, clock):
    return AttentionLease(profile, journal, clock=clock)


@pytest.fixture
def manager(profile, lease, journal, clock):
    return AttentionManager(profile, lease, journal, clock=clock)


# ---- lease ----


def test_lease_expiry_is_evidence_bounded(lease, clock, profile):
    lease.acquire("david", "kws:Reachy")
    assert lease.active()
    clock.advance(profile.lease_base_s + 0.1)
    assert not lease.active()
    assert lease.expire_if_due() is True
    assert "attention.lease_expired" in lease.journal.types()


def test_renew_extends_but_never_past_max(lease, clock, profile):
    lease.acquire("david", "kws:Reachy")
    for _ in range(20):
        clock.advance(5)
        lease.renew("david", "continued-speech")
        assert lease._expires <= lease._started + profile.lease_max_s
    clock.advance(profile.lease_max_s)
    assert not lease.active()


def test_transfer_on_explicit_wake(lease, clock):
    lease.acquire("david", "kws:Reachy")
    clock.advance(2)
    lease.acquire("priya", "kws:robot")
    assert lease.holder == "priya"
    assert "attention.lease_transferred" in lease.journal.types()


def test_confidence_decays_with_lease(lease, clock):
    lease.acquire("david", "kws:Reachy")
    assert lease.confidence() == pytest.approx(1.0)
    clock.advance(10)
    assert 0 < lease.confidence() < 1.0
    clock.advance(60)
    assert lease.confidence() == 0.0


# ---- manager cascade ----


def obs(**kw):
    return Observation(**kw)


def test_keyword_always_engages(manager, lease):
    d = manager.on_keyword("david", "Reachy")
    assert d.action is Action.ENGAGE and d.confidence == 1.0
    assert lease.holder == "david"


def test_active_lease_same_participant_engages_and_renews(manager, lease, journal):
    lease.acquire("david", "kws:Reachy")
    r = ParticipantResolver(manager.profile, journal).resolve(
        obs(face_identity="david", face_confidence=0.9, face_oriented_to_robot=True)
    )
    d = manager.decide(r, obs())
    assert d.action is Action.ENGAGE and d.renewed


def test_side_conversation_is_ignored_fail_closed(manager, lease):
    lease.acquire("david", "kws:Reachy")
    r = ParticipantResolver(manager.profile, manager.journal).resolve(
        obs(face_identity="priya", face_confidence=0.9, face_oriented_to_robot=True)
    )
    d = manager.decide(r, obs())
    assert d.action is Action.IGNORE
    assert d.evidence["side_conversation"] == "priya"


def test_ambient_default_is_ignore(profile, journal, clock):
    # the fail-open regression: no evidence at all must NOT engage
    lease = AttentionLease(profile, journal, clock=clock)
    mgr = AttentionManager(profile, lease, journal, clock=clock)
    r = ParticipantResolver(profile, journal).resolve(obs())
    d = mgr.decide(r, obs())
    assert d.action is Action.IGNORE


def test_ambient_strong_evidence_engages(profile, journal, clock):
    lease = AttentionLease(profile, journal, clock=clock)
    mgr = AttentionManager(profile, lease, journal, clock=clock)
    resolver = ParticipantResolver(profile, journal)
    resolver.last_doa_deg = 10.0
    r = resolver.resolve(obs(face_identity="david", face_confidence=1.0, face_oriented_to_robot=True, doa_deg=12.0))
    lease.last_interaction = clock() - 1.0  # recent accepted dialogue
    d = mgr.decide(r, obs())
    assert d.action is Action.ENGAGE


def test_tv_duty_cycle_only_raises_the_bar(journal, clock):
    profile = PROFILES["home_tv"]  # limit 0.60; quiet's 0.85 needs a real crowd
    lease = AttentionLease(profile, journal, clock=clock)
    mgr = AttentionManager(profile, lease, journal, clock=clock)
    for _ in range(50):
        mgr.record_speech(1.0)  # 50s of speech in the last 60s
    assert mgr.duty_cycle > profile.duty_cycle_limit
    r = ParticipantResolver(profile, journal).resolve(obs())
    d = mgr.decide(r, obs())
    assert d.action is Action.IGNORE
    # same evidence that passed before now faces a higher bar
    assert d.evidence["bar"] > profile.ambient_directedness


def test_state_changing_tool_requires_confidence(profile, journal, clock):
    lease = AttentionLease(profile, journal, clock=clock)
    mgr = AttentionManager(profile, lease, journal, clock=clock)
    from shell.tools import ToolRouter

    router = ToolRouter(
        [],
        {"set_volume": lambda **k: {"ok": True}},
        journal,
        confidence_fn=mgr.tool_confidence,
        confidence_threshold=profile.tool_confidence,
    )
    assert router.execute("set_volume", {})["ok"] is False  # no lease at all
    lease.acquire("david", "kws:Reachy")
    assert router.execute("set_volume", {})["ok"] is True


# ---- profiles ----


def test_profiles_roundtrip():
    import yaml

    from shell.attention.profiles import AttentionProfile, export_yaml

    for name, p in PROFILES.items():
        assert AttentionProfile.from_dict(yaml.safe_load(export_yaml(p))) == p


def test_unknown_profile_rejected():
    with pytest.raises(KeyError):
        load_profile("make-it-up")


def test_exchange_slides_max_window_for_real_conversation(journal):
    from shell.attention.lease import AttentionLease
    from shell.attention.profiles import PROFILES

    clock = [0.0]
    lease = AttentionLease(PROFILES["quiet"], journal, clock=lambda: clock[0])
    lease.acquire("p", "kws:Reachy")
    for _ in range(20):  # a 4-minute back-and-forth, well past lease_max_s
        clock[0] += 12.0
        assert lease.exchange()
    assert lease.active()


def test_speech_start_alone_still_capped(journal):
    from shell.attention.lease import AttentionLease
    from shell.attention.profiles import PROFILES

    clock = [0.0]
    prof = PROFILES["quiet"]
    lease = AttentionLease(prof, journal, clock=lambda: clock[0])
    lease.acquire("p", "kws:Reachy")
    while lease.renew(None, "speech-start"):  # TV talking non-stop
        clock[0] += 5.0
    assert clock[0] <= prof.lease_max_s + prof.lease_renew_s


def test_keyword_mid_conversation_renews_without_new_wake(journal):
    from shell.attention.lease import AttentionLease
    from shell.attention.manager import AttentionManager
    from shell.attention.profiles import PROFILES

    prof = PROFILES["quiet"]
    lease = AttentionLease(prof, journal)
    mgr = AttentionManager(prof, lease, journal)
    first = mgr.on_keyword(None, "Reachy")
    holder = lease.holder
    again = mgr.on_keyword(None, "Reachy")
    assert first.acquired and not again.acquired
    assert lease.holder == holder


def test_exchange_rearms_lease_that_lapsed_during_reply(journal):
    from shell.attention.lease import AttentionLease
    from shell.attention.profiles import PROFILES

    clock = [0.0]
    lease = AttentionLease(PROFILES["quiet"], journal, clock=lambda: clock[0])
    lease.acquire("p", "kws:Reachy")
    clock[0] += 40.0  # long reply played past the window; not yet expired
    assert not lease.active()
    assert lease.exchange("reply-played")
    assert lease.active()
    lease.expire_if_due()
    clock[0] += 60.0
    lease.expire_if_due()
    assert not lease.exchange()  # truly expired: needs a wake word again


def test_speech_start_renewal_bounded_since_wake_or_reply(journal):
    from shell.attention.lease import AttentionLease
    from shell.attention.profiles import PROFILES

    clock = [0.0]
    prof = PROFILES["quiet"]
    lease = AttentionLease(prof, journal, clock=lambda: clock[0])
    lease.acquire("p", "kws:Reachy")
    while lease.renew(None, "speech-start", cap_s=prof.lease_unanswered_s):  # TV
        clock[0] += 3.0
    assert clock[0] <= prof.lease_unanswered_s + 3.0
    # a real exchange gives the user a fresh window
    lease.acquire("p", "kws:Reachy")
    clock[0] += 10.0
    lease.exchange("reply-played")
    clock[0] += 10.0
    assert lease.renew(None, "speech-start", cap_s=prof.lease_unanswered_s)
    assert lease.remaining_s <= prof.lease_unanswered_s - 10.0 + 1e-9
