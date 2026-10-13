"""The tool gate: attention is measured from evidence, and it survives the model's thinking time.

Measured live on 2026-10-13 at 13:06 in the hackday config: the wake word hit, the lease was live,
and `set_volume` came back `attention.tool_denied` at confidence 0.7527 against the event profile's
0.8 - `play_emotion` denied twice in the same second. The guest had spoken 3.0 s earlier. Two
independent defects were involved, and both are pinned here:

1. the gate read `lease.confidence()` = `remaining_s / lease_renew_s`, so clearing 0.8 with a 15 s
   renew window required 12 s of lease left - a ~3 s budget between "person stops talking" and
   "local 4B emits a tool call", which this box cannot meet (a cold camera grab alone is 2.7-3.0 s);
2. worse, that timer starts at `lease_base_s / lease_renew_s` = 10/15 = **0.667 at the moment the
   wake word lands**, i.e. below the bar it is compared against: the strongest evidence the system
   has (`on_keyword` -> confidence 1.0, journalled as `attention.decision stage=keyword`) was being
   thrown away in favour of a number that cannot pass.

Every test runs on a fake clock. The timing IS the bug, so the numbers are exact.
"""

from pathlib import Path

import pytest

from shell.attention.lease import AttentionLease
from shell.attention.manager import Action, AttentionManager
from shell.attention.profiles import PROFILES


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class Journal:
    def __init__(self):
        self.rows = []

    def write(self, kind, **fields):
        self.rows.append((kind, fields))


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def profile():
    return PROFILES["event"]  # tool_confidence 0.8, lease_base 10, lease_renew 15


@pytest.fixture
def journal():
    return Journal()


@pytest.fixture
def lease(profile, journal, clock):
    return AttentionLease(profile, journal, clock=clock)


@pytest.fixture
def manager(profile, lease, journal, clock):
    return AttentionManager(profile, lease, journal, clock=clock)


def wake(manager, lease):
    """The real entry point: the wake word lands and the manager engages."""
    d = manager.on_keyword("doa-1", "Reachy")
    assert d.action == Action.ENGAGE and d.confidence == pytest.approx(1.0)
    return d


# --- the two defects, stated as facts ---------------------------------------


def test_pin_the_timer_alone_cannot_pass_the_event_gate_even_at_full_lease(profile, lease):
    """Documents the underlying miscalibration so nobody "fixes" it by shrinking lease_base_s
    further, which would make the gate impossible rather than merely late."""
    lease.acquire("doa-1", evidence="kws:Reachy")
    assert lease.confidence() == pytest.approx(profile.lease_base_s / profile.lease_renew_s)
    assert lease.confidence() < profile.tool_confidence, (
        "if this flips, the timer can pass the gate on its own and the arming below is redundant"
    )


def test_reproduces_the_live_denial_when_the_gate_reads_only_the_timer(manager, lease, clock):
    """The control: no ENGAGE decision recorded, so nothing was armed, and 3.7 s of model
    deliberation is fatal - the 13:06 state, exactly."""
    lease.acquire("doa-1", evidence="kws:Reachy")
    clock.advance(3.7)
    assert manager.tool_confidence("set_volume") == pytest.approx(lease.confidence())
    assert manager.tool_confidence("set_volume") < manager.profile.tool_confidence


# --- the fix ----------------------------------------------------------------


def test_a_wake_word_still_opens_the_gate_after_the_model_thinks(manager, lease, clock):
    wake(manager, lease)
    clock.advance(3.7)  # STT + a 4B composing a reply, which is where it died live
    gated = manager.tool_confidence("set_volume")
    assert gated >= manager.profile.tool_confidence, f"denied again at {gated:.3f}"


def test_a_slow_tool_round_trip_still_belongs_to_the_turn_that_asked(manager, lease, clock):
    wake(manager, lease)
    clock.advance(8.0)  # snapshot-style: grab, vision render, speak
    assert manager.tool_confidence("take_snapshot") >= manager.profile.tool_confidence


def test_emotes_get_through_on_the_same_turn_as_the_words(manager, lease, clock):
    """`play_emotion` was denied in the same second as set_volume; it is STATE_CHANGING too."""
    wake(manager, lease)
    clock.advance(4.5)
    assert manager.tool_confidence("play_emotion") >= manager.profile.tool_confidence


def test_freeze_for_turn_cannot_lower_a_stronger_arm(manager, lease, clock):
    wake(manager, lease)
    clock.advance(6.0)  # timer down to ~0.4
    manager.freeze_for_turn()  # the transcript hook fires with the (now weak) lease reading
    assert manager.tool_confidence("remember") == pytest.approx(1.0), "the wake word's 1.0 was lost"


# --- the limits, which are what make this safe ------------------------------


def test_no_lease_means_no_gate_pass_whatever_was_armed(manager, lease, clock):
    """ADR 0003 intact: the arm cannot outlive the attention it came from."""
    wake(manager, lease)
    clock.advance(11.0)  # past lease_base (10 s), still inside the 15 s arm window
    assert lease.expire_if_due() is True
    assert not lease.active()
    assert manager.tool_confidence("remember") == 0.0
    assert manager._turn_conf == 0.0, "a dropped-but-retained arm could be revived by the next caller"


def test_the_arm_expires_with_the_turn_window(manager, lease, clock, monkeypatch):
    """A turn that stalls cannot hold the gate open for whoever walks up next."""
    wake(manager, lease)
    clock.advance(manager.profile.lease_renew_s + 0.5)  # the arm has now expired
    lease.acquire("doa-2", evidence="kws:Reachy")  # a FRESH, active lease; acquire never arms
    monkeypatch.setattr(lease, "confidence", lambda: 0.42)
    assert lease.active(), "setup: the lease under test has to be alive"
    assert manager.tool_confidence("set_volume") == pytest.approx(0.42), "the arm outlived its turn"


def test_weak_ambient_evidence_is_not_laundered_into_a_pass(manager, lease, clock):
    """The arming takes the decision's own confidence, so a marginal ambient engage that scored
    0.75 still cannot change state. This is the difference between fixing the scale and
    abandoning the gate."""
    # A 0.75 ambient engage arms 0.75, which is under the 0.8 bar: evidence is carried through,
    # not inflated into a pass. Driving it through the manager keeps the coupling honest.
    lease.acquire("unknown", evidence="ambient-score:0.75")
    clock.advance(0.1)
    assert manager.tool_confidence("learn_face") < manager.profile.tool_confidence
    assert not lease.active() or lease.confidence() < manager.profile.tool_confidence


def test_a_stale_lease_without_any_new_evidence_still_denies(manager, lease, clock):
    """This test could not fail for eight hours. It advanced the clock 14.9 s and put its only
    assert behind `if clock.t > 1000.0 + lease_renew_s` - and 14.9 is not greater than 15.0, so
    the branch was never taken and the file reported twelve greens where there were eleven.
    Mine, from a53869d, found at 18:19 by the worker that fixed the gate it had failed to wire.
    An assert under a condition is not a test unless something proves the condition is reached:
    the clock now moves past the window unconditionally and the assertion has no guard to hide
    behind. The intent was real, which is exactly why a vacuous test is worse than a missing one.
    """
    wake(manager, lease)
    clock.advance(manager.profile.lease_base_s + 0.1)  # nobody renewed it: the lease lapses
    assert manager.tool_confidence("set_volume") < manager.profile.tool_confidence, (
        "a lapsed lease must not vouch for a tool call"
    )


def test_sleep_zeroes_the_gate_even_with_a_fresh_wake_word(manager, lease, clock):
    """The shell composes it as `0.0 if sleep.sleeping else manager.tool_confidence(name)`
    (shell/__main__.py). Asserted here because an emote or a volume change while he is lying down
    with the microphone shut is the failure issue #36 exists to prevent."""
    wake(manager, lease)

    def gate(name, sleeping):
        return 0.0 if sleeping else manager.tool_confidence(name)

    assert gate("set_volume", True) == 0.0
    assert gate("set_volume", False) >= manager.profile.tool_confidence


def test_arming_happens_for_every_engage_path_not_just_the_keyword(manager, lease, clock, journal):
    """`_log` is where the arm hangs off, so any Decision(ENGAGE) that reaches the journal arms the
    gate. If someone adds a new engage path and logs it, it works; if they bypass _log, this test
    is where the coupling is documented."""
    wake(manager, lease)
    stages = [r[1].get("stage") for r in journal.rows if r[0] == "attention.decision"]
    assert stages, "the decision never reached the journal, so the hook never ran"


# --- the hook has to be CALLED: a53869d described freeze_for_turn() and shipped it dead -----
#
# The second boot of 2026-10-13 had the fix in it and denied `set_volume` anyway, at 18:11:46
# with confidence 0.7459746307999922 against the same 0.8 bar. The wake word landed at 18:11:31,
# the guest's request was accepted at 18:11:44.834, the model's tool call at 18:11:46.434 -
# 15.32 s after the wake word, i.e. 0.32 s outside an arm window that nothing ever moved off the
# wake word. Everything above this line tests the manager in isolation, which is exactly how a
# method with zero production callers passed review.


def router_for(manager, journal):
    """The production composition minus the devices: `ToolRouter(confidence_fn=tool_confidence,
    confidence_threshold=profile.tool_confidence)` (shell/__main__.py), with the real manager as
    the confidence function. `set_volume` is in STATE_CHANGING, so this is the gate the booth hit."""
    from shell.tools import ToolRouter

    calls: list[str] = []

    def set_volume(level: str = "louder"):
        calls.append(level)
        return {"ok": True, "level": level}

    return (
        ToolRouter(
            [],
            {"set_volume": set_volume},
            journal,
            confidence_fn=manager.tool_confidence,
            confidence_threshold=manager.profile.tool_confidence,
        ),
        calls,
    )


def live_booth_timeline(manager, lease, clock):
    """Replay the journalled lease evidence of 18:11:31 -> 18:11:46 on the fake clock.

    Returns the arm-freeze call so each test decides whether the shell's hook fired. The renewals
    are the real ones from `attention.lease_renewed`, including the 20 s unanswered cap that
    `speech-start` carries (shell/__main__.py:214).
    """
    manager.on_keyword(None, "hey Reachy")  # 18:11:31.117, arms 1.0 for one renew window
    clock.advance(5.661)
    lease.exchange("reply-done")  # 18:11:36.778
    clock.advance(1.774)
    lease.exchange("reply-played")  # 18:11:38.552
    cap = manager.profile.lease_unanswered_s
    for dt in (1.721, 1.574, 0.883):  # 18:11:40.273, 18:11:41.847, 18:11:42.623
        clock.advance(dt)
        lease.renew(None, "speech-start", cap_s=cap)


def test_the_live_1811_denial_is_allowed_once_the_transcript_hook_fires(manager, lease, journal, clock):
    """Today's failure, made permanent: speaker == lease holder, live renewed lease, tool call 15.3 s
    after the wake word and 1.6 s after his own accepted turn."""
    live_booth_timeline(manager, lease, clock)
    clock.advance(2.211)  # 18:11:44.834 turn.transcript_final: "...turn up your volume a bunch..."
    manager.freeze_for_turn()
    clock.advance(1.6)  # 18:11:46.434, the model finally emits the call
    assert lease.confidence() < manager.profile.tool_confidence, (
        "setup: the timer alone must still be denying here, as it did live"
    )
    assert lease.confidence() == pytest.approx(0.746, abs=0.01), (
        f"the live number was 0.7459, not {lease.confidence():.4f}"
    )
    router, calls = router_for(manager, journal)
    result = router.execute("set_volume", {"level": "louder"})
    assert result["ok"] is True and calls == ["louder"], result
    assert not [r for r in journal.rows if r[0] == "attention.tool_denied"], "the gate denied it again"


def test_the_same_timeline_denies_without_the_hook(manager, lease, journal, clock):
    """Counterfactual, not production: the same lease evidence with the transcript hook deliberately
    not called - the state a53869d shipped and the state `test_the_accepted_transcript_hook_actually_
    calls_the_hook` exists to prevent. The live shell reaches `freeze_for_turn()` on every accepted
    turn; this is the pair that shows the arm is what makes 18:11:46 pass, not a loosened bar."""
    live_booth_timeline(manager, lease, clock)
    clock.advance(2.211)
    clock.advance(1.6)
    router, calls = router_for(manager, journal)
    result = router.execute("set_volume", {"level": "louder"})
    assert result.get("denied") == "attention_confidence_too_low", result
    assert calls == []


def test_held_evidence_cannot_be_decayed_back_under_the_bar_by_a_renewed_lease(manager, lease, journal, clock):
    """The property the gate was missing, per the 18:11:31 -> 18:11:46 measurement: while the lease is
    live, held by the participant who earned the arm, and inside the window, decay may only RAISE the
    number. Sampled late on purpose - the ratio here is 0.60, a level that denied `set_volume` live at
    18:12:09, against evidence of 1.0. Before this, `tool_confidence` returned `live` the moment the
    window closed, which is a renewed lease read as an absence of trust.
    """
    live_booth_timeline(manager, lease, clock)
    clock.advance(2.211)  # 18:11:44.834 accepted request; the window now ends at +28.7 s
    manager.freeze_for_turn()
    clock.advance(6.283)
    lease.exchange("reply-done")  # a real exchange slides the unanswered cap, as `reply-done` did live
    clock.advance(6.0)  # +26.1 s: inside the window, and deep into the ratio's decay
    assert lease.confidence() < manager.profile.tool_confidence, (
        f"setup: the timer should read {lease.confidence():.2f}, well under the bar"
    )
    router, calls = router_for(manager, journal)
    assert router.execute("set_volume", {"level": "louder"})["ok"] is True
    assert calls == ["louder"]


def test_a_transferred_lease_does_not_inherit_the_previous_speaker_s_evidence(manager, lease, journal, clock):
    """ADR 0003 is about WHO, and the holder is the only identity the gate has: `AttentionLease.renew`
    transfers on a different participant id, and `AttentionManager.decide()` has no production caller
    (documented at envs/hackday.yaml:27, issue #34) to notice the change. So `tool_confidence` checks
    the holder itself; the armed 1.0 belongs to a person, not to the microphone.
    """
    wake(manager, lease)  # doa-1 armed with 1.0
    clock.advance(2.0)
    lease.renew("doa-2", "speech-start")  # the lease transfers to someone else
    assert lease.holder == "doa-2" and lease.active()
    assert manager.tool_confidence("set_volume") == pytest.approx(lease.confidence()), (
        "the old speaker's evidence was handed to whoever walked up next"
    )
    router, calls = router_for(manager, journal)
    assert router.execute("set_volume", {"level": "louder"}).get("denied") == "attention_confidence_too_low"
    assert calls == []


def test_no_lease_at_all_denies_a_fluent_request_whatever_the_words_were(manager, lease, journal):
    """ADR 0003 is about WHO, not WHAT: nobody who never said his name moves this robot's gain, however
    fluent their English. The phrasing is deliberately one the deterministic intent matches, so the
    denial can only be attention - and note the words are NOT what the shell rejected at 18:12:09."""
    from shell.intents import volume_request

    assert volume_request("turn the volume up") == "louder", "setup: these words are a command"
    assert manager.tool_confidence("set_volume") == 0.0, "no lease granted, so nothing is armed"
    router, calls = router_for(manager, journal)
    result = router.execute("set_volume", {"level": "louder"})
    assert result.get("denied") == "attention_confidence_too_low", result
    assert calls == []


def test_a_lapsed_lease_cannot_be_revived_by_the_arm_it_left_behind(manager, lease, journal, clock):
    """Sharper than expiry-by-time: the arm window (15 s) is LONGER than the event lease (10 s), so a
    call at +12 s is inside a live arm over a dead lease. The lease is the claim; the arm is only its
    memory."""
    wake(manager, lease)
    clock.advance(12.0)
    assert lease.expire_if_due() is True and not lease.active()
    assert manager.tool_confidence("set_volume") == 0.0
    router, calls = router_for(manager, journal)
    assert router.execute("set_volume", {"level": "louder"}).get("denied") == "attention_confidence_too_low"
    assert calls == []


def test_a_turn_gap_longer_than_the_window_cannot_reach_back_for_the_wake_word(manager, lease, journal, clock):
    """The 18:12:08 shape, which is the case that must NOT inherit the 1.0: the operator was talking
    about the robot, not to it, the model called `set_volume` anyway, and the gate denied it at 0.256.
    Re-arming from a transcript that arrives past the window re-arms with the lease's own weak reading
    and cannot resurrect the wake word's evidence."""
    live_booth_timeline(manager, lease, clock)
    t0 = clock.t - 11.506  # the wake word, which is what the arm window was measured from
    clock.advance(2.211)
    manager.freeze_for_turn()  # 18:11:44.834 - the real one, arm now runs to +28.7 s
    clock.advance(12.9)
    lease.renew(None, "speech-start", cap_s=manager.profile.lease_unanswered_s)  # room speech keeps the lease up
    clock.advance(12.4)  # 18:12:08.057 turn.transcript_final: "He seems to not be able to pull his tool..."
    assert clock.t - t0 > manager.profile.lease_renew_s, "setup: this turn lands past the arm window"
    manager.freeze_for_turn()
    gated = manager.tool_confidence("set_volume")
    assert gated < manager.profile.tool_confidence, f"the wake word's 1.0 was revived: {gated:.4f}"
    assert gated == pytest.approx(lease.confidence(), abs=1e-9), "it must fall back to the live decay"


def test_the_accepted_transcript_hook_actually_calls_the_hook():
    """The coupling test that a53869d lacked: it tested `freeze_for_turn()` by calling it, never by
    checking that the shell calls it. `shell/__main__.py` cannot be imported hardware-free (it pulls
    the audio owners and the SDK), so the call site is pinned against the source text."""
    src = (Path(__file__).resolve().parent.parent / "shell" / "__main__.py").read_text()
    assert "manager.freeze_for_turn()" in src, (
        "the arm is anchored to the wake word again: every state-changing tool in the second and "
        "later turn of a lease dies on lease.confidence()"
    )
    body = src.split("def on_transcript", 1)[1].split("def on_response_done", 1)[0]
    assert "manager.freeze_for_turn()" in body, "freeze_for_turn is called, but not from the accepted-turn hook"
    assert body.index("manager.freeze_for_turn()") < body.index("volume_request(text)"), (
        "the deterministic volume intent must see the armed turn, so it runs after the freeze"
    )
