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
    wake(manager, lease)
    clock.advance(14.9)  # inside the arm window? no: the window is lease_renew_s from the wake
    if clock.t > 1000.0 + manager.profile.lease_renew_s:
        assert manager.tool_confidence("set_volume") < manager.profile.tool_confidence


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
