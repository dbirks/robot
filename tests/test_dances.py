"""The dance layer: same contract as the emote layer, one pack over.

Modelled on tests/test_emotes.py deliberately - the two packs share `emotes.parse_emote`, so the
tests share their shape too. Everything here is hardware-free: `parse_dance` is pure, and
`huggingface_hub` is imported lazily inside the resolver specifically so a box with neither the
package nor the cache can still assert the network is not an option. Where a test needs the real
pack it skips out loud rather than inventing a fake one.
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from shell.motion import dances, emotes
from shell.motion.owner import Animation


def _have_pack() -> bool:
    """Can this box actually read a dance? CI has neither huggingface_hub nor the cache."""
    try:
        dances.load("chicken_peck")
    except Exception:
        return False
    return True


HAVE_PACK = _have_pack()
NEEDS_PACK = pytest.mark.skipif(
    not HAVE_PACK,
    reason="dances library is offline-only; this box has neither huggingface_hub nor the cache",
)


def payload(seconds=2.0, frames=100):
    """A library-shaped file, identical in shape to the emotions library's."""
    time = [seconds * i / (frames - 1) for i in range(frames)]
    data = [
        {
            "head": np.eye(4).tolist(),
            "antennas": [0.1 * (i % 5) - 0.2, -0.1 * (i % 5) + 0.2],
            "body_yaw": 0.0,
            "check_collision": None,
        }
        for i in range(frames)
    ]
    return {"description": "test", "time": time, "set_target_data": data}


class FakeMotion:
    def __init__(self, journal=None, sleeping=False):
        self.played: list[Animation] = []
        self.preempt_flags: list[bool] = []
        self.woken = 0
        self.sleeping = sleeping
        self.journal = journal

    def play(self, anim, *, preempt=False):
        self.preempt_flags.append(preempt)
        self.played.append(anim)

    def wake(self):
        self.woken += 1


class FakeRobot:
    connected = True
    mini = object()


class RecordingJournal:
    def __init__(self):
        self.rows: list[tuple[str, dict]] = []

    def write(self, kind, **fields):
        self.rows.append((kind, fields))


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def _unreadable(name: str) -> Path:
    """Stand in for a missing cache entry. A def rather than the throw-lambda
    `tests/test_emotes.py` uses, so a new file does not add `ty` diagnostics of its own; same
    behaviour, and the same reason `huggingface_hub` is faked: CI has no pack to miss."""
    raise dances.DanceUnavailable(name)


def handler(motion, clock=None):
    from shell.robot_tools import make_motion_handlers

    return make_motion_handlers(FakeRobot(), motion, clock=clock)["dance"]


# --- the vocabulary ----------------------------------------------------------


def test_schema_enum_is_the_vocabulary_is_the_file_map():
    enum = dances.DANCE_TOOL["function"]["parameters"]["properties"]["move"]["enum"]
    assert enum == list(dances.DANCE_NAMES) == list(dances.DANCE_FILES)
    assert len(enum) == len(set(enum)) > 0


def test_every_name_is_the_files_own_stem():
    """The vendor names the clips; translating them into friendlier words would be a second
    vocabulary to keep in sync and it buys the model nothing."""
    for name, filename in dances.DANCE_FILES.items():
        assert filename == f"{name}.json"


def test_the_description_names_every_dance_and_fences_off_its_neighbours():
    """`dance` overlaps two tools that already exist: `play_emotion` has a `dance` entry, and
    `simple_nod` / `yeah_nod` / `uh_huh_tilt` are what the `nod` tool does. Unfenced, they compete
    and the routing measurement stops meaning anything."""
    text = dances.DANCE_TOOL["function"]["description"]
    for name in dances.DANCE_NAMES:
        assert name in text, f"{name} is callable but never mentioned in the description"
    assert "play_emotion" in text and "nod" in text
    assert "MOVEMENT" in text and "ONLY way his body performs" in text
    assert "At most once" in text


def test_the_tool_is_named_dance_and_takes_one_required_argument():
    fn = dances.DANCE_TOOL["function"]
    assert fn["name"] == "dance"
    assert dances.DANCE_TOOLS == [dances.DANCE_TOOL]
    assert fn["parameters"]["required"] == ["move"]
    assert set(fn["parameters"]["properties"]) == {"move"}


@NEEDS_PACK
def test_the_enum_is_the_whole_pack_on_disk_not_a_subset():
    """The honest version of "an explicit enum of what is actually loaded": if Pollen adds a
    twentieth routine, this fails and the tool gains it deliberately rather than the pack quietly
    growing past the vocabulary."""
    first = dances._path_for(dances.DANCE_NAMES[0])
    on_disk = {p.stem for p in Path(first).parent.glob("*.json")}
    assert on_disk == set(dances.DANCE_FILES), sorted(on_disk ^ set(dances.DANCE_FILES))


@NEEDS_PACK
def test_the_dance_pack_is_not_the_emotion_packs_dance_entry():
    """`EMOTE_FILES["dance"]` is `dance1.json` from the emotions library. If a future refactor
    pointed both names at one repo, the tool would be lying about what it loaded."""
    assert dances.REPO_ID != emotes.REPO_ID
    assert set(dances.DANCE_FILES.values()).isdisjoint(set(emotes.EMOTE_FILES.values()))


def test_dance_is_registered_once_and_reaches_the_router():
    """build_handlers replaces the legacy tool list; two schemas for one name means the service
    picks whichever it likes. There is no legacy `dance`, so it must arrive exactly once."""
    from shell.robot_tools import build_handlers

    tools, handlers = build_handlers(
        types.SimpleNamespace(connected=False, mini=None),
        None,
        motion=FakeMotion(),
    )
    names = [t["function"]["name"] for t in tools]
    if not names:
        # Skips, loudly, rather than failing: in a bare environment (what CI installs)
        # shell/robot_tools.py cannot import the legacy bridge - it logs "legacy robot tools
        # unavailable: ModuleNotFoundError" at :489 - and build_handlers then returns NO schemas
        # at all, so "registered exactly once" has nothing to observe. A red CI run that is really
        # "this box has no legacy stack" teaches nobody anything; a skip that says which line
        # decided it does.
        pytest.skip(
            "legacy robot_tools bridge is unavailable here (shell/robot_tools.py:489 logs it), "
            "so build_handlers returns an empty schema list and registration cannot be asserted"
        )
    assert names.count("dance") == 1, names
    assert "dance" in handlers


# --- gating: a recorded decision, not an absence -----------------------------


def test_dance_is_ungated_by_decision_and_the_reason_is_still_true():
    """`dance` is NOT in STATE_CHANGING (see the comment on that set for both reasons). What this
    pins is the mechanical half: the dashboard resolves every gated name against
    `app/robot_tools.py:TOOLS` + `shell/memories.py:MEMORY_TOOLS`, and a shell-native motion
    schema is in neither - so gating it today would break the panel's own consistency gate. When
    Phase 2 moves the motion tools into a source the panel reads, this test is the one to flip
    together with adding the name."""
    from shell.tools import STATE_CHANGING

    assert "dance" not in STATE_CHANGING
    import dashboard.env_panel as ep

    known = set(ep._schema_names(ep.Path.cwd(), ep._TOOLS_SOURCE, ep._TOOLS_LIST))
    known |= set(ep._schema_names(ep.Path.cwd(), ep._MEMORY_SOURCE, ep._MEMORY_LIST))
    assert "dance" not in known, "the reason to gate it has arrived; now add it to STATE_CHANGING"


def test_the_exclusion_lever_works_without_any_new_code():
    """The containment offered instead of an attention lease: a room can drop the schema AND the
    handler, so a booth that must not perform has no way to perform."""
    from shell.tools import apply_tool_policy

    schema = dances.DANCE_TOOL
    kept, handlers = apply_tool_policy([schema], {"dance": lambda: {}}, exclude=["dance"])
    assert kept == [] and handlers == {}


# --- the shared keyframe math ------------------------------------------------


def test_parse_uses_the_files_own_clock_and_the_dance_ceiling():
    anim = dances.parse_dance("chicken_peck", payload(seconds=2.0, frames=400))
    assert len(anim.keyframes) == emotes.SUBSAMPLE_TO
    assert sum(kf.duration for kf in anim.keyframes) == pytest.approx(2.0, rel=0.02)
    assert anim.name == "dance:chicken_peck"
    assert anim.priority == 0, "a routine queues behind the gaze; it does not fight it"
    assert anim.relative is False, "absolute keyframes are what let the owner blend over the gaze"


def test_the_ceiling_scales_the_routine_rather_than_truncating_it():
    """Same rule as REACHY_EMOTE_MAX_S, and the reason it exists at all: the pack is upstream-owned
    and community-tagged, so the day Pollen ships an 18 s routine the default has to be 'faster',
    not 'his head is busy for the rest of the guest's turn'."""
    big = payload(seconds=20.0, frames=900)
    anim = dances.parse_dance("dizzy_spin", big)
    assert len(anim.keyframes) == emotes.SUBSAMPLE_TO, "frames were dropped, not sped up"
    assert sum(kf.duration for kf in anim.keyframes) == pytest.approx(6.0, rel=0.02)


def test_ceiling_zero_is_the_written_length_and_nonsense_is_the_default(monkeypatch):
    monkeypatch.setenv(dances.DANCE_MAX_S_ENV, "0")
    assert dances.dance_max_s() == 0.0
    anim = dances.parse_dance("grid_snap", payload(seconds=20.0), max_s=0.0)
    assert sum(kf.duration for kf in anim.keyframes) == pytest.approx(20.0, rel=0.02)
    monkeypatch.setenv(dances.DANCE_MAX_S_ENV, "not-a-number")
    assert dances.dance_max_s() == dances.DANCE_MAX_S_DEFAULT


def test_the_dance_ceiling_is_not_the_emote_ceiling(monkeypatch):
    """Two packs, two knobs. Sharing one number would mean trimming a 5 s dance to fit a 12 s
    shrug, or letting a 12 s emotion hold the queue because dances are short."""
    monkeypatch.setenv(emotes.EMOTE_MAX_S_ENV, "3")
    monkeypatch.setenv(dances.DANCE_MAX_S_ENV, "6")
    assert emotes.emote_max_s() == 3.0 and dances.dance_max_s() == 6.0


@pytest.mark.parametrize("bad", [{"description": "empty"}, {"time": [], "set_target_data": []}])
def test_an_empty_library_file_is_unavailable_not_unknown(bad):
    with pytest.raises(dances.DanceUnavailable):
        dances.parse_dance("jackson_square", bad)


# --- the offline guarantee ---------------------------------------------------


def test_the_resolver_never_touches_the_network(monkeypatch):
    seen: dict = {}

    def fake_download(repo, filename, **kwargs):
        seen.update({"repo": repo, "filename": filename, **kwargs})
        return "/tmp/fake.json"

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_download))
    dances._path_for("groovy_sway_and_roll")
    assert seen["local_files_only"] is True
    assert seen["repo"] == dances.REPO_ID
    assert seen["filename"] == dances.DANCE_FILES["groovy_sway_and_roll"]
    assert seen["repo_type"] == dances.REPO_TYPE


def test_a_cache_miss_becomes_dance_unavailable(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("LocalEntryNotFoundError: no cached file and traffic is off")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=boom))
    with pytest.raises(dances.DanceUnavailable):
        dances.load("dizzy_spin")


def test_a_box_without_huggingface_hub_reports_unavailable_not_a_crash(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(dances.DanceUnavailable) as excinfo:
        dances.load("dizzy_spin")
    assert excinfo.value.name == "dizzy_spin"


def test_unknown_name_raises_before_any_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(dances.UnknownDance):
        dances._path_for("the_twist")


# --- the handler -------------------------------------------------------------


def test_not_connected_returns_a_dict_and_moves_nothing():
    """No robot, no raise - the shell boots and answers with fewer capabilities."""
    from shell.robot_tools import NOT_CONNECTED, make_motion_handlers

    motion = FakeMotion()
    out = make_motion_handlers(types.SimpleNamespace(connected=False, mini=None), motion)["dance"](move="dizzy_spin")
    assert out == NOT_CONNECTED
    assert motion.played == [] and motion.woken == 0


def test_blank_and_garbage_arguments_never_raise():
    motion = FakeMotion()
    play = handler(motion)
    for bad in ("", None, 42, ["dance"], {"move": "dizzy_spin"}, object()):
        out = play(move=bad)
        assert isinstance(out, dict) and out["ok"] is False and set(out) >= {"ok", "error"}
        assert "available" in out or "retry" in out
    assert motion.played == []


def test_unknown_word_says_nothing_about_the_pack():
    motion, journal = FakeMotion(), RecordingJournal()
    motion.journal = journal
    out = handler(motion)(move="the_twist")
    assert out["ok"] is False and "Unknown dance: the_twist" in out["error"]
    assert set(out["available"]) == set(dances.DANCE_NAMES)
    assert "library" not in out["error"].lower() and "pack" not in out["error"].lower()
    assert journal.rows == [], "a wrong word is not an event worth the journal"


def test_unavailable_pack_stops_the_retry_instead_of_inviting_one(monkeypatch):
    motion, journal = FakeMotion(), RecordingJournal()
    motion.journal = journal
    monkeypatch.setattr(dances, "_path_for", _unreadable)
    out = handler(motion)(move="jackson_square")
    assert out["ok"] is False and out["retry"] is False
    assert "do not try another dance" in out["error"]
    assert "available" not in out, "handing back the vocabulary is exactly the retry bait"
    assert journal.rows == [("dance.played", {"name": "jackson_square", "ok": False, "error": out["error"][:120]})]


def test_a_failure_row_is_never_rate_limited(monkeypatch):
    """Two denials, two rows: 'the pack is missing' arriving once at the table is exactly the row
    someone needs to see, and a rate limit that swallows it hides a box-setup fault all evening."""
    motion, journal = FakeMotion(), RecordingJournal()
    motion.journal = journal
    clock = Clock()
    monkeypatch.setattr(dances, "_path_for", _unreadable)
    play = handler(motion, clock=clock)
    play(move="jackson_square")
    clock.advance(0.1)
    play(move="jackson_square")
    assert len(journal.rows) == 2


# --- journaling, rate-limited the way motion events are ----------------------


@NEEDS_PACK
def test_success_journals_what_was_reserved():
    motion, journal = FakeMotion(), RecordingJournal()
    motion.journal = journal
    out = handler(motion)(move="Chicken_Peck")
    assert out["ok"] is True and out["move"] == "chicken_peck"  # case-folded for the model
    assert out["duration_s"] == pytest.approx(1.84, abs=0.2)
    kind, fields = journal.rows[0]
    assert kind == "dance.played"
    assert fields["ok"] is True and fields["keyframes"] == emotes.SUBSAMPLE_TO
    assert fields["duration_s"] == out["duration_s"] and fields["woke"] is False


@NEEDS_PACK
def test_a_repeated_routine_is_journaled_once_but_a_different_one_is_not():
    """`SCAN_JOURNAL_S`'s rule, for the same reason: a guest asking for `dizzy_spin` twice to show
    a friend should not cost two rows, and a ring buffer flushed by repeats loses everything else."""
    motion, journal = FakeMotion(), RecordingJournal()
    motion.journal = journal
    clock = Clock()
    play = handler(motion, clock=clock)
    play(move="chicken_peck")
    clock.advance(1.0)
    play(move="chicken_peck")  # repeat inside the window: played, not journalled
    clock.advance(1.0)
    play(move="yeah_nod")  # different name: journalled
    assert [r[1]["name"] for r in journal.rows] == ["chicken_peck", "yeah_nod"]
    assert len(motion.played) == 3, "the rate limit is on the journal, never on the motion"


@NEEDS_PACK
def test_a_routine_never_preempts_the_live_gaze():
    """The safety answer to a dance fighting the lock-on stack: it does not, because it queues at
    priority 0 and the owner composes it over the gaze with a blend at both ends. If someone
    reinstates the legacy `preempt=True, priority=1`, the head yanks off the guest mid-sentence."""
    motion = FakeMotion()
    handler(motion)(move="chicken_peck")
    assert motion.preempt_flags == [False]
    assert motion.played[0].priority == 0
    assert motion.played[0].blend_in > 0.0 and motion.played[0].blend_out > 0.0


@NEEDS_PACK
def test_a_request_that_needs_the_head_wakes_him():
    motion = FakeMotion(sleeping=True)
    out = handler(motion)(move="chicken_peck")
    assert motion.woken == 1 and out["woke"] is True


@NEEDS_PACK
def test_result_is_json_serialisable():
    json.dumps(handler(FakeMotion())(move="dizzy_spin"))


# --- the real pack, when this box has it -------------------------------------


@NEEDS_PACK
def test_the_whole_pack_parses_offline_and_fits_the_measured_envelope():
    """Three numbers from the report, measured here so they cannot rot into folklore: nothing in
    the pack needs the ceiling today, the servos never see a body turn, and the head stays inside
    the envelope the owner already clamps gaze to."""
    st = dances.status()
    assert len(st["cached"]) == len(dances.DANCE_NAMES), st["missing"]
    assert st["revision"] and len(st["revision"]) == 40
    assert st["bytes"] == 420505
    longest = 0.0
    for name in dances.DANCE_NAMES:
        anim = dances.load(name)
        seconds = sum(kf.duration for kf in anim.keyframes)
        longest = max(longest, seconds)
        assert 1.5 <= seconds <= 5.2, f"{name} ran {seconds}s - outside the pack as pinned"
        for kf in anim.keyframes:
            assert kf.pose.shape == (6,) and np.isfinite(kf.pose).all()
            assert kf.body_yaw is not None and abs(kf.body_yaw) < 1e-6, "the base must never turn"
            roll, pitch, yaw = np.abs(np.degrees(kf.pose[3:6]))
            assert roll <= 25.5 and pitch <= 20.5 and yaw <= 45.5, f"{name}: {roll}, {pitch}, {yaw}"
    assert longest <= dances.DANCE_MAX_S_DEFAULT, "the ceiling is trimming a clip it should not"


# --- coexistence with the live gaze stack ------------------------------------


class _NoRobot:
    """`.mini is None`, which is the hardware-free path: `tick()` still computes a Command and
    `_send()` drops it. The lock-on question is about composition, not about the servos."""

    mini = None
    connected = False


def _owner():
    import random

    from shell.motion.owner import MotionOwner

    clock = Clock()
    owner = MotionOwner(_NoRobot(), None, clock=clock, rng=random.Random(1), hz=50)
    return owner, clock


def test_a_dance_overlays_the_gaze_and_hands_it_back():
    """The safety answer, in code rather than prose. A dance runs while the head is held at +35 deg
    by face lock-on / the idle sweep, and the question is whether the two fight.

    They do not, because of one line in `MotionOwner.tick()`: `pose = base + w * (apose - base)`.
    The gaze spring keeps integrating its target underneath and `w` decides who is audible, so the
    greeter does not have to be told to stand down. Asserted: the pose leaves the gaze's +35 and
    follows the routine; the *gaze target* never moves, so there is no cancellation to undo; and
    the face is handed back when the routine ends. The synthetic payload is all-identity matrices -
    a routine that commands yaw 0 - which makes "who is winning" unambiguous and lets this run in
    CI without the pack. Sampled at the END of the blend-in, because the first ticks are the ramp.
    """
    import math

    owner, clock = _owner()
    owner.look_at(35.0, hold_s=60.0, source="face")
    for _ in range(200):  # let the gaze spring settle onto the face
        clock.advance(0.02)
        owner.tick()
    assert owner.gaze_target_yaw_deg == pytest.approx(35.0, abs=1.0)

    owner.play(dances.parse_dance("dizzy_spin", payload(seconds=1.0, frames=50)))
    during = []
    for _ in range(40):
        clock.advance(0.02)
        cmd = owner.tick()
        if cmd is not None:
            during.append(abs(math.degrees(cmd.pose[5])))
    assert sum(during[-10:]) / 10 < 5.0, "the routine did not take the head off the gaze target"
    assert owner.gaze_target_yaw_deg == pytest.approx(35.0, abs=1.0), "a dance must not move the gaze"

    for _ in range(80):  # the routine is 1 s and blend_out is 0.35 s; then it is the face's again
        clock.advance(0.02)
        cmd = owner.tick()
    assert owner.gaze_target_yaw_deg == pytest.approx(35.0, abs=1.0)
    assert cmd is not None and abs(math.degrees(cmd.pose[5])) > 25.0, "the face was not handed back"


def test_a_dance_never_turns_the_base():
    """`body_yaw` is 0.0 in all 19 files, so the performance is head and ears only and the torso
    joint is never loaded - the most useful thing to have established before waking a servo bus.
    The owner routes body_yaw through a first-order lag, so pinning the COMMANDED value is what
    stands in for the bus tonight."""
    owner, clock = _owner()
    owner.play(dances.parse_dance("jackson_square", payload(seconds=2.0, frames=100)))
    bodies = []
    for _ in range(120):
        clock.advance(0.02)
        cmd = owner.tick()
        if cmd is not None:
            bodies.append(cmd.body_yaw)
    assert bodies and max(abs(b) for b in bodies) < 1e-9, "a dance moved the base"


# --- the real pack, when this box has it -------------------------------------


@NEEDS_PACK
def test_the_pinned_revision_is_the_cached_revision():
    """docs/pins.yaml records a revision the loader can actually see; a cache that drifted to a new
    HEAD with the pin unchanged is the pin doing nothing."""
    text = Path("docs/pins.yaml").read_text()
    assert dances.REPO_ID in text
    revision = dances.status()["revision"]
    assert f"revision: {revision}" in text, f"pins.yaml does not name the cached revision {revision}"
