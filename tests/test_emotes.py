"""The emote layer: vocabulary drift, offline-only loading, and a queue that stays bounded.

Runs in the bare CI environment: `parse_emote` is pure, and `huggingface_hub` is imported lazily
inside the resolver specifically so this file can inject a fake and assert the network is not an
option. Where a test needs the real library it skips out loud rather than passing quietly.
"""

import sys
import types

import numpy as np
import pytest

from shell.motion import emotes
from shell.motion.owner import Animation


def _have_library() -> bool:
    """Can this box actually read a gesture? The library is offline-only by design, so CI has
    neither huggingface_hub nor the cache, and the wiring tests below must skip rather than
    invent a fake library - a fake would test my imagination, not the resolver."""
    try:
        emotes.load("attentive")
    except Exception:
        return False
    return True


HAVE_LIBRARY = _have_library()
NEEDS_LIBRARY = pytest.mark.skipif(
    not HAVE_LIBRARY,
    reason="gesture library is offline-only; this box has neither huggingface_hub nor the cache",
)


def payload(seconds=2.0, frames=100):
    """A library-shaped file: 4x4 head matrices, two antennas, a body yaw, a real time axis."""
    time = [seconds * i / (frames - 1) for i in range(frames)]
    data = [
        {
            "head": np.eye(4).tolist(),
            "antennas": [0.1 * (i % 5) - 0.2, -0.1 * (i % 5) + 0.2],
            "body_yaw": 0.02 * ((i % 7) - 3),
            "check_collision": False,
        }
        for i in range(frames)
    ]
    return {"description": "test", "time": time, "set_target_data": data}


class FakeMotion:
    def __init__(self, journal=None, sleeping=False, connected=True):
        self.played: list[Animation] = []
        self.preempt_flags: list[bool] = []
        self.woken = 0
        self.sleeping = sleeping
        self.journal = journal
        self.connected = connected

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


def handler(motion):
    from shell.robot_tools import make_motion_handlers

    return make_motion_handlers(FakeRobot(), motion)["play_emotion"]


# --- the vocabulary ---------------------------------------------------------


def test_schema_enum_is_the_vocabulary_is_the_file_map():
    """Three sources of truth that must not drift: the schema the model sees, the friendly-name
    map the resolver uses, and EMOTE_NAMES. Rename a file upstream without renaming here and
    this fails; add a name to the schema that the map cannot load and this fails."""
    enum = emotes.EMOTE_TOOL["function"]["parameters"]["properties"]["emotion"]["enum"]
    assert enum == list(emotes.EMOTE_NAMES) == list(emotes.EMOTE_FILES)
    assert len(enum) == len(set(enum)) > 0


def test_the_description_says_a_gesture_is_a_movement_and_only_the_call_makes_one():
    """Written after measuring, not before. Live routing (scripts/bench_emotes.py, hackday,
    qwen3.5-4b) produced two distinct refusals: "I am a machine, I cannot feel surprise" - a dry
    persona declining an instruction to *express a feeling* - and "I am currently playing hello"
    with no tool call at all, narration standing in for the movement. Each clause below targets
    one of them; "at most once" targets the third failure mode from the description audit, which
    is enthusiasm rather than ignorance."""
    text = emotes.EMOTE_TOOL["function"]["description"]
    assert "MOVEMENT" in text, "an instruction to express a feeling is what dry personas decline"
    assert "ONLY way his body moves" in text, "narrating a gesture must be named as doing nothing"
    assert "at most once" in text
    for name in emotes.EMOTE_NAMES:
        assert name in text, f"{name} is callable but never mentioned in the description"


def test_the_shared_prompt_carves_gestures_out_of_the_greeting_ban():
    """TOOLS_NOTE said greetings and goodbyes need no tool while the emote tool said to use hello
    and goodbye for exactly those. Measured consequence: 0 of 4 greetings produced a gesture, on
    every repeat, and the model instead said "I am currently playing hello" while standing still -
    because the same note tells him to narrate tool use. The two instructions had to stop
    contradicting each other; this is the gate that keeps them from starting again."""
    from shell.config import TOOLS_NOTE

    assert "except play_emotion" in TOOLS_NOTE
    assert "narrating a tool call is not" in TOOLS_NOTE


def test_parse_uses_the_files_own_clock_not_a_default_duration():
    anim = emotes.parse_emote("curious", payload(seconds=2.0, frames=400))
    assert len(anim.keyframes) == emotes.SUBSAMPLE_TO
    total = sum(kf.duration for kf in anim.keyframes)
    assert total == pytest.approx(2.0, rel=0.02), "a 2 s gesture must take 2 s"
    assert anim.priority == 0, "an emote queues behind the gaze; it does not fight it"
    assert anim.name == "emote:curious"


def test_every_keyframe_is_a_finite_six_vector_with_antennas_and_yaw():
    anim = emotes.parse_emote("cheerful", payload())
    for kf in anim.keyframes:
        assert kf.pose.shape == (6,) and np.isfinite(kf.pose).all()
        assert kf.antennas is not None and len(kf.antennas) == 2
        assert kf.body_yaw is not None


def test_the_first_and_last_frame_of_the_gesture_survive_subsampling():
    """ "Evenly spaced, always including first and last" is what stops a gesture ending mid-way
    through the pose the library author ended it on."""
    seen: list[int] = []
    orig = emotes._subsample
    emotes._subsample = lambda n, target=emotes.SUBSAMPLE_TO: seen.append(n) or orig(n, target)
    try:
        anim = emotes.parse_emote("sad", payload(seconds=4.0, frames=300))
    finally:
        emotes._subsample = orig
    assert seen == [300]
    assert len(anim.keyframes) == emotes.SUBSAMPLE_TO


def test_the_ceiling_scales_the_gesture_rather_than_truncating_it():
    """`curious1.json` is 12.6 s and `sad1.json` 9.7 s. Dropping keyframes would stop the motion
    mid-shape; shortening each step keeps the gesture and bounds how long it holds the queue."""
    big = payload(seconds=20.0, frames=900)
    anim = emotes.parse_emote("curious", big)
    assert len(anim.keyframes) == emotes.SUBSAMPLE_TO, "frames were dropped, not sped up"
    assert sum(kf.duration for kf in anim.keyframes) == pytest.approx(7.0, rel=0.02)
    steps = [kf.duration for kf in anim.keyframes]
    assert steps == sorted(steps, reverse=True) or len(set(round(s, 6) for s in steps)) == 1


def test_ceiling_zero_is_the_written_length_and_nonsense_is_the_default(monkeypatch):
    monkeypatch.setenv(emotes.EMOTE_MAX_S_ENV, "0")
    assert emotes.emote_max_s() == 0.0
    anim = emotes.parse_emote("curious", payload(seconds=20.0))
    assert sum(kf.duration for kf in anim.keyframes) == pytest.approx(20.0, rel=0.02)
    monkeypatch.setenv(emotes.EMOTE_MAX_S_ENV, "not-a-number")
    assert emotes.emote_max_s() == emotes.EMOTE_MAX_S_DEFAULT


@pytest.mark.parametrize("bad", [{"description": "empty"}, {"time": [], "set_target_data": []}])
def test_an_empty_library_file_is_unavailable_not_unknown(bad):
    """The distinction the whole change turns on: "that is not a word" sends the model to try
    another word; "I cannot read my files" has to stop it trying."""
    with pytest.raises(emotes.EmoteUnavailable):
        emotes.parse_emote("sad", bad)


def test_keyframes_without_head_poses_are_unavailable():
    data = {"description": "d", "time": [0.0, 1.0], "set_target_data": [{"antennas": [0, 0]}] * 4}
    with pytest.raises(emotes.EmoteUnavailable):
        emotes.parse_emote("dance", data)


def test_subsample_edge_cases():
    assert emotes._subsample(0) == []
    assert emotes._subsample(1) == [0]
    assert emotes._subsample(3) == [0, 1, 2]
    idx = emotes._subsample(500)
    assert idx[0] == 0 and idx[-1] == 499 and len(idx) == emotes.SUBSAMPLE_TO


# --- the offline guarantee ---------------------------------------------------


def test_the_resolver_never_touches_the_network(monkeypatch):
    """`local_files_only=True` is the entire difference between a booth that degrades quietly and
    one that spends a 12 s tool timeout on an etag request to a filtered venue network - and then
    tells the model the emotion it asked for does not exist."""
    seen: dict = {}

    def fake_download(repo, filename, **kwargs):
        seen.update({"repo": repo, "filename": filename, **kwargs})
        return "/tmp/fake.json"

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_download))
    emotes._path_for("cheerful")
    assert seen["local_files_only"] is True
    assert seen["repo"] == emotes.REPO_ID and seen["filename"] == emotes.EMOTE_FILES["cheerful"]
    assert seen["repo_type"] == emotes.REPO_TYPE


def test_a_cache_miss_becomes_emote_unavailable(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("LocalEntryNotFoundError: no cached file and traffic is off")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=boom))
    with pytest.raises(emotes.EmoteUnavailable):
        emotes.load("dance")


def test_a_box_without_huggingface_hub_reports_unavailable_not_a_crash(monkeypatch):
    """The contract this whole module exists to keep: the data being unreadable is a *result*, never
    an exception that reaches the caller. Before the import moved inside the try, an environment
    without huggingface_hub raised ModuleNotFoundError out of `load()` - and CI was the only place
    that saw it, because the dev box always has the package. Runs anywhere, by design."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(emotes.EmoteUnavailable) as excinfo:
        emotes.load("dance")
    assert excinfo.value.name == "dance"


def test_unknown_name_raises_before_any_import(monkeypatch):
    """Vocabulary rejection must not need huggingface_hub at all, so a box without the library
    still gets "that is not a word" rather than an import error."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(emotes.UnknownEmote):
        emotes._path_for("wink")


# --- the handler -------------------------------------------------------------


def test_not_connected_reports_the_convention_and_plays_nothing():
    from shell.robot_tools import NOT_CONNECTED, make_motion_handlers

    motion = FakeMotion()
    result = make_motion_handlers(types.SimpleNamespace(connected=False, mini=None), motion)["play_emotion"](
        emotion="cheerful"
    )
    assert result == NOT_CONNECTED
    assert motion.played == []


@NEEDS_LIBRARY
def test_success_journals_what_was_reserved():
    journal, motion = RecordingJournal(), FakeMotion(journal=None)
    motion.journal = journal
    out = handler(motion)(emotion="Attentive")
    assert out["ok"] is True and out["emotion"] == "attentive"  # case-folded for the model
    assert out["duration_s"] == pytest.approx(4.4, abs=0.6)
    assert journal.rows[0][0] == "emotion.played"
    fields = journal.rows[0][1]
    assert fields["ok"] is True and fields["keyframes"] == emotes.SUBSAMPLE_TO
    assert fields["duration_s"] == out["duration_s"]


@NEEDS_LIBRARY
def test_an_emote_never_preempts_the_live_gaze():
    """The legacy call passed preempt=True, priority=1: a "cheerful" took his head off whoever he
    was looking at. Eye contact is the booth's whole illusion."""
    motion = FakeMotion()
    handler(motion)(emotion="cheerful")
    assert motion.preempt_flags == [False]
    assert motion.played[0].priority == 0


@NEEDS_LIBRARY
def test_a_request_that_needs_the_head_wakes_him():
    """Issue #36's rule, applied to a gesture: he must not emote lying down with the microphone
    shut, and being spoken to is the thing that gets him up."""
    motion = FakeMotion(sleeping=True)
    out = handler(motion)(emotion="hello")
    assert motion.woken == 1
    assert out["woke"] is True


def test_unknown_word_says_nothing_about_the_library():
    motion, journal = FakeMotion(), RecordingJournal()
    motion.journal = journal
    out = handler(motion)(emotion="wink")
    assert out["ok"] is False and "Unknown emotion: wink" in out["error"]
    assert set(out["available"]) == set(emotes.EMOTE_NAMES)
    assert "library" not in out["error"].lower()
    assert journal.rows == [], "a wrong word is not an event worth the journal"


def test_unavailable_library_stops_the_retry_instead_of_inviting_one():
    motion, journal = FakeMotion(), RecordingJournal()
    motion.journal = journal
    orig = emotes._path_for
    emotes._path_for = lambda name: (_ for _ in ()).throw(emotes.EmoteUnavailable(name))
    try:
        out = handler(motion)(emotion="sad")
    finally:
        emotes._path_for = orig
    assert out["ok"] is False and out["retry"] is False
    assert "do not try another emotion" in out["error"]
    assert "available" not in out, "handing back the vocabulary is exactly the retry bait"
    assert journal.rows[0][1]["ok"] is False


def test_blank_and_garbage_arguments_never_raise():
    motion = FakeMotion()
    play = handler(motion)
    for bad in ("", None, 42, ["sad"], {"emotion": "sad"}, object()):
        out = play(emotion=bad)
        assert isinstance(out, dict) and out["ok"] is False and set(out) >= {"ok", "error"}
        assert "available" in out or "retry" in out
    assert motion.played == []


@NEEDS_LIBRARY
def test_result_is_json_serialisable():
    import json

    motion = FakeMotion()
    json.dumps(handler(motion)(emotion="dance"))


# --- the real library, when this box has it ---------------------------------


def test_all_ten_resolve_offline_on_a_box_with_the_library():
    try:
        st = emotes.status()
    except Exception as e:  # pragma: no cover - environment, not behaviour
        pytest.skip(f"huggingface_hub unavailable: {e!r}")
    if len(st["cached"]) < 3:
        pytest.skip(f"emotion library not cached here (cached={st['cached']})")
    assert st["revision"] and len(st["revision"]) == 40
    for name in st["cached"]:
        anim = emotes.load(name)
        assert len(anim.keyframes) == emotes.SUBSAMPLE_TO
        assert 0.5 <= sum(kf.duration for kf in anim.keyframes) <= emotes.emote_max_s() + 0.2
