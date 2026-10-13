"""The emote routing bench, minus the LLM.

scripts/bench_emotes.py needs a running llama.cpp, so it cannot be a CI test. What CAN live here
is everything that would make its numbers meaningless: the case list's internal consistency, the
grading rules, and the gates. A bench whose expected values drift from the real vocabulary, or
whose "ambient speech" group quietly gains an `emote` case, produces a confident wrong number -
which is worse than no number, because it gets quoted.

The last test here is the important one: importing this script and running --list must construct
no HTTP client at all. If someone adds a module-level client (or a call at import time), collecting
this file would fire 42 requests at a GPU that is already carrying the booth.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench_emotes", ROOT / "scripts" / "bench_emotes.py")
be = importlib.util.module_from_spec(spec)
sys.modules["bench_emotes"] = be
spec.loader.exec_module(be)

from shell.motion import emotes  # noqa: E402

KINDS = {"emote", "no_emote", "either"}


def shot(utterance, chosen=None, others=(), text="ok", error=""):
    return be.Shot(
        utterance=utterance, repeat=0, chosen=chosen, other_tools=others, text=text, error=error, seconds=0.5
    )


# --- the case list ---------------------------------------------------------


def test_case_list_is_internally_consistent():
    assert be.CASES, "an empty bench measures nothing"
    seen = set()
    for case in be.CASES:
        assert case.expect in KINDS, f"{case.utterance}: unknown expectation {case.expect}"
        assert case.utterance.strip() == case.utterance and case.utterance not in seen, (
            f"duplicate or untrimmed utterance: {case.utterance!r} - two identical cases would be"
            " counted twice in the same denominator"
        )
        seen.add(case.utterance)
        assert case.why, f"{case.utterance}: a case nobody explains will be 'fixed' by deleting it"
        assert case.group, f"{case.utterance}: ungrouped cases cannot be gated"
        if case.expect == "emote":
            assert case.allow, f"{case.utterance}: an `emote` case with no acceptable answer"
        if case.expect == "no_emote":
            assert not case.allow, f"{case.utterance}: `no_emote` cannot allow gestures"
        for name in case.allow:
            assert name in be.ALL_EMOTES, f"{case.utterance}: expects {name!r}, which is not in the library"


def test_the_two_groups_that_must_never_emote_are_still_all_no_emote():
    """The one structural guarantee in this file.

    `silence` cases are handled by shell/intents.py before the model ever sees them, and `ambient`
    cases must not be acted on at all (ADR 0003: room speech is not an instruction). If either
    group ever gains a case that *invites* a gesture, the gate in gates() stops meaning what it
    says and a regression in the room-speech firewall shows up as a shrug.
    """
    for group in ("silence", "ambient", "collision", "negation"):
        cases = [c for c in be.CASES if c.group == group]
        assert cases, f"the {group!r} group went missing - a whole failure mode unmeasured"
        assert all(c.expect == "no_emote" for c in cases), (
            f"{group}: {[c.utterance for c in cases if c.expect != 'no_emote']} - this group may"
            " only contain cases where a gesture is a defect"
        )


def test_every_gesture_in_the_library_is_asked_for_directly_somewhere():
    """A name that only ever appears in `either` cases has never been *required*, so a typo in the
    library mapping (dance -> dace) could pass a full green run."""
    required = set().union(*(c.allow for c in be.CASES if c.expect == "emote"))
    missing = set(be.ALL_EMOTES) - required
    assert not missing, f"no case requires these directly: {sorted(missing)}"


def test_the_case_set_covers_our_real_confusions():
    """Each of these collided with something else at some point tonight; the groups exist to keep
    the collisions from being "solved" by deleting the awkward cases."""
    by_group = {g: [c.utterance for c in be.CASES if c.group == g] for g in {c.group for c in be.CASES}}
    assert any("salt" in u for u in by_group["ambient"]), "room speech must stay in the bench"
    assert any("turn it down" in u for u in by_group["collision"]), "volume is loudness, not feeling"
    assert any("time" in u for u in by_group["collision"]), "the clock must not be emoted at"
    assert any("Stop talking" in u for u in by_group["silence"])
    assert any("not sad" in u for u in by_group["negation"])


# --- extraction ------------------------------------------------------------


class FakeCall:
    def __init__(self, calls):
        self.tool_calls = calls


def test_play_emotion_arguments_are_extracted_and_other_tools_kept_separate():
    chosen, others = be.called_emote(
        FakeCall([("play_emotion", {"emotion": "Dance"}), ("get_time", {}), ("look_left", {})])
    )
    assert chosen == "dance", "case-folding matters: the schema's enum is lowercase, models are not"
    assert others == ("get_time", "look_left")


def test_a_made_up_emotion_is_visible_not_silently_none():
    """If `wink` came back as None, a hallucinated gesture would look identical to declining to
    gesture, and the bench would score an obvious failure as an acceptable abstention."""
    chosen, _ = be.called_emote(FakeCall([("play_emotion", {"emotion": "wink"})]))
    assert chosen == "invalid:wink"


def test_no_tool_calls_reads_as_no_emote():
    chosen, others = be.called_emote(FakeCall([]))
    assert chosen is None and others == ()


def test_multiple_emotion_calls_report_the_first_and_keep_the_rest_as_others():
    chosen, others = be.called_emote(
        FakeCall([("play_emotion", {"emotion": "dance"}), ("play_emotion", {"emotion": "sad"})])
    )
    assert chosen == "dance" and others == ("play_emotion",)


# --- grading ---------------------------------------------------------------


def test_explicit_request_hit_and_miss():
    case = be.Case("Do a dance.", "emote", frozenset({"dance"}), "test", "explicit")
    hit = be.grade(case, [shot("Do a dance.", "dance"), shot("Do a dance.", "dance")])
    assert hit.kind == "hit" and hit.agreement == 1.0
    partial = be.grade(case, [shot("Do a dance.", "dance"), shot("Do a dance.", "cheerful")])
    assert partial.kind == "miss", "one right answer out of two is not a hit for an explicit ask"
    assert partial.agreement == 0.5, "and the disagreement is the interesting number"
    none = be.grade(case, [shot("Do a dance."), shot("Do a dance.")])
    assert none.kind == "miss" and "never emoted" in none.detail


def test_a_frozen_model_that_is_wrong_is_still_wrong_consistently():
    """The failure this bench exists to catch: agreement 1.00 with accuracy 0.59 was tonight's
    first measurement. Reporting only agreement would have called that 'consistent behaviour'."""
    case = be.Case("Look sad.", "emote", frozenset({"sad"}), "test", "explicit")
    verdict = be.grade(case, [shot("Look sad.") for _ in range(3)])
    assert verdict.agreement == 1.0
    assert verdict.kind == "miss"


def test_no_emote_cases_only_pass_by_not_gesturing():
    case = be.Case("What time is it?", "no_emote", frozenset(), "test", "collision")
    assert be.grade(case, [shot("What time is it?")]).kind == "hit"
    gestured = be.grade(case, [shot("What time is it?", "curious")])
    assert gestured.kind == "miss" and "should not have" in gestured.detail
    # calling a DIFFERENT tool is correct behaviour here, and must not be scored as an emote miss
    other = be.grade(case, [shot("What time is it?", None, ("get_time",))])
    assert other.kind == "hit"


def test_either_cases_accept_words_and_gestures_but_not_random_gestures():
    case = be.Case("Hi!", "either", frozenset({"hello", "cheerful"}), "test", "greeting")
    assert be.grade(case, [shot("Hi!")]).kind == "abstain_ok"
    assert be.grade(case, [shot("Hi!", "hello")]).kind == "abstain_ok"
    wrong = be.grade(case, [shot("Hi!", "dance")])
    assert wrong.kind == "miss" and "not a reading" in wrong.detail


def test_all_error_shots_are_reported_as_errors_not_as_failures_of_the_prompt():
    case = be.Case("Do a dance.", "emote", frozenset({"dance"}), "test", "explicit")
    verdict = be.grade(case, [shot("Do a dance.", error="aborted at 45s"), shot("Do a dance.", error="429")])
    assert verdict.kind == "error"
    assert "aborted" in verdict.detail and "429" in verdict.detail


def test_agreement_counts_all_repeats_not_just_the_first():
    case = be.Case("Hi!", "either", frozenset({"hello"}), "test", "greeting")
    verdict = be.grade(case, [shot("Hi!", "hello"), shot("Hi!", "hello"), shot("Hi!")])
    assert verdict.agreement == pytest.approx(2 / 3)


# --- the gates -------------------------------------------------------------


def rep_of(pairs):
    """pairs: (case, shots) -> summary dict, through the real code path."""
    verdicts = [be.grade(c, s) for c, s in pairs]
    return be.summary(verdicts), verdicts


def test_an_ambient_leak_fails_the_run_even_at_perfect_overall_accuracy():
    """The property that makes this a safety gate and not a scoreboard: one animation triggered by
    two strangers talking is a booth-visible defect, and it must not be averaged away."""
    polite = be.Case("Do a dance.", "emote", frozenset({"dance"}), "test", "explicit")
    leak = be.Case("Rachel, pass the salt.", "no_emote", frozenset(), "test", "ambient")
    rep, _ = rep_of(
        [(polite, [shot(polite.utterance, "dance")]) for _ in range(19)] + [(leak, [shot(leak.utterance, "curious")])]
    )
    assert rep["accuracy"] > 0.9, f"setup: {rep['accuracy']}"
    failures = be.gates(rep, 0.8, 0.75)
    assert any("ambient" in f for f in failures), failures


def test_silence_leaks_gate_the_same_way():
    polite = be.Case("Do a dance.", "emote", frozenset({"dance"}), "test", "explicit")
    leak = be.Case("Stop talking.", "no_emote", frozenset(), "test", "silence")
    rep, _ = rep_of([(polite, [shot(polite.utterance, "dance")])] * 9 + [(leak, [shot(leak.utterance, "sad")])])
    assert any("silence" in f for f in be.gates(rep, 0.5, 0.5))


def test_a_run_where_everything_errored_blames_the_endpoint_not_the_prompt():
    down = be.Case("Do a dance.", "emote", frozenset({"dance"}), "test", "explicit")
    rep, _ = rep_of([(down, [shot(down.utterance, error="connection refused")])])
    failures = be.gates(rep, 0.8, 0.75)
    assert failures == ["every call errored (1) - the endpoint, not the prompt"], failures


def test_errors_are_kept_out_of_the_accuracy_denominator_and_still_counted():
    good = be.Case("Do a dance.", "emote", frozenset({"dance"}), "test", "explicit")
    bad = be.Case("Look sad.", "emote", frozenset({"sad"}), "test", "explicit")
    rep, _ = rep_of(
        [
            (good, [shot(good.utterance, "dance")]),
            (bad, [shot(bad.utterance, error="aborted at 45s, no answer")]),
        ]
    )
    assert rep["scored"] == 1 and rep["errors"] == 1
    assert rep["accuracy"] == 1.0, "an endpoint timeout is not a routing decision"


# --- the CI-safety guarantee ------------------------------------------------


def test_listing_cases_and_importing_the_script_make_no_http_connection(monkeypatch, capsys):
    """Collecting this file must never put load on a GPU that is carrying the booth."""
    calls = []

    class NoClient:
        def __init__(self, *a, **k):
            calls.append("constructed")
            raise AssertionError("bench_emotes opened an HTTP client during import/--list")

        def __enter__(self):
            raise AssertionError("unreachable")

    monkeypatch.setattr(be.httpx, "Client", NoClient)
    assert be.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "cases;" in out and calls == [], "a client was constructed while merely listing cases"
    assert be.main(["--list", "--group", "silence"]) == 0
    assert "Stop talking" in capsys.readouterr().out


def test_group_filter_accepts_several_groups_without_a_network_call(monkeypatch, capsys):
    monkeypatch.setattr(be.httpx, "Client", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    assert be.main(["--list", "--group", "silence,ambient"]) == 0
    out = capsys.readouterr().out
    assert "salt" in out and "Stop talking" in out
