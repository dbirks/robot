"""The self-hearing verdict in scripts/bench_aec.py, pinned with synthetic rows.

Hardware-free and device-free on purpose: this is the one part of the AEC bench that
CI may run, so nothing here opens a microphone, a speaker, the XVF control interface
or a live journal. Every test hands the module (wall, type, payload) tuples in the
shape rows_from_db() returns them, and the chip functions get a fake handle.

The bugs these pin are the ones that made the old bench lie:
  * a transcript BEFORE the ack is the wake phrase that caused the ack - counting it
    as self-hearing invents an echo (measured on the live journal: -0.03 s pairs);
  * the service re-emits the same turn ~1 s later - counting that is the same lie;
  * zero acks in the window is "never exercised", not a pass;
  * AECCONVERGED read while he is idle is not a far-end-active read.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import bench_aec as ba  # noqa: E402

E = 1_791_857_600.0  # an arbitrary epoch; only deltas matter


def ev(off: float, type_: str, **payload: object) -> ba.Row:
    """One journal row, payload rendered the way shell/journal.py stores it."""
    return (E + off, type_, json.dumps({"env": "home", **payload}))


def ack(off: float, sound: str) -> ba.Row:
    return ev(off, ba.ACK_EVENT, sound=sound)


def said(off: float, text: str) -> ba.Row:
    return ev(off, ba.TRANSCRIPT_EVENT, text=text)


# ---------------------------------------------------------------- clip vocabulary


def test_the_clip_word_comes_from_the_make_acks_filename():
    assert ba.clip_word("hmm-70.wav") == "hmm"
    assert ba.clip_word("sounds/acks/obama/wake/yes-62.wav") == "yes"
    assert ba.clip_word("right-41.wav") == "right"
    # An unknown word is not judged: we cannot say what a transcript should look like.
    assert ba.clip_word("something-else.wav") is None
    assert ba.clip_word("") is None


def test_only_the_known_clip_words_are_judgeable():
    # The corpus is written by scripts/make_acks.py; these are its keep-words.
    assert set(ba.CLIP_WORDS) == {"hmm", "yes", "yeah", "right", "okay", "well"}


def test_parakeet_spellings_of_a_grunt_count_but_a_sentence_does_not():
    assert ba.resembles("Hm.", "hmm")
    assert ba.resembles("Mhm", "hmm")
    assert ba.resembles("Yes.", "yes")
    assert ba.resembles("Rite.", "right")  # one edit, both >= 4 chars
    assert not ba.resembles("Hey Richie what day is today", "hmm")  # a person, 6 tokens
    assert not ba.resembles("Yes.", "hmm")  # a different clip's word
    assert not ba.resembles("", "yes")  # the empty-text rows the journal really has
    assert not ba.resembles("Hm.", None)  # an unknown clip can never be heard


def test_three_tokens_is_the_ceiling_for_an_echo():
    assert ba.resembles("hm hm hm", "hmm")
    assert not ba.resembles("hm hm hm hm", "hmm")


# ---------------------------------------------------------------- ack/transcript pairing


def test_a_self_played_ack_that_comes_back_as_its_own_word_is_self_hearing():
    rows = [said(0.0, "Hey Richie."), ack(0.02, "yes-60.wav"), said(1.4, "Yes.")]
    sh = ba.self_hearing(rows)
    assert sh["heard"] == 1 and sh["state"] == "self-hearing-observed"
    clip = sh["clips"][0]
    assert clip["heard_himself"] is True
    assert clip["transcript"] == "Yes."
    assert clip["transcript_delta_s"] == 1.38  # 1.4 - 0.02: measured from the ACK, not from 0


def test_the_wake_phrase_that_caused_the_ack_is_never_self_hearing():
    """The pairing trap: reactions.py fires the ack AFTER the wake transcript, so the
    transcript that explains an ack sits at a negative delta. On the live journal
    those pairs are 30 ms apart, which any window-on-both-sides test swallows."""
    rows = [said(0.0, "Hey Richie."), ack(0.03, "hmm-70.wav")]
    sh = ba.self_hearing(rows)
    assert sh["heard"] == 0
    assert sh["clips"][0]["transcript"] == ""  # the causal row is not shown as evidence


def test_the_same_turn_re_emitted_after_the_ack_is_not_new_audio():
    """The service re-emits a turn a second later; identical text on both sides of an
    ack is the human twice, not the robot hearing himself."""
    rows = [said(0.0, "Hey Richie."), ack(0.03, "yes-60.wav"), said(1.3, "Hey Richie.")]
    assert ba.self_hearing(rows)["heard"] == 0


def test_him_being_silent_behind_enough_clips_is_the_real_negative():
    rows = [ack(10.0 * i, "hmm-70.wav") for i in range(ba.MIN_ACKS_FOR_A_NO)]
    assert ba.self_hearing(rows)["state"] == "no-self-transcription-observed"


def test_fewer_clips_than_that_is_a_small_number_not_a_result():
    rows = [ack(10.0 * i, "hmm-70.wav") for i in range(ba.MIN_ACKS_FOR_A_NO - 1)]
    assert ba.self_hearing(rows)["state"] == "inconclusive"


def test_no_acks_at_all_is_never_a_pass():
    """Zero clips means nothing was played at him. That is a missing measurement, and
    main() exits non-zero on it so a checklist cannot read it as green."""
    sh = ba.self_hearing([said(0.0, "Hey Richie.")])
    assert sh["acks"] == 0 and sh["state"] == "never-exercised"
    assert ba.self_hearing_state(0, 0) == "never-exercised"


def test_an_ack_whose_word_we_do_not_know_is_not_counted_as_judged():
    rows = [ack(i * 10.0, "mystery-1.wav") for i in range(6)]
    sh = ba.self_hearing(rows)
    assert sh["acks"] == 6 and sh["judged"] == 0
    assert sh["state"] == "never-exercised"


def test_a_long_transcript_that_happens_to_contain_the_word_is_his_opener_not_his_echo():
    rows = [ack(0.0, "right-40.wav"), said(0.8, "Right, so here is the thing about the AEC.")]
    assert ba.self_hearing(rows)["heard"] == 0


# ---------------------------------------------------------------------- gate + lease


def test_a_cut_playback_barge_in_proves_the_gate_was_open_mid_reply():
    rows = [
        ev(0.0, "response.created"),
        ev(1.0, "response.first_audio"),
        ev(4.0, ba.SPEECH_STARTED, cut_playback=True),
        ev(4.1, "response.cancelled"),
    ]
    gate = ba.gate_state(rows)
    assert gate["gate_open_during_reply"] is True
    assert gate["barge_in_cut_playback"] == 1 and gate["replies"] == 1


def test_a_speech_started_without_the_flag_is_not_evidence_of_an_audio_cut():
    rows = [ev(1.0, ba.SPEECH_STARTED, cut_playback=False), ev(2.0, "response.created")]
    assert ba.gate_state(rows)["gate_open_during_reply"] is None


def test_no_barge_in_over_replies_is_unproven_not_closed():
    gate = ba.gate_state([ev(0.0, "response.created"), ev(1.0, "response.completed")])
    assert gate["gate_open_during_reply"] is None and gate["replies"] == 1
    assert gate["barge_in_cut_playback"] == 0


def test_a_lease_that_never_expired_still_covers_the_moment():
    rows = [ev(0.0, "attention.lease_started"), ev(5.0, "attention.lease_renewed")]
    ivs = ba.lease_intervals(rows, now=E + 9.0)
    assert ba.lease_live(ivs, E + 8.0) is True
    assert ba.lease_live(ivs, E + 9.5) is False


def test_an_expired_lease_is_reported_idle():
    rows = [ev(0.0, "attention.lease_started"), ev(3.0, ba.LEASE_CLOSE)]
    assert ba.lease_live(ba.lease_intervals(rows, now=E + 9.0), E + 4.0) is False


def test_self_hearing_reports_whether_a_lease_was_live_at_the_ack():
    rows = [ack(0.0, "hmm-70.wav"), ev(2.0, "attention.lease_started"), ev(9.0, ba.LEASE_CLOSE)]
    clips = ba.self_hearing(rows)["clips"]
    assert clips[0]["lease_live"] is False
    assert (
        ba.self_hearing([ev(2.0, "attention.lease_started"), ack(3.0, "hmm-70.wav")])["clips"][0]["lease_live"] is True
    )


# --------------------------------------------------------- the audible (far-end-live) window


def test_the_reply_window_opens_at_first_audio_not_at_creation():
    """An idle chip read between response.created and the first audio block would be
    sold as a read taken with a live far-end. It is not, so creation opens nothing."""
    rows = [ev(0.0, "response.created"), ev(3.0, "response.first_audio"), ev(8.0, "response.completed")]
    wins = ba.reply_windows(rows, now=E + 20.0)
    assert wins == [(E + 3.0, E + 8.0)]
    assert ba.is_audible(wins, E + 1.0) is False  # created, still silent
    assert ba.is_audible(wins, E + 5.0) is True


def test_a_reply_that_never_closed_runs_to_now():
    wins = ba.reply_windows([ev(3.0, "response.first_audio")], now=E + 12.0)
    assert wins == [(E + 3.0, E + 12.0)] and ba.is_audible(wins, E + 11.0)


def test_a_cancel_closes_the_window():
    wins = ba.reply_windows([ev(3.0, "response.first_audio"), ev(4.0, "response.cancelled")], now=E + 12.0)
    assert wins == [(E + 3.0, E + 4.0)] and ba.is_audible(wins, E + 6.0) is False


# --------------------------------------------------------------------- chip read-back


class FakeChip:
    """The shape that matters: read_values() decodes, read() does not."""

    def __init__(self, values: dict[str, tuple[float, ...] | None]) -> None:
        self.values = values
        self.closed = False

    def read_values(self, name: str):
        if name not in self.values:
            raise OSError("no such control")
        return self.values[name]

    def close(self) -> None:
        self.closed = True


def test_read_numbers_never_raises_and_records_why_a_read_failed():
    dev = FakeChip({"AEC_AECCONVERGED": (1.0,)})
    got = ba.read_numbers(dev, ("AEC_AECCONVERGED", "PP_MIN_NS"))
    assert got["AEC_AECCONVERGED"] == 1.0
    assert got["PP_MIN_NS"] is None and got["PP_MIN_NS_error"] == "OSError"


def test_first_number_refuses_shapes_that_are_not_numbers():
    assert ba.first_number((0.8,)) == 0.8 and ba.first_number((3,)) == 3.0
    assert ba.first_number(None) is None and ba.first_number(True) is None
    assert ba.first_number("2.5") is None  # a `char` read is a string, not a value
    assert ba.first_number([0, 8, 0]) == 0.0  # decoded uint8: the value really is 0


def test_a_reference_that_only_shows_up_mid_reply_is_the_one_that_counts():
    """Idle AECCONVERGED=0 with during-playback 1s is a working AEC; the old bench
    read the idle value and called it the answer."""
    idle = {"AEC_AECCONVERGED": 0.0, "AEC_NUM_FARENDS": 0.0, "PP_ECHOONOFF": 1.0, "PP_MIN_NS": 0.8}
    during = [{"AEC_AECCONVERGED": 1.0, "AEC_NUM_FARENDS": 1.0, "PP_ECHOONOFF": 1.0, "PP_MIN_NS": 0.8}] * 4
    chip = ba.summarize_chip(idle, during)
    assert chip["samples"] == 4 and chip["reference_present"] is True
    assert chip["params"]["AEC_AECCONVERGED"]["idle"] == 0.0
    assert chip["params"]["AEC_AECCONVERGED"]["during"] == [1.0, 1.0, 1.0, 1.0]
    assert any("convergence in all 4" in line for line in chip["verdict"])


def test_no_far_end_while_he_is_playing_is_the_bad_answer():
    idle = {"AEC_AECCONVERGED": 1.0, "AEC_NUM_FARENDS": 1.0}
    during = [{"AEC_AECCONVERGED": 0.0, "AEC_NUM_FARENDS": 0.0, "PP_ECHOONOFF": 1.0}] * 3
    chip = ba.summarize_chip(idle, during)
    assert chip["reference_present"] is False
    assert any("no far-end reference" in line for line in chip["verdict"])


def test_echo_switched_off_in_firmware_is_named_even_with_a_reference_present():
    during = [{"AEC_NUM_FARENDS": 1.0, "AEC_AECCONVERGED": 0.0, "PP_ECHOONOFF": 0.0}] * 2
    chip = ba.summarize_chip({}, during)
    assert chip["reference_present"] is True
    assert any("switched OFF in firmware" in line for line in chip["verdict"])


def test_zero_mid_reply_samples_is_a_missing_measurement_not_a_no():
    chip = ba.summarize_chip({"AEC_AECCONVERGED": 1.0}, [])
    assert chip["samples"] == 0 and chip["reference_present"] is None
    assert any("MISSING measurement" in line for line in chip["verdict"])
    # An idle-only read never produces a reference verdict at all.
    assert ba.summarize_chip({"AEC_AECCONVERGED": 1.0, "AEC_NUM_FARENDS": 1.0}, [])["reference_present"] is None


# ----------------------------------------------------------------------- the summary


def chip_ok() -> dict:
    return ba.summarize_chip(
        {"AEC_AECCONVERGED": 0.0, "AEC_NUM_FARENDS": 0.0, "PP_ECHOONOFF": 1.0},
        [{"AEC_AECCONVERGED": 1.0, "AEC_NUM_FARENDS": 1.0, "PP_ECHOONOFF": 1.0}] * 3,
    )


def test_the_summary_answers_both_decisions_in_order():
    rows = [
        ev(0.0, "response.created"),
        ev(1.0, "response.first_audio"),
        ev(4.0, ba.SPEECH_STARTED, cut_playback=True),
        ev(4.2, "response.cancelled"),
    ]
    lines = ba.summary_lines(ba.self_hearing(rows), ba.gate_state(rows), chip_ok())
    assert sum("GATE OPEN DURING A REPLY" in x for x in lines) == 1
    assert sum("REACH STT" in x for x in lines) == 1
    assert "YES" in lines[0]  # the barge-in decided the gate question
    assert lines[-1].startswith("LIMITS")


def test_the_summary_never_calls_a_bare_run_a_pass():
    lines = ba.summary_lines(ba.self_hearing([]), ba.gate_state([]), chip_ok())
    st = next(x for x in lines if "REACH STT" in x)
    assert "NEVER EXERCISED" in st and "NOT A PASS" in st
    assert "YES" not in st.split("\n")[0]


def test_no_self_transcription_is_worded_as_observation_not_as_working_aec():
    rows = [ack(10.0 * i, "hmm-70.wav") for i in range(8)]
    lines = ba.summary_lines(ba.self_hearing(rows), ba.gate_state(rows), chip_ok())
    st = next(x for x in lines if "REACH STT" in x)
    assert "NO SELF-TRANSCRIPTION OBSERVED" in st and 'not "the AEC works"' in st


def test_the_summary_says_when_the_rms_number_is_the_only_thing_there_is():
    """The honest wording for an idle chip read: no mid-far-end sample, no verdict."""
    lines = ba.summary_lines(ba.self_hearing([ack(0.0, "hmm-70.wav")]), ba.gate_state([]), ba.summarize_chip({}, []))
    assert any("read at idle only" in x or "chip was never read" in x or "MISSING measurement" in x for x in lines)


def test_the_limits_line_names_the_ack_weakness_and_the_rms_confound():
    text = " ".join(ba.summary_lines(ba.self_hearing([]), ba.gate_state([]), chip_ok()))
    assert "RUNNING speech" in text and "x2.7" in text and "ERLE" in text


# ------------------------------------------------------------- the two invented APIs


def test_the_script_defers_its_device_imports_so_ci_can_import_it():
    """`sounddevice` must not be importable at module scope: the CI env has no
    PortAudio and the self-check path must never load a driver."""
    src = (ROOT / "scripts" / "bench_aec.py").read_text()
    top = src.split("def rows_from_db")[0]
    assert "import sounddevice" not in top
    assert "sd.play" in src and "sd.CallbackInputStream" not in src and "sd.read(" not in src
    assert "sd.InputStream" in src
    assert "input(" in src and "sys.stdin.isatty()" in src  # a non-tty run must not hang


def test_read_wav_loads_the_corpus_without_soundfile(tmp_path: Path):
    """sd.read does not exist; the corpus is read with stdlib wave. Stereo and a
    foreign sample rate both have to survive, or step 3 skips its own corpus and
    still prints a clean run."""
    mono = np.zeros(160, dtype="<i2")
    stereo = np.column_stack([np.zeros(160, dtype="<i2"), np.full(160, 1000, dtype="<i2")])
    for name, rate, frames in (("mono16k.wav", 16000, mono), ("stereo8k.wav", 8000, stereo)):
        p = tmp_path / name
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1 if frames.ndim == 1 else 2)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(np.ascontiguousarray(frames).tobytes())
        x = ba.read_wav(p)
        assert x.ndim == 1 and x.dtype == np.float32
        assert abs(len(x) - 160 * (16000 // rate)) <= 2  # 8 kHz got resampled up to SR
    assert float(np.abs(ba.read_wav(tmp_path / "stereo8k.wav")).max()) > 0.0  # the right channel averaged in


def test_rms_and_tone_stay_pure():
    assert ba.rms(np.zeros((0, 2), dtype=np.float32)) == 0.0
    assert ba.rms(np.ones((10, 1))) == 1.0
    t = ba.tone(0.01)
    assert t.dtype == np.float32 and len(t) == 160 and float(np.abs(t).max()) <= 0.35


# ---------------------------------------------------------------------- the self check


def test_self_check_passes_and_opens_nothing(capsys):
    """The CI entry point: `bench_aec.py --self-check` exits 0 with no device, no DB
    and no subprocess. If a pure function regresses, this is the gate that says so."""
    assert ba.main(["--self-check"]) == 0
    out = capsys.readouterr().out
    assert "self-check:" in out and "FAIL" not in out


def test_rows_from_db_reads_only_and_tolerates_a_missing_file(tmp_path: Path):
    assert ba.rows_from_db(tmp_path / "nope.db", 60.0) == []
    import sqlite3

    db = tmp_path / "events.db"
    con = sqlite3.connect(db)
    con.execute("create table events (id integer primary key, mono real, wall real, type text, payload text)")
    con.executemany(
        "insert into events (mono, wall, type, payload) values (?,?,?,?)",
        [
            (1.0, E, ba.TRANSCRIPT_EVENT, '{"text": "Yes."}'),
            (2.0, E - 10_000.0, ba.ACK_EVENT, '{"sound": "yes-60.wav"}'),
        ],
    )
    con.commit()
    con.close()
    rows = ba.rows_from_db(db, 60.0, now=E + 1.0)
    assert [r[1] for r in rows] == [ba.TRANSCRIPT_EVENT]  # the stale-window row is out


# ----------------------------------------------------- the whole script, dry, on a temp DB


def make_db(tmp_path: Path, rows: list[ba.Row]) -> Path:
    import sqlite3

    db = tmp_path / "events.db"
    con = sqlite3.connect(db)
    con.execute("create table events (id integer primary key, mono real, wall real, type text, payload text)")
    con.executemany("insert into events (mono, wall, type, payload) values (1.0,?,?,?)", rows)
    con.commit()
    con.close()
    return db


def test_main_walks_every_section_without_touching_an_audio_device(monkeypatch, tmp_path, capsys):
    """The dry path a reviewer (and CI) can run: route check on a fake wpctl, the real
    self-hearing pass on a temp journal, and a chip that is simply not there - which
    has to fail soft, not crash."""
    monkeypatch.setattr(ba, "default_devices", lambda: ("* Reachy Mini USB Speaker", "* Reachy Mini USB Mic"))
    monkeypatch.setattr(ba, "open_chip", lambda: (None, "no XVF3800 on USB (test)"))
    db = make_db(
        tmp_path,
        [
            ev(0.0, "response.created"),
            ev(1.0, "response.first_audio"),
            ev(3.0, ba.SPEECH_STARTED, cut_playback=True),
            ev(3.2, "response.cancelled"),
            said(100.0, "Hey Richie."),
            ack(100.02, "yes-60.wav"),
            said(101.4, "Yes."),
        ],
    )
    out = tmp_path / "stats.json"
    assert ba.main(["--db", str(db), "--minutes", "1e12", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "SELF-HEARING" in printed and "HEARD-HIMSELF" in printed
    assert "no XVF3800 on USB (test)" in printed  # fail soft, with the reason
    assert "1. GATE OPEN DURING A REPLY: YES" in printed
    stats = json.loads(out.read_text())
    assert stats["self_hearing"]["heard"] == 1
    assert stats["self_hearing"]["state"] == "self-hearing-observed"
    assert stats["gate"]["barge_in_cut_playback"] == 1
    assert stats["chip_error"] == "no XVF3800 on USB (test)"  # a missing readback is recorded, not fatal


def test_a_run_that_heard_nothing_at_all_exits_non_zero(monkeypatch, tmp_path):
    """`never-exercised` must not be green: a checklist that reads exit 0 as a pass
    would ship on a run that measured nothing."""
    monkeypatch.setattr(ba, "default_devices", lambda: ("* Reachy Mini", "* Reachy Mini"))
    monkeypatch.setattr(ba, "open_chip", lambda: (None, "no chip (test)"))
    db = make_db(tmp_path, [said(0.0, "Hey Richie."), ev(1.0, "response.created")])
    assert ba.main(["--db", str(db), "--minutes", "1e12"]) == 1


def test_the_capture_step_refuses_before_it_loads_a_driver(monkeypatch, tmp_path, capsys):
    """ADR 0002: the mic-owning step still refuses a live shell, and it refuses BEFORE
    importing sounddevice - so the refusal itself is safe beside a live session."""
    monkeypatch.setattr(ba, "default_devices", lambda: ("* Reachy Mini", "* Reachy Mini"))
    monkeypatch.setattr(ba, "open_chip", lambda: (None, "no chip (test)"))
    monkeypatch.setattr(ba, "shell_running", lambda: True)
    db = make_db(tmp_path, [ack(0.0, "hmm-70.wav")])
    assert ba.main(["--db", str(db), "--minutes", "1e12", "--echo-rms"]) == 2
    out = capsys.readouterr().out
    assert "REFUSED" in out and "stop reachy-shell" in out


def test_self_hear_asks_no_usb_handle_at_all(monkeypatch, tmp_path, capsys):
    """`--self-hear` is the mode that is safe beside a live session: journal, wpctl
    status, and no XVF control handle - the idle read-back answers nothing about
    playback and the mid-far-end read is opt-in via --chip-during-reply."""

    def no_chip():
        raise AssertionError("--self-hear must not open the XVF control interface")

    monkeypatch.setattr(ba, "default_devices", lambda: ("* Reachy Mini", "* Reachy Mini"))
    monkeypatch.setattr(ba, "open_chip", no_chip)
    db = make_db(tmp_path, [ack(0.0, "yes-60.wav"), said(1.2, "Yes.")])
    assert ba.main(["--self-hear", "--db", str(db), "--minutes", "1e12"]) == 0
    out = capsys.readouterr().out
    assert "skipped by --self-hear" in out and "HEARD-HIMSELF" in out
    assert any(
        "not read this run" in x
        for x in ba.summary_lines(
            ba.self_hearing([]),
            ba.gate_state([]),
            {"skipped": True, "samples": 0, "reference_present": None, "params": {}, "verdict": []},
        )
    )
