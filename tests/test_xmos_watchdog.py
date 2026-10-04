from shell.audio.xmos_watchdog import XmosWatchdog


class FakeJournal:
    def __init__(self):
        self.events = []

    def write(self, type_, **payload):
        self.events.append((type_, payload))


def healthy_zeros(**over):
    h = {"muted": False, "pipewire_mute": False, "seconds_since_last_block": 0.03, "seconds_since_sound": 45.0}
    h.update(over)
    return h


def make(clock):
    calls = []
    wd = XmosWatchdog(FakeJournal(), lambda: calls.append(1), silent_s=30, backoff_s=300, clock=lambda: clock[0])
    return wd, calls


def test_reboots_on_sustained_exact_zeros():
    clock = [1000.0]
    wd, calls = make(clock)
    assert wd.check(healthy_zeros()) is True
    assert calls == [1]
    assert wd.journal.events[0][0] == "audio.xmos_reboot"


def test_never_reboots_muted_or_unknown_mute():
    wd, calls = make([0.0])
    assert not wd.check(healthy_zeros(muted=True))
    assert not wd.check(healthy_zeros(pipewire_mute=True))
    assert not wd.check(healthy_zeros(pipewire_mute=None))
    assert calls == []


def test_ignores_short_silence_and_stalled_stream():
    wd, calls = make([0.0])
    assert not wd.check(healthy_zeros(seconds_since_sound=12.0))
    assert not wd.check(healthy_zeros(seconds_since_last_block=9.0))
    assert calls == []


def test_backoff_between_reboots():
    clock = [1000.0]
    wd, calls = make(clock)
    wd.check(healthy_zeros())
    clock[0] += 60
    assert not wd.check(healthy_zeros())
    clock[0] += 300
    assert wd.check(healthy_zeros())
    assert len(calls) == 2


def test_mic_reports_seconds_since_sound(mic):
    from conftest import pump_mic

    pump_mic(mic, value=0)
    h = mic.health()
    assert "seconds_since_sound" in h
    pump_mic(mic, value=500)
    assert mic.health()["seconds_since_sound"] < 1.0
