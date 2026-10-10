import math

from shell.audio.xmos_tuning import XmosTuner


class FakeJournal:
    def __init__(self):
        self.events = []

    def write(self, type_, **payload):
        self.events.append((type_, payload))


class FakeDev:
    def __init__(self, agc_on=1, max_gain=64.0, fail=False):
        self.regs = {"PP_AGCONOFF": [0, agc_on, 0, 0, 0], "PP_AGCMAXGAIN": (max_gain,), "AEC_FIXEDBEAMSONOFF": [0, 0]}
        self.writes = []
        self.fail = fail

    def read(self, name):
        if self.fail:
            raise OSError("No such device")
        return self.regs.get(name)

    def write(self, name, values):
        if self.fail:
            raise OSError("No such device")
        self.writes.append((name, list(values)))
        if name in ("PP_AGCONOFF", "AEC_FIXEDBEAMSONOFF"):
            self.regs[name] = [0, values[0], 0, 0, 0]
        else:
            self.regs[name] = tuple(values)

    def close(self):
        pass


def test_agc_cap_applied_once_and_reapplied_after_reboot():
    dev = FakeDev()
    j = FakeJournal()
    t = XmosTuner(j, agc_max_gain=10.0, open_device=lambda: dev)
    t.ensure()
    assert dev.regs["PP_AGCMAXGAIN"] == (10.0,)
    t.ensure()
    assert len(dev.writes) == 1  # idempotent
    dev.regs["PP_AGCMAXGAIN"] = (64.0,)  # XMOS REBOOT restores boot defaults
    t.ensure()
    assert len(dev.writes) == 2
    assert [e for e, _ in j.events] == ["xmos.tuned", "xmos.tuned"]


def test_agc_off_when_gain_zero():
    dev = FakeDev()
    XmosTuner(FakeJournal(), agc_max_gain=0, open_device=lambda: dev).ensure()
    assert dev.writes == [("PP_AGCONOFF", [0])]


def test_focus_and_release():
    dev = FakeDev()
    j = FakeJournal()
    t = XmosTuner(j, open_device=lambda: dev)
    t.focus(None)  # no DOA at wake: leave the beams alone
    assert dev.writes == []
    t.focus(math.pi / 3)
    assert ("AEC_FIXEDBEAMSAZIMUTH_VALUES", [math.pi / 3, math.pi / 3]) in dev.writes
    assert dev.writes[-1] == ("AEC_FIXEDBEAMSONOFF", [1])
    assert t.focused is not None
    t.release()
    assert dev.writes[-1] == ("AEC_FIXEDBEAMSONOFF", [0])
    assert t.focused is None
    t.release()  # idempotent
    assert dev.writes.count(("AEC_FIXEDBEAMSONOFF", [0])) == 1


def test_beam_focus_disabled_and_dead_device_never_raise():
    dev = FakeDev()
    XmosTuner(FakeJournal(), beam_focus=False, open_device=lambda: dev).focus(1.0)
    assert dev.writes == []
    dead = XmosTuner(FakeJournal(), open_device=lambda: FakeDev(fail=True))
    dead.ensure()
    dead.focus(1.0)
    assert dead.focused is None
    XmosTuner(FakeJournal(), open_device=lambda: None).ensure()
