import pytest

from shell.intents import wants_silence, wants_sleep


@pytest.mark.parametrize(
    "text",
    [
        "Hey Reachy, can you go to sleep?",
        "Go to sleep.",
        "Yeah, Reachy, can you go to sleep? Maybe.",
        "Time for bed, Reachy.",
        "Reachy, take a nap.",
        "Okay buddy, shut down for the night.",
    ],
)
def test_sleep_requests(text):
    assert wants_sleep(text)


@pytest.mark.parametrize(
    "text",
    [
        "Don't go to sleep yet.",
        "I'm going to sleep now, goodnight.",
        "I'm going to bed.",
        "We're off to bed, bye Reachy.",
        "Goodnight Reachy!",
        "Bye, see you later.",
        "The kids went to sleep early.",
        "",
    ],
)
def test_not_sleep_requests(text):
    assert not wants_sleep(text)


@pytest.mark.parametrize("text", ["Reachy.", "Hey Richie?", "Okay, Reachy, Reachy!", "Ricci", " hey reachy "])
def test_bare_wake(text):
    from shell.intents import is_bare_wake

    assert is_bare_wake(text)


@pytest.mark.parametrize("text", ["Reachy, what time is it?", "Hey Reachy, go to sleep.", "Yeah.", "", "Richard"])
def test_not_bare_wake(text):
    from shell.intents import is_bare_wake

    assert not is_bare_wake(text)


@pytest.mark.parametrize(
    "text",
    [
        "Stop talking.",
        "stop talking",
        "Be quiet.",
        "quiet please",
        "shut up",
        "That's enough",
        "I said stop talking",
        "hey Reachy be quiet",
        "Say no more",
        "hush",
        "enough",
        "Silence.",
    ],
)
def test_a_demand_to_stop_speaking_is_recognised(text):
    assert wants_silence(text)


@pytest.mark.parametrize(
    "text",
    [
        "Don't stop talking",
        "do not stop talking",
        "I will be quiet now",
        "I'll be quiet",
        "was it quiet in Rome?",
        "the quiet room was nice",
        "tell me about the roman empire",
        "make it quieter",  # a real request for set_volume, must stay one
        "Put it back.",
        "I didn't mean that.",
        "hello",
    ],
)
def test_a_remark_about_quiet_is_not_a_demand(text):
    assert not wants_silence(text), text


def test_quieter_stays_a_volume_request_not_silence():
    """`set_volume` has a 'quieter' step. If stop-detection matched it, asking to be
    heard better would cut his reply mid-sentence instead."""
    assert wants_silence("quieter") is False
    assert wants_silence("a bit quieter please") is False


def test_self_reference_in_the_plain_form_does_not_order_him_around():
    """SELF_RE knew "I'll" but not "I will", so "I'm going to bed" was ignored while
    "I will go to bed" put HIM to sleep. Same class of miss as the stop command."""
    assert wants_sleep("I'm going to bed") is False
    assert wants_sleep("I will go to bed") is False
    assert wants_sleep("we will go to sleep now") is False
    assert wants_sleep("go to sleep") is True
