import pytest

from shell.intents import wants_sleep


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
