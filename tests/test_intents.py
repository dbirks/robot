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


# --- volume: measured 2026-10-13, the model reached for set_volume on 2 of 4 plain ---
# --- requests in a clean session, so the shell does it instead.                      ---


def test_volume_request_reads_the_booth_phrases():
    from shell.intents import volume_request

    for text, want in [
        ("turn the volume up a bit", "louder"),
        ("Hey Reachy, could you turn it up?", "louder"),
        ("much louder please", "much louder"),
        ("way louder", "much louder"),
        ("speak up", "louder"),
        ("turn it down", "quieter"),
        ("that's too loud", "quieter"),
        ("a lot quieter", "much quieter"),
        ("lower your voice", "quieter"),
        ("hey reachy turn your volume down", "quieter"),
        ("can you turn it up", "louder"),
        ("set the volume to 150 percent", "150%"),
        ("volume to 80%", "80%"),
        ("make it louder", "louder"),
    ]:
        assert volume_request(text) == want, f"{text!r} -> {volume_request(text)!r}, want {want!r}"


def test_volume_request_stays_out_of_the_other_intents_lane():
    """'stop talking' and friends must never move the gain - that exact misroute was measured
    live on 2026-10-12, four times, which is why the silence intent exists. Bare 'quiet' and
    'quieter' overlap by a letter, so the boundary is asserted, not assumed."""
    from shell.intents import volume_request, wants_silence, wants_sleep

    for text in ("stop talking", "be quiet", "shut up", "quiet please", "that's enough", "go to sleep", "never mind"):
        assert volume_request(text) is None, f"{text!r} became a volume command"
    assert wants_silence("stop talking") and wants_sleep("go to sleep")
    # ...and the volume phrases are NOT silence/sleep, so the caller's ordering cannot blur them
    assert not wants_silence("turn it down") and not wants_sleep("speak up")


def test_volume_request_refuses_questions_and_self_reference():
    """The read path (`level: "current"`) is for asking; changing the room's volume because
    somebody wondered about it is the false positive that costs the most at a demo."""
    from shell.intents import volume_request

    for text in (
        "what's your volume level?",
        "how loud are you",
        "is it too loud in here?",
        "why are you so loud?",
        "I'll be quieter",
        "we can be softer",
        "don't turn it up",
        "louder or quieter is better for you?",
        "Rachel, pass the salt",
        "Hey Reachy.",
        "can you hear me",
        "what do you see",
        "remember that I prefer standing demos",
    ):
        assert volume_request(text) is None, f"{text!r} should not move the gain"


def test_volume_request_is_bounded_in_length():
    """The word cap is what keeps a paragraph containing "louder" from being a command."""
    from shell.intents import VOLUME_MAX_WORDS, volume_request

    padded = "louder " * VOLUME_MAX_WORDS  # exactly at the cap: still a command
    assert volume_request(padded.strip()) == "louder"
    assert volume_request(padded + "and more words please") is None, "the cap is not being applied"
    rambling = (
        "so before we begin I wanted to ask whether you think it would be better for the room "
        "if the whole demo were a bit louder and slower"
    )
    assert len(rambling.split()) > VOLUME_MAX_WORDS
    assert volume_request(rambling) is None
