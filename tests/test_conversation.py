import asyncio

from shell.conversation import ConversationMemory, render_turns


class FakeJournal:
    def __init__(self):
        self.events = []

    def write(self, type_, **payload):
        self.events.append(type_)


def make(summary="They talked about Yoshi.", fail=False):
    clock = [1_800_000_000.0]
    calls = []

    async def summarize(prior, turns):
        calls.append((prior, list(turns)))
        if fail:
            raise RuntimeError("llm down")
        return summary

    mem = ConversationMemory(FakeJournal(), summarize, compact_after_s=180, reset_after_s=600, clock=lambda: clock[0])
    return mem, clock, calls


def run(coro_fn):
    async def main():
        await coro_fn()

    asyncio.run(main())


def test_compacts_after_idle_then_fresh_session_on_wake():
    mem, clock, calls = make()

    async def go():
        mem.add("user", "Tell me about Yoshi")
        mem.add("assistant", "He's a green dinosaur.")
        clock[0] += 60
        mem.tick(engaged=False)
        assert not mem.compacting  # not idle long enough
        clock[0] += 200
        mem.tick(engaged=False)
        await mem._task
        clock[0] += 3600
        note = mem.on_wake()
        assert note and "Yoshi" in note and "new conversation" in note
        assert mem.turns == []

    run(go)
    assert len(calls) == 1


def test_wake_during_compaction_keeps_existing_session():
    mem, clock, _ = make()

    async def go():
        mem.add("user", "hi")
        clock[0] += 700
        mem.tick(engaged=False)
        assert mem.compacting
        assert mem.on_wake() is None  # summary not ready: keep the session
        await mem._task

    run(go)
    assert "conversation.reset_skipped" in mem.journal.events


def test_short_gap_keeps_session_and_engaged_never_compacts():
    mem, clock, calls = make()

    async def go():
        mem.add("user", "hi")
        clock[0] += 300
        mem.tick(engaged=True)
        assert not mem.compacting
        assert mem.on_wake() is None  # only 5 minutes

    run(go)
    assert calls == []


def test_failed_summary_is_journaled_and_session_kept():
    mem, clock, _ = make(fail=True)

    async def go():
        mem.add("user", "hi")
        clock[0] += 700
        mem.tick(engaged=False)
        await mem._task
        assert mem.on_wake() is None

    run(go)
    assert "conversation.compaction_failed" in mem.journal.events


def test_rolling_summary_includes_prior():
    mem, clock, calls = make()

    async def go():
        mem.add("user", "first")
        clock[0] += 200
        mem.tick(engaged=False)
        await mem._task
        mem.add("user", "second")
        clock[0] += 200
        mem.tick(engaged=False)
        await mem._task

    run(go)
    assert calls[1][0] == "They talked about Yoshi." and [t[2] for t in calls[1][1]] == ["second"]


def test_render_turns():
    assert "User: hi" in render_turns([(1_800_000_000.0, "user", "hi")])


def test_tick_resets_proactively_once_summary_is_ready():
    mem, clock, _ = make()

    async def go():
        mem.add("user", "My name is Dave")
        clock[0] += 200
        assert mem.tick(engaged=False) is None  # starts compaction
        await mem._task
        assert mem.tick(engaged=False) is None  # summarized, but not quiet long enough
        clock[0] += 500
        assert mem.tick(engaged=True) is None  # never while engaged
        note = mem.tick(engaged=False)
        assert note and "Yoshi" in note
        assert mem.tick(engaged=False) is None  # once
        assert mem.on_wake() is None  # the wake word finds it already done

    run(go)
    assert mem.journal.events.count("conversation.reset") == 1


def test_grounded_drops_invented_names_and_vague_facts():
    from shell.conversation import grounded

    said = "my wife Rachel and my son Barney are seven, we live in McCordsville"
    facts = [
        "The user's wife is named Rachel.",
        "The user's wife is named Sarah.",  # never said
        "The user has kids.",  # nothing specific
        "The user has a son named Barney who is 7 years old.",
        "The user lives in McCordsville, Indiana.",  # Indiana never said
    ]
    assert grounded(facts, said) == [facts[0], facts[3]]


def test_parse_facts_tolerates_chatter_and_rejects_non_user_facts():
    from shell.conversation import parse_facts

    raw = 'Sure! ["The user\'s name is David.", "Reachy is a robot.", 42, "The user likes tea."] Hope that helps.'
    assert parse_facts(raw) == ["The user's name is David.", "The user likes tea."]
    assert parse_facts("no json here") == [] and parse_facts("[") == []


def test_compaction_mines_new_turns_for_facts_once():
    saved = []
    clock = [1_800_000_000.0]
    calls = []

    async def summarize(prior, turns):
        return "A summary."

    async def extract(turns):
        calls.append(len(turns))
        return ["The user's wife is named Rachel."]

    mem = ConversationMemory(
        FakeJournal(),
        summarize,
        compact_after_s=180,
        clock=lambda: clock[0],
        extract_facts=extract,
        save_fact=saved.append,
    )

    async def go():
        mem.add("user", "my wife is Rachel")
        mem.add("assistant", "Nice.")
        clock[0] += 200
        mem.tick(engaged=False)
        await mem._task
        mem.add("assistant", "Anything else?")  # no new user speech: nothing to mine
        clock[0] += 200
        mem.tick(engaged=False)
        await mem._task

    run(go)
    assert saved == ["The user's wife is named Rachel."] and calls == [2]
    assert "conversation.facts_saved" in mem.journal.events
