import json

from shell.memories import MEMORY_TOOLS, MemoryStore, make_memory_handlers


def test_remember_recall_forget_roundtrip(tmp_path):
    path = tmp_path / "memories.json"
    h = make_memory_handlers(MemoryStore(path))
    assert h["remember"](fact="David's favorite color is green.")["ok"]
    assert h["remember"](fact="The spare key is in the blue bowl.")["ok"]
    assert h["remember"](fact="david's favorite color is green.")["saved"]  # dedup, case-insensitive
    store = MemoryStore(path)  # survives a restart
    assert len(store.all()) == 2
    found = make_memory_handlers(store)["recall"](query="where is the key?")["memories"]
    assert "blue bowl" in found[0]  # best match first
    out = make_memory_handlers(store)["forget_memory"](query="spare key")
    assert out["forgotten"] == ["The spare key is in the blue bowl."]
    assert [m["text"] for m in json.loads(path.read_text())] == ["David's favorite color is green."]


def test_empty_and_unknown(tmp_path):
    h = make_memory_handlers(MemoryStore(tmp_path / "m.json"))
    assert not h["remember"](fact="  ")["ok"]
    assert h["recall"](query="anything")["memories"] == []
    assert not h["forget_memory"](query="nothing here")["ok"]


def test_large_store_returns_only_matches(tmp_path):
    s = MemoryStore(tmp_path / "m.json")
    for i in range(40):
        s.add(f"Filler fact number {i}.")
    s.add("The dentist appointment is on Friday.")
    assert [m["text"] for m in s.search("when is the dentist?")] == ["The dentist appointment is on Friday."]


def test_prompt_block_and_schemas(tmp_path):
    s = MemoryStore(tmp_path / "m.json")
    assert s.prompt_block() == ""
    s.add("Bob likes tea.")
    assert "Bob likes tea." in s.prompt_block()
    assert {t["function"]["name"] for t in MEMORY_TOOLS} == {"remember", "recall", "forget_memory"}
