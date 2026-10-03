from shell.attention.kws import _keywords_file


def test_keywords_file_stamps_per_keyword_thresholds(tmp_path):
    (tmp_path / "reachy-keywords.txt").write_text(
        "▁RE A CH Y @Reachy\n▁HE Y ▁RE A CH Y @hey_Reachy\n▁RO B O T @robot\n"
    )
    out = _keywords_file(str(tmp_path), {"Reachy": 0.25, "hey Reachy": 0.2}, 0.3)
    assert open(out).read().splitlines() == [
        "▁RE A CH Y @Reachy #0.25",
        "▁HE Y ▁RE A CH Y @hey_Reachy #0.2",
        "▁RO B O T @robot #0.3",
    ]


def test_keywords_file_missing_points_at_installer(tmp_path):
    try:
        _keywords_file(str(tmp_path), {}, 0.3)
    except FileNotFoundError as e:
        assert "get_kws_model.sh" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")
