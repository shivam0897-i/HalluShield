from hallushield.core.text import content_tokens


def test_lowercases_and_drops_stopwords():
    assert content_tokens("The Metformin Dose") == {"metformin", "dose"}


def test_keeps_alphanumeric_units():
    assert "500mg" in content_tokens("Take 500mg now")


def test_drops_single_char_tokens():
    assert content_tokens("a 5 cd") == {"cd"}


def test_all_stopwords_is_empty():
    assert content_tokens("the a of to in") == set()


def test_empty_input():
    assert content_tokens("") == set()
