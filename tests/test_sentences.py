from hallushield.core.sentences import split_sentences


def test_basic_split():
    assert split_sentences("First sentence. Second sentence.") == [
        "First sentence.",
        "Second sentence.",
    ]


def test_real_boundary_still_splits():
    assert split_sentences("The dose is 500mg. It cures nothing.") == [
        "The dose is 500mg.",
        "It cures nothing.",
    ]


def test_no_split_on_abbreviation_eg():
    assert split_sentences("Use a beta-blocker, e.g. Metoprolol, daily.") == [
        "Use a beta-blocker, e.g. Metoprolol, daily."
    ]


def test_no_split_on_title_abbreviation():
    assert len(split_sentences("Dr. Smith prescribed it.")) == 1


def test_no_split_on_vs():
    assert len(split_sentences("Compare A vs. B here.")) == 1


def test_no_split_on_initial():
    assert len(split_sentences("Written by A. Smith today.")) == 1


def test_empty_and_single():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
    assert split_sentences("Just one clause") == ["Just one clause"]
