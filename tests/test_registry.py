import importlib.util

import pytest

from hallushield.signals import build_signal


def test_lexical_builds():
    assert build_signal("lexical").name == "lexical"


def test_unknown_signal_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown signal"):
        build_signal("nope")


@pytest.mark.skipif(
    importlib.util.find_spec("lettucedetect") is not None,
    reason="lettucedetect installed; cannot assert the missing-extras path",
)
def test_grounding_without_extras_raises_importerror():
    with pytest.raises(ImportError, match="ML extras"):
        build_signal("grounding")
