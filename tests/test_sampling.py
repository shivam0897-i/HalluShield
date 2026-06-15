from benchmarks.ragtruth_eval import demo_examples, sample_examples


def test_sample_caps_and_is_deterministic():
    exs = demo_examples()  # 4 examples
    s1 = sample_examples(exs, 2, seed=0)
    s2 = sample_examples(exs, 2, seed=0)
    assert len(s1) == 2
    assert [e.id for e in s1] == [e.id for e in s2]  # same seed -> same sample


def test_sample_returns_all_when_limit_zero_or_larger():
    exs = demo_examples()
    assert len(sample_examples(exs, 0)) == len(exs)
    assert len(sample_examples(exs, 999)) == len(exs)


def test_sample_is_a_subset():
    exs = demo_examples()
    ids = {e.id for e in exs}
    assert {e.id for e in sample_examples(exs, 3, seed=1)} <= ids
