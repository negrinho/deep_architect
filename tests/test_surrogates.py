def test_dummy():
    from deep_architect.surrogates import DummySurrogate

    assert DummySurrogate().eval(feats=None) == 0.0
