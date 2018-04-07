def test_dummy():
    from darch.surrogates import DummySurrogate

    assert DummySurrogate().eval(feats=None) == 0.0
