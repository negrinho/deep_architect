def test_dummy():
    from darch.surrogates import DummyModel

    assert DummyModel().eval(feats=None) == 0.0