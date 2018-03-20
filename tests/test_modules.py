def test_empty():
    from darch.modules import Empty
    from darch.core import Input

    empty = Empty()
    assert isinstance(empty.outputs, dict) and isinstance(empty.inputs, dict)
    assert 'Out' in empty.outputs
    assert 'In' in empty.inputs

    assert isinstance(empty.inputs['In'], Input)
    assert not empty.inputs['In'].is_connected()
    assert empty.inputs['In'].get_module() == empty

    # try putting a value in input
    arbitray_value = 'arbitrary value'
    empty.inputs['In'].val = arbitray_value
    empty.forward()
    assert empty.outputs['Out'].val == arbitray_value
