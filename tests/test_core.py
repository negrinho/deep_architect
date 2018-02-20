from darch.core import *


def test_ordered_set():
    ord_s = OrderedSet()
    assert len(ord_s) == 0
    assert 'test' not in ord_s
    ord_s.add('test')
    assert 'test' in ord_s

    test_list = ['test' + str(i) for i in range(100)]
    cmp_list = ['test']
    cmp_list.extend(test_list)
    ord_s.update(test_list)
    assert [x for x in ord_s] == cmp_list


def test_scope():
    scope = Scope()
    elem = 2  # TODO change this to something that makes more sense
    scope.register(elem, 'test-name')
    assert scope.get_name(elem) == 'test-name'


def test_addressable():
    add = Addressable(Scope(), name='test_addressable')
    assert add.get_name() == 'test_addressable'
    assert str(add) == add.get_name()
    assert add._get_base_name() == 'Addressable'


def test_hyperparameter():
    name = 'test_hyp'
    val = 2
    hyp = Hyperparameter(name=name)
    hyp2 = Hyperparameter()

    assert not hyp.is_set()
    hyp.set_val(val)
    assert hyp.is_set()
    assert hyp.get_val() == val
