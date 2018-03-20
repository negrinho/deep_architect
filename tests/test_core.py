def test_ordered_set():
    from darch.core import OrderedSet

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
    from darch.core import Scope

    scope = Scope()
    e1 = {'name': 'test-name', 'elem': 'arbitrary-val'}
    e2 = {'name': 'test-name2', 'elem': 'arbitrary-val2'}
    scope.register(e1['name'], e1['elem'])
    scope.register(name=e2['name'], elem=e2['elem'])

    assert scope.get_name(e1['elem']) == e1['name']
    assert scope.get_name(e2['elem']) == e2['name']

    assert scope.get_elem(e1['name']) == e1['elem']
    assert scope.get_elem(e2['name']) == e2['elem']


def test_addressable():
    from darch.core import Addressable, Scope

    add = Addressable(Scope(), name='test_addressable')
    assert add.get_name() == 'test_addressable'
    assert str(add) == add.get_name()
    assert add._get_base_name() == 'Addressable'
