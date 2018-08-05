
import deep_architect.core as co
import deep_architect.hyperparameters as hp
import deep_architect.searchers as se

def test_dependent_hyperparameter():
    co.Scope.reset_default_scope()

    h = hp.Discrete([0, 1, 2])
    h_dep = hp.DependentHyperparameter(lambda v: v + 1, {'v' : h})
    h_other = h_dep.get_unassigned_dependent_hyperparameter()
    se.random_specify_hyperparameter(h_other)
    assert h_other.has_value_assigned()
    assert h_dep.has_value_assigned()
    print h.get_value(), h_dep.get_value()

    h_dep_other = hp.DependentHyperparameter(lambda v: v + 1, {'v' : h})
    print h_dep_other.get_value()

def create_hyperparameter_chain(h_start, chain_len):
    h_chain = [h_start]
    for _ in xrange(chain_len):
        h_next = hp.DependentHyperparameter(lambda v: v + 1, {'v' : h_chain[-1]})
        h_chain.append(h_next)
    return h_chain

def test_dependent_hyperparameter_chain():
    # check propagation.
    h_start = hp.Discrete([0, 1])
    h_chain = create_hyperparameter_chain(h_start, 8)
    h = h_chain[-1].get_unassigned_dependent_hyperparameter()
    print h_chain[-1].get_name(), h.get_name(), "here"
    assert not h_chain[-1].has_value_assigned()
    assert not h_chain[0].has_value_assigned()
    h_chain[0].assign_value(0)
    assert h_chain[0].has_value_assigned()
    # if the propagation was successfull, this should be true.
    assert h_chain[-1].has_value_assigned()

    # starts with the hyperparameter being specified.
    h_start_other = hp.Discrete([0, 1])
    h_start_other.assign_value(0)
    h_chain_other = create_hyperparameter_chain(h_start, 8)
    assert h_chain_other[0].has_value_assigned()
    assert h_chain_other[-1].has_value_assigned()

# NOTE: probably do the propagation once. I think that this should work in the
# case where there are multiple models.

# depends_on
# DependentHyperparameter([h1, h2, h3])


if __name__ == '__main__':
    # test_dependent_hyperparameter()
    test_dependent_hyperparameter_chain()


