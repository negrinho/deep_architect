
import darch.core as co
import darch.hyperparameters as hp
import darch.searchers as se

def test_dependent_hyperparameter():
    co.Scope.reset_default_scope()

    h = hp.Discrete([0, 1, 2])
    h_dep = hp.DependentHyperparameter(lambda v: v + 1, {'v' : h})
    h_other = h_dep.get_unset_dependent_hyperparameter()
    se.random_specify_hyperparameter(h_other)
    assert h_other.is_set()
    assert h_dep.is_set()
    print h.get_val(), h_dep.get_val()

    h_dep_other = hp.DependentHyperparameter(lambda v: v + 1, {'v' : h})
    print h_dep_other.get_val()

if __name__ == '__main__':
    test_dependent_hyperparameter()


