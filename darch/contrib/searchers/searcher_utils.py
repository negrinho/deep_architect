import darch.core as co
import darch.hyperparameters as hp
import numpy as np

def unset_hyperparameter_iterator(output_lst, hyperp_lst=None):
    if hyperp_lst is not None:
        for h in hyperp_lst:
            if not h.is_set():
                yield h

    while not co.is_specified(output_lst):
        hs = co.get_unset_hyperparameters(output_lst)
        for h in hs:
            if not h.is_set():
                yield h


def random_specify_hyperparameter(hyperp):
    assert not hyperp.is_set()

    if isinstance(hyperp, hp.Discrete):
        v = hyperp.vs[np.random.randint(len(hyperp.vs))]
        hyperp.set_val(v)
    else:
        raise ValueError
    return v


def random_specify(output_lst, hyperp_lst=None):
    vs = []
    for h in unset_hyperparameter_iterator(output_lst, hyperp_lst):
        v = random_specify_hyperparameter(h)
        vs.append(v)
    return vs


def specify(output_lst, vs, hyperp_lst=None):
    for i, h in enumerate(unset_hyperparameter_iterator(output_lst, hyperp_lst)):
        h.set_val(vs[i])