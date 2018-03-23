from six import itervalues, iteritems
import numpy as np
from collections import OrderedDict
import darch.core as co


class HyperparameterSharer:
    """Used to implement different sharing patterns of hyperparameters.

    Hyperparameters are registered with a name and a function that creates the
    hyperparameter and stores it in a dictionary. Once created, following calls
    get the hyperparameter from the dictionary.
    """
    def __init__(self):
        self.name_to_h_fn = OrderedDict()
        self.name_to_h = OrderedDict()

    def register(self, hyperp_name, hyperp_fn):
        assert hyperp_name not in self.name_to_h_fn
        self.name_to_h_fn[hyperp_name] = hyperp_fn

    def get(self, hyperp_name):
        assert hyperp_name in self.name_to_h_fn

        if hyperp_name not in self.name_to_h:
            self.name_to_h[hyperp_name] = self.name_to_h_fn[hyperp_name]()
        return self.name_to_h[hyperp_name]


class DependentHyperparameter(co.Hyperparameter):

    def __init__(self, fn, hs, scope=None, name=None):
        co.Hyperparameter.__init__(self, scope, name)

        self._hs = hs
        self._fn = fn

    def is_set(self):
        if self.set_done:
            return True
        else:
            if all(h.is_set() for h in itervalues(self._hs)):
                kwargs = {name: h.get_val() for name, h in iteritems(self._hs)}
                self.set_val(self._fn(**kwargs))

            return self.set_done

    def get_unset_dependent_hyperparameter(self):
        """
        :rtype: darch.core.Hyperparameter
        """
        assert not self.set_done
        for h in itervalues(self._hs):
            if not h.is_set():
                return h

    def _check_val(self, val):
        pass


class Discrete(co.Hyperparameter):
    def __init__(self, vs, scope=None, name=None):
        """
        :type vs: collections.Iterable
        :param vs: List of possible parameter values.
        """
        co.Hyperparameter.__init__(self, scope, name)
        self.vs = vs

    def _check_val(self, val):
        assert val in self.vs


class Bool(Discrete):
    def __init__(self, scope=None, name=None):
        Discrete.__init__(self, [0, 1], scope, name)


class OneOfK(Discrete):
    def __init__(self, k, scope=None, name=None):
        """
        Equivalent to `Discrete(range(k))`.

        :param k: Maximum value to try.
        :type k: int
        """
        Discrete.__init__(self, range(k), scope, name)


class OneOfKFactorial(Discrete):
    def __init__(self, k, scope=None, name=None):
        """
        Equivalent to `Discrete(np.product(np.arange(1, k + 1)))`

        :type k: int
        """
        Discrete.__init__(self, np.product(np.arange(1, k + 1)), scope, name)
