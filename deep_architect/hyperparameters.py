from six import itervalues, iteritems
import numpy as np
from collections import OrderedDict
import deep_architect.core as co


class HyperparameterSharer:
    """Dictionary of hyperparameters used to help share hyperparameters between
    modules.

    Used to implement different sharing patterns of hyperparameters.
    Hyperparameters are registered with a name and a function that creates the
    hyperparameter and stores it in a dictionary. Once created, following calls
    get the hyperparameter from the dictionary.
    """

    def __init__(self):
        self.name_to_h_fn = OrderedDict()
        self.name_to_h = OrderedDict()

    def register(self, hyperp_name, hyperp_fn):
        """Registers a function that can be called to create an hyperparameter
        with a given name.

        Delayed creation of the hyperparameter is implemented through a thunk.

        Args:
            hyperp_name (str): Hyperparameter name to associate with the function.
                The hyperparameter name must be unique.
            hyperp_fn (() -> (deep_architect.core.Hyperparameter)): Function that returns
                an hyperparameter when called.
        """
        assert hyperp_name not in self.name_to_h_fn
        self.name_to_h_fn[hyperp_name] = hyperp_fn

    def get(self, hyperp_name):
        """Gets the hyperparameter with the desired name.

        If the hyperparameter has not been created yet, it calls the
        function to create it, stores it in the dictionary, and returns it.
        If the hyperparameter has already been created, it returns it from the
        dictionary. Asserts ``False`` if the name is not registered in the sharer.

        Args:
            hyperp_name (str): Hyperparameter name.

        Returns:
            deep_architect.core.Hyperparameter: Hyperparameter associated to ``hyperp_name``.
        """
        assert hyperp_name in self.name_to_h_fn

        if hyperp_name not in self.name_to_h:
            self.name_to_h[hyperp_name] = self.name_to_h_fn[hyperp_name]()
        return self.name_to_h[hyperp_name]


class Discrete(co.Hyperparameter):
    """List valued hyperparameter.

    This type of hyperparameter has a finite number of possible values.

    Args:
        vs (list[object]): List of possible parameter values that the
            hyperparameter can take.
        scope (deep_architect.core.Scope, optional): The scope in which to register the
            hyperparameter in.
        name (str, optional): Name from which the name of the hyperparameter
            in the scope is derived.
    """

    def __init__(self, vs, scope=None, name=None):
        assert len(vs) > 0
        co.Hyperparameter.__init__(self, scope, name)
        self.vs = vs

    def _check_value(self, val):
        """Checks if the chosen values is in the list of valid values.

        Asserts ``False`` if the value is not in the list.
        """
        if val not in self.vs:
            print(self.get_name())
            print(self.vs)
            print(val)
        assert val in self.vs


class Bool(Discrete):

    def __init__(self, scope=None, name=None):
        Discrete.__init__(self, [0, 1], scope, name)


class OneOfK(Discrete):

    def __init__(self, k, scope=None, name=None):
        Discrete.__init__(self, range(k), scope, name)


class OneOfKFactorial(Discrete):

    def __init__(self, k, scope=None, name=None):
        Discrete.__init__(self, range(np.product(np.arange(1, k + 1))), scope,
                          name)


# abbreviations
D = Discrete
