import numpy as np
import deep_architect.core as co
import deep_architect.hyperparameters as hp


class Searcher:
    """Abstract base class from which new searchers should inherit from.

    A search takes a function that when called returns a new search space, i.e.,
    a search space where all the hyperparameters have not being specified.
    Searchers essentially sample a sequence of models in the search space by
    specifying the hyperparameters sequentially. After the sampled architecture
    has been evaluated somehow, the state of the searcher can be updated with
    the performance information, guaranteeing that future architectures
    are sampled from the search space in a more informed manner.

    Args:
        search_space_fn (() -> (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output], dict[str,deep_architect.core.Hyperparameter])):
            Search space function that when called returns a dictionary of
            inputs, dictionary of outputs, and dictionary of hyperparameters
            encoding the search space from which models can be sampled by
            specifying all hyperparameters (i.e., both those arising in the
            graph part and those in the dictionary of hyperparameters).
    """

    def __init__(self, search_space_fn):
        self.search_space_fn = search_space_fn

    def sample(self):
        """Returns a model from the search space.

        Models are encoded via a dictionary of inputs, a dictionary of outputs,
        and a dictionary of hyperparameters. The forward computation for the
        model can then be done as all values for the hyperparameters have been
        chosen.

        Returns:
            (dict[str, deep_architect.core.Input], dict[str, deep_architect.core.Output], dict[str, deep_architect.core.Hyperparameter], list[object], dict[str, object]):
                Tuple encoding the model sampled from the search space.
                The positional arguments have the following semantics:
                1: Dictionary of names to inputs of the model.
                2: Dictionary of names to outputs of the model.
                3: List with list of values that can be to replay the sequence
                of values assigned to the hyperparameters, and therefore,
                reproduce, given the search space, the model sampled.
                4: Searcher evaluation token that is sufficient for the searcher
                to update its state when combined with the results of the
                evaluation.
        """
        raise NotImplementedError

    def update(self, val, searcher_eval_token):
        """Updates the state of the searcher based on the searcher token
        for a particular evaluation and the results of the evaluation.

        Args:
            val (object): Result of the evaluation to use to update the state of the searcher.
            searcher_eval_token (dict[str, object]): Searcher evaluation token
                that is sufficient for the searcher to update its state when
                combined with the results of the evaluation.
        """
        raise NotImplementedError


# TODO: generalize this for other types of hyperparameters. currently only supports
# discrete hyperparameters.
def random_specify_hyperparameter(hyperp):
    """Choose a random value for an unspecified hyperparameter.

    The hyperparameter becomes specified after the call.

    hyperp (deep_architect.core.Hyperparameter): Hyperparameter to specify.
    """
    assert not hyperp.has_value_assigned()

    if isinstance(hyperp, hp.Discrete):
        v = hyperp.vs[np.random.randint(len(hyperp.vs))]
        hyperp.assign_value(v)
    else:
        raise ValueError
    return v


def random_specify(output_lst):
    """Chooses random values to all the unspecified hyperparameters.

    The hyperparameters will be specified after this call, meaning that the
    compile and forward functionalities will be available for being called.

    Args:
        output_lst (list[deep_architect.core.Output]): List of output which by being
            traversed back will reach all the modules in the search space, and
            correspondingly all the current unspecified hyperparameters of the
            search space.
    """
    hyperp_value_lst = []
    for h in co.unassigned_independent_hyperparameter_iterator(output_lst):
        v = random_specify_hyperparameter(h)
        hyperp_value_lst.append(v)
    return hyperp_value_lst


def specify(output_lst, hyperp_value_lst):
    """Specify the parameters in the search space using the sequence of values
    passed as argument.

    .. note::
        This functionality is useful to replay the sequence of steps that were
        used to sample a model from the search space. This is typically used if
        it is necessary to replicate a model that was saved to disk by the
        logging functionality. Using the same sequence of values will yield the
        same model as long as the sequence of values takes the search space
        from fully unspecified to fully specified. Be careful otherwise.

    Args:
        output_lst (list[deep_architect.core.Output]): List of output which by being
            traversed back will reach all the modules in the search space, and
            correspondingly all the current unspecified hyperparameters of the
            search space.
        hyperp_value_lst (list[object]): List of values used to specify the hyperparameters.
    """
    for i, h in enumerate(
            co.unassigned_independent_hyperparameter_iterator(output_lst)):
        h.assign_value(hyperp_value_lst[i])
