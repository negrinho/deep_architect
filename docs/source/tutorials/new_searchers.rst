
Implementing new searchers
--------------------------

Searchers are used to determine the policy that is used to search a search space.
The search space merely encodes the set of architectures that are to be
considered, allowing the encoding an expert's inductive bias in a very direct
way.
We consider a very simple API for a searcher, composed of two main methods and
two auxiliary methods.

.. code:: python

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


Implementing a new searcher is done by inheriting from this class.
A searcher is initialized with a function that returns a search space,
which is simply a dictionary of inputs and dictionary of outputs, where some
of the hyperparameters of the modules in the search space do not have a value
assigned (if all hyperparameters that are reachable have a value assigned,
then there is nothing for the searcher to do).
The sample method is used to pick an architecture from this search space.
The architecture returned results simply from assigning a value to all the
hyperparameters in the search space, until there are no more hyperparameters
left to assign a value to.
Sample returns the inputs and outputs, which encode the architecture.
The other two returned terms deserve a bit more explanation.
The third element returned by sample is the list of hyperparameter values that
were used to arrive at the particular architecture that is being returned.
If we call the search space function, iterate over the independent hyperparameters
that do not have a value assigned yet, and assign the values from this list in
order to them, we will obtain back the same architecture.
The combination of the search space function and this list of hyperparameters
effectively encodes the structure of the architecture sampled as it can be
used to recover the sampled architecture.
The iteration through the hyperparameters is guaranteed to be deterministic, i.e.,
each time that the search space function is called and we iterate over the hyperparameters,
we will traverse the hyperparameters in the same order (provided that the same
hyperparameters are present; note that the specific values that we assign to
hyperparameters may influence the hyperparameters (number and type) of hyperparameters
that we have to assign a value to).
The fourth element that is returned by the sample function is what we call a
searcher evaluation token.
This token is used to keep any information that is necessary for the
searcher to update its state once the result of evaluating the sample
architecture is passed back to the searcher in the call to update.
Having this token is especially useful for multiple worker type settings where
we have multiple workers being serviced by a searcher.
In this case, the results of evaluating the architectures may arrive in a
different order than they were relayed to the workers.
Having the searcher evaluation token allows the searcher to update its state
regardless of this, guaranteeing that the incoming results are used correctly.

Finally, we look at the update function of the searcher.
The update function takes the results of the evaluation and the searcher
evalution token (which allows the searcher to identify which architecture
the results refer to) and updates the state of the searcher with this new
information. The searcher is a stateful object; updates to the searcher
change the state of the searcher and therefore, the behavior of the searcher
may change as a result.

The other two auxiliary functions that we have for the searcher are save_state
and load_state, which allows us to save the state of the searcher to disk
(e.g., for checkpointing) and load it back at a later stage.
This is especially useful for long running searches that require resuming
from saved state multiple times due to limits in job length on a server or
potential hardware issues.

We will now go over a two different searchers for the reader to ground the
ideas that we have discussed here.
The simplest possible searcher is a random searcher, which assigns a random
value to each of the unassigned hyperparameters.

.. code:: python

    from deep_architect.searchers.common import random_specify, Searcher


    class RandomSearcher(Searcher):

        def __init__(self, search_space_fn):
            Searcher.__init__(self, search_space_fn)

        def sample(self):
            inputs, outputs = self.search_space_fn()
            vs = random_specify(outputs.values())
            return inputs, outputs, vs, {}

        def update(self, val, searcher_eval_token):
            pass


The implementation of this searcher is very short. It uses the implementation
of random_specify, which is also fairly compact. We copy it here for reference.


.. code:: python

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

These are the two main auxiliary functions to randomly specify hyperparameters
and to pick a random architecture from the search space by picking values
for all the hyperparameters independently at random.
As we can see, this functionality is concise and self-explanatory.

Let us now see a SMBO searcher, which is more complex than the searcher than
the random searcher that we looked at right now.
We copy the implementation here for ease of reference.

.. code:: python

    from deep_architect.searchers.common import random_specify, specify, Searcher
    from deep_architect.surrogates.common import extract_features
    import numpy as np


    class SMBOSearcher(Searcher):

        def __init__(self, search_space_fn, surrogate_model, num_samples, eps_prob):
            Searcher.__init__(self, search_space_fn)
            self.surr_model = surrogate_model
            self.num_samples = num_samples
            self.eps_prob = eps_prob

        def sample(self):
            if np.random.rand() < self.eps_prob:
                inputs, outputs = self.search_space_fn()
                best_vs = random_specify(outputs.values())
            else:
                best_model = None
                best_vs = None
                best_score = -np.inf
                for _ in range(self.num_samples):
                    inputs, outputs = self.search_space_fn()
                    vs = random_specify(outputs.values())

                    feats = extract_features(inputs, outputs)
                    score = self.surr_model.eval(feats)
                    if score > best_score:
                        best_model = (inputs, outputs)
                        best_vs = vs
                        best_score = score

                inputs, outputs = best_model

            searcher_eval_token = {'vs': best_vs}
            return inputs, outputs, best_vs, searcher_eval_token

        def update(self, val, searcher_eval_token):
            (inputs, outputs) = self.search_space_fn()
            specify(outputs.values(), searcher_eval_token['vs'])
            feats = extract_features(inputs, outputs)
            self.surr_model.update(val, feats)

This searcher can be found in the searchers/smbo_random.py.
A SMBO (surrogate model based optimization) searcher relies on a surrogate
function on the space of architectures that can be evaluated for each architecture
of the space to give us an estimate of the performance of that architecture
(or at least a score that should preserve the ordering of the architectures, i.e.,
more performance architectures should ideally be scored higher than less performant
ones).

Sampling an architecture from the search space is done as a result
of optimizing the surrogation function. In the implementation above, the
optimization of the surrogate function is done by sampling a number of
random architectures from the search space, evaluating the surrogate function,
and picking the best one. We also just pick an architecture at random from the
search space with fixed probabability.

Updating the searcher in this case corresponds to updating the surrogate function
with the observed results for the architecture in question. In this case,
changes to the searcher policy occur as a result of updates to the surrogate
function as it hopefully becomes more accurate as we get more data for the
search space.
The API definition for a surrogate function can be found in surrogates/common.py.

Implementing a new searcher amounts to implementing the sample and update
methods for it. We see that these are fairly simple methods. One of the
advantages of this API definition for the searcher is that all the state of the
searcher is kept locally in the searcher object.

We point the reader to searchers folder for more example implementations of
searcheres. There is a single searcher per file. We very much welcome searcher
contributions, so if you would like to contribute with a search algorithm that
you developed for DeepArchitect, please write a issue to discuss the implementation.
One of the goals of DeepArchitect is to make architecture search research widely
available and reusable.
