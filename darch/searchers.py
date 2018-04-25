import random
from collections import deque
import numpy as np
import darch.hyperparameters as hp
import darch.surrogates as su
from darch.search_logging import join_paths, write_jsonfile
import darch.core as co

# TODO: perhaps change to not have to work until everything is specified.
# this can be done through a flag.
def unset_hyperparameter_iterator(output_lst, hyperp_lst=None):
    """Returns an iterator over the hyperparameters that are not specified in
    the current search space.

    This iterator is used by the searchers to go over the unspecified
    hyperparameters.

    .. note::
        It is assumed that all the hyperparameters that are touched by the
        iterator will be specified (most likely, right away). Otherwise, the
        iterator will never terminate.

    Args:
        output_lst (list[darch.core.Output]): List of output which by being
            traversed back will reach all the modules in the search space, and
            correspondingly all the current unspecified hyperparameters of the
            search space.
        hyperp_lst (list[darch.core.Hyperparameter], optional): List of
            additional hyperparameter that are not involved in the search space.
            Often used to specif additional hyperparameters, e.g., learning
            rate.

    Yields:
        (darch.core.Hyperparameter):
            Next unspecified hyperparameter of the search space.
    """
    if hyperp_lst is not None:
        for h in hyperp_lst:
            if not h.is_set():
                yield h

    while not co.is_specified(output_lst):
        hs = co.get_unset_hyperparameters(output_lst)
        for h in hs:
            if not h.is_set():
                yield h

# TODO: generalize this for other types of hyperparameters. currently only supports
# discrete hyperparameters.
def random_specify_hyperparameter(hyperp):
    """Choose a random value for an unspecified hyperparameter.

    The hyperparameter becomes specified after the call.

    hyperp (darch.core.Hyperparameter): Hyperparameter to specify.
    """
    assert not hyperp.is_set()

    if isinstance(hyperp, hp.Discrete):
        v = hyperp.vs[np.random.randint(len(hyperp.vs))]
        hyperp.set_val(v)
    else:
        raise ValueError
    return v

def random_specify(output_lst, hyperp_lst=None):
    """Chooses random values to all the unspecified hyperparameters.

    The hyperparameters will be specified after this call, meaning that the
    compile and forward functionalities will be available for being called.

    Args:
        output_lst (list[darch.core.Output]): List of output which by being
            traversed back will reach all the modules in the search space, and
            correspondingly all the current unspecified hyperparameters of the
            search space.
        hyperp_lst (list[darch.core.Hyperparameter], optional): List of
            additional hyperparameters that are not involved in the search space.
            Often used to specify additional hyperparameters, e.g., learning
            rate.
    """
    vs = []
    for h in unset_hyperparameter_iterator(output_lst, hyperp_lst):
        v = random_specify_hyperparameter(h)
        vs.append(v)
    return vs

def specify(output_lst, hyperp_lst, vs):
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
        output_lst (list[darch.core.Output]): List of output which by being
            traversed back will reach all the modules in the search space, and
            correspondingly all the current unspecified hyperparameters of the
            search space.
        hyperp_lst (list[darch.core.Hyperparameter], optional): List of
            additional hyperparameters that are not involved in the search space.
            Often used to specify additional hyperparameters, e.g., learning
            rate.
        vs (list[object]): List of values used to specify the hyperparameters.
    """
    for i, h in enumerate(unset_hyperparameter_iterator(output_lst, hyperp_lst)):
        h.set_val(vs[i])

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
        search_space_fn (() -> (dict[str,darch.core.Input], dict[str,darch.core.Output], dict[str,darch.core.Hyperparameter])):
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
            (dict[str, darch.core.Input], dict[str, darch.core.Output], dict[str, darch.core.Hyperparameter], list[object], dict[str, object]):
                Tuple encoding the model sampled from the search space.
                The positional arguments have the following semantics:
                1: Dictionary of names to inputs of the model.
                2: Dictionary of names to outputs of the model.
                3: Dictionary of names to hyperparameters (typically extra, i.e.,
                not involved in the structural search space).
                4: List with list of values that can be to replay the sequence
                of values assigned to the hyperparameters, and therefore,
                reproduce, given the search space, the model sampled.
                5: Searcher evaluation token that is sufficient for the searcher
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

### Random
class RandomSearcher(Searcher):
    def sample(self):
        inputs, outputs, hs = self.search_space_fn()
        vs = random_specify(outputs.values(), hs.values())
        return inputs, outputs, hs, vs, {}

    def update(self, val, searcher_eval_token):
        pass

### MCTS
# keeps the statistics and knows how to update information related to a node.
class MCTSTreeNode:
    """Encapsulates the information contained in a single node of the MCTS tree.

    See also :class:`darch.searchers.MCTSearcher`.
    """
    def __init__(self, parent_node):
        self.num_trials = 0
        self.sum_scores = 0.0

        self.parent = parent_node
        self.children = None

    def is_leaf(self):
        return self.children == None

    def update_stats(self, score):
        self.sum_scores += score
        self.num_trials += 1

    # returns the child with the highest UCT score.
    def best_child(self, exploration_bonus):
        assert not self.is_leaf()

        # if two nodes have the same score.
        best_inds = None
        best_score = -np.inf

        best_child = best_i = None

        parent_log_nt = np.log(self.num_trials)
        for (i, node) in enumerate(self.children):
            # NOTE: potentially, do a different definition for the scores.
            # especially once the surrogate model is introduced.
            # selection policy may be somewhat biased towards what the
            # rollout policy based on surrogate functions says.
            # think about how to extend this.
            if node.num_trials > 0:
                score = (node.sum_scores / node.num_trials +
                         exploration_bonus * np.sqrt(
                             2.0 * parent_log_nt / node.num_trials))
            else:
                score = np.inf

            # keep the best node.
            if score > best_score:
                best_inds = [i]
                best_score = score
            elif score == best_score:
                best_inds.append(i)

            # draw a child at random and expand.
            best_i = np.random.choice(best_inds)
            best_child = self.children[best_i]

        assert best_child is not None and best_i is not None
        return (best_child, best_i)

    # expands a node creating all the placeholders for the children.
    def expand(self, num_children):
        self.children = [MCTSTreeNode(self) for _ in range(num_children)]

class MCTSearcher(Searcher):
    def __init__(self, search_space_fn, exploration_bonus=1.0):
        Searcher.__init__(self, search_space_fn)
        self.exploration_bonus = exploration_bonus
        self.mcts_root_node = MCTSTreeNode(None)

    # NOTE: this operation changes the state of the tree.
    def sample(self):
        inputs, outputs, hyperps = self.search_space_fn()

        h_it = unset_hyperparameter_iterator(outputs.values(), hyperps.values())
        tree_hist, tree_vs = self._tree_walk(h_it)
        rollout_hist, rollout_vs = self._rollout_walk(h_it)
        vs = tree_vs + rollout_vs
        searcher_eval_token = {'tree_hist' : tree_hist, 'rollout_hist' : rollout_hist}

        return inputs, outputs, hyperps, vs, searcher_eval_token

    def update(self, val, searcher_eval_token):
        node = self.mcts_root_node
        node.update_stats(val)

        for i in searcher_eval_token['tree_hist']:
            node = node.children[i]
            node.update_stats(val)

    def _tree_walk(self, h_it):
        hist = []
        vs = []

        node = self.mcts_root_node
        for h in h_it:
            if not node.is_leaf():
                node, i = node.best_child(self.exploration_bonus)
                v = h.vs[i]
                h.set_val(v)

                hist.append(i)
                vs.append(v)
            else:
                # NOTE: only implemented for discrete hyperparameters.
                # does the expansion after tree walk.
                if isinstance(h, hp.Discrete):
                    node.expand(len(h.vs))

                    i = np.random.randint(0, len(h.vs))
                    v = h.vs[i]
                    h.set_val(v)

                    hist.append(i)
                    vs.append(v)
                else:
                    raise ValueError
                break
        return hist, vs

    def _rollout_walk(self, h_it):
        hist = []
        vs = []

        for h in h_it:
            if isinstance(h, hp.Discrete):
                i = np.random.randint(0, len(h.vs))
                v = h.vs[i]
                h.set_val(v)

                hist.append(i)
                vs.append(v)
            else:
                raise ValueError
        return hist, vs

### SMBO
class SMBOSearcher(Searcher):
    def __init__(self, search_space_fn, surrogate_model, num_samples, eps_prob):
        Searcher.__init__(self, search_space_fn)
        self.surr_model = surrogate_model
        self.num_samples = num_samples
        self.eps_prob = eps_prob

    def sample(self):
        if np.random.rand() < self.eps_prob:
            inputs, outputs, hyperps = self.search_space_fn()
            best_vs = random_specify(outputs.values(), hyperps.values())
        else:
            best_model = None
            best_vs = None
            best_score = - np.inf
            for _ in range(self.num_samples):
                inputs, outputs, hyperps = self.search_space_fn()
                vs = random_specify(outputs.values(), hyperps.values())

                feats = su.extract_features(inputs, outputs, hyperps)
                score = self.surr_model.eval(feats)
                if score > best_score:
                    best_model = (inputs, outputs, hyperps)
                    best_vs = vs
                    best_score = score

            inputs, outputs, hyperps = best_model

        searcher_eval_token = {'vs' : best_vs}
        return inputs, outputs, hyperps, best_vs, searcher_eval_token

    def update(self, val, searcher_eval_token):
        (inputs, outputs, hyperps) = self.search_space_fn()
        specify(outputs.values(), hyperps.values(), searcher_eval_token['vs'])
        feats = su.extract_features(inputs, outputs, hyperps)
        self.surr_model.update(val, feats)

# surrogate with MCTS optimization.
# TODO: make sure that can keep the tree while the surrogate changes behind me.
# TODO: I would just compute the std for the scores.
class SMBOSearcherWithMCTSOptimizer(Searcher):
    def __init__(self, search_space_fn, surrogate_model, num_samples,
        eps_prob, tree_refit_interval):
        Searcher.__init__(self, search_space_fn)
        self.surr_model = surrogate_model
        self.mcts = MCTSearcher(self.search_space_fn)
        self.num_samples = num_samples
        self.eps_prob = eps_prob
        self.tree_refit_interval = tree_refit_interval
        self.cnt = 0

    def sample(self):
        if np.random.rand() < self.eps_prob:
            inputs, outputs, hyperps = self.search_space_fn()
            best_vs = random_specify(outputs.values(), hyperps.values())
        # TODO: ignoring the size of the model here.
        # TODO: needs to add the exploration bonus.
        else:
            best_model = None
            best_vs = None
            best_score = - np.inf
            for _ in range(self.num_samples):
                (inputs, outputs, hyperps, vs, m_cfg_d) = self.mcts.sample()
                feats = su.extract_features(inputs, outputs, hyperps)
                score = self.surr_model.eval(feats)
                if score > best_score:
                    best_model = (inputs, outputs, hyperps)
                    best_vs = vs
                    best_score = score

                self.mcts.update(score, m_cfg_d)
            inputs, outputs, hyperps = best_model

        searcher_eval_token = {'vs' : best_vs}
        return inputs, outputs, hyperps, best_vs, searcher_eval_token

    def update(self, val, searcher_eval_token):
        (inputs, outputs, hyperps) = self.search_space_fn()
        specify(outputs.values(), hyperps.values(), searcher_eval_token['vs'])
        feats = su.extract_features(inputs, outputs, hyperps)
        self.surr_model.update(val, feats)

        self.cnt += 1
        if self.cnt % self.tree_refit_interval == 0:
            self.mcts = MCTSearcher(self.search_space_fn)

### EVOLUTION SEARCHER
def mutatable(h):
    return h.get_name().startswith('H.Mutatable')

def mutate(output_lst, vs, mutatable_fn, search_space_fn):
    mutate_candidates = []
    new_vs = []
    for i, h in enumerate(unset_hyperparameter_iterator(output_lst)):
        if mutatable_fn(h):
            mutate_candidates.append((i, h))
        h.set_val(vs[i])
        new_vs.append(vs[i])
    # mutate a random hyperparameter
    m_ind, m_h = mutate_candidates[random.randint(
        0, len(mutate_candidates) - 1)]
    v = m_h.vs[random.randint(0, len(m_h.vs) - 1)]

    # ensure that same value is not chosen again
    while v == vs[m_ind]:
        v = m_h.vs[random.randint(0, len(m_h.vs) - 1)]

    inputs, outputs, hs = search_space_fn()
    output_lst = outputs.values()
    for i, h in enumerate(unset_hyperparameter_iterator(output_lst)):
        h.set_val(new_vs[i])
    return inputs, outputs, hs, new_vs

class EvolutionSearcher(Searcher):
    def __init__(self, search_space_fn, mutatable_fn, P, S, regularized=False):
        Searcher.__init__(self, search_space_fn)

        # Population size
        self.P = P
        # Sample size
        self.S = S

        self.population = deque(maxlen=P)
        self.regularized = regularized
        self.initializing = True
        self.mutatable = mutatable_fn

    def sample(self):
        if self.initializing:
            inputs, outputs, hs = self.search_space_fn()
            vs = random_specify(outputs.values(), hs.values())
            if len(self.population) >= self.P - 1:
                self.initializing = False
            return inputs, outputs, hs, vs, {'vs': vs}
        else:
            sample_inds = sorted(random.sample(
                range(len(self.population)), min(self.S, len(self.population))))
            # delete weakest model
            weak_ind = self.get_weakest_model_index(sample_inds)

            # mutate strongest model
            inputs, outputs, hs = self.search_space_fn()
            vs, _ = self.population[self.get_strongest_model_index(
                sample_inds)]
            inputs, outputs, hs, new_vs = mutate(outputs.values(), vs, self.mutatable, self.search_space_fn)

            del self.population[weak_ind]
            return inputs, outputs, hs, new_vs, {'vs': new_vs}

    def get_searcher_state_token(self):
        return {
            "P": self.P,
            "S": self.S,
            "population": list(self.population),
            "regularized": self.regularized,
            "initializing": self.initializing,
        }

    def save_state(self, folder_name):
        state = self.get_searcher_state_token()
        write_jsonfile(state, join_paths([folder_name, 'searcher_state.json']))

    def load(self, state):
        self.P = state["P"]
        self.S = state["S"]
        self.regularized = state['regularized']
        self.population = deque(state['population'])
        self.initializing = state['initializing']

    def update(self, val, cfg_d):
        self.population.append((cfg_d['vs'], val))

    def get_weakest_model_index(self, sample_inds):
        if self.regularized:
            return sample_inds[0]
        else:
            min_acc = 1.
            min_acc_ind = -1
            for i in range(len(sample_inds)):
                _, acc = self.population[sample_inds[i]]
                if acc < min_acc:
                    min_acc = acc
                    min_acc_ind = i
            return sample_inds[min_acc_ind]

    def get_strongest_model_index(self, sample_inds):
        max_acc = 0.
        max_acc_ind = -1
        for i in range(len(sample_inds)):
            _, acc = self.population[sample_inds[i]]
            if acc > max_acc:
                max_acc = acc
                max_acc_ind = i
        return sample_inds[max_acc_ind]