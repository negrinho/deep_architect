import deep_architect.core as co
import deep_architect.hyperparameters as hp
from deep_architect.searchers.common import Searcher
import numpy as np


# keeps the statistics and knows how to update information related to a node.
class MCTSTreeNode:
    """Encapsulates the information contained in a single node of the MCTS tree.

    See also :class:`deep_architect.searchers.MCTSSearcher`.
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
                score = (node.sum_scores / node.num_trials + exploration_bonus
                         * np.sqrt(2.0 * parent_log_nt / node.num_trials))
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


class MCTSSearcher(Searcher):
    def __init__(self, search_space_fn, exploration_bonus=1.0):
        Searcher.__init__(self, search_space_fn)
        self.exploration_bonus = exploration_bonus
        self.mcts_root_node = MCTSTreeNode(None)

    # NOTE: this operation changes the state of the tree.
    def sample(self):
        inputs, outputs, hyperps = self.search_space_fn()

        h_it = co.unassigned_independent_hyperparameter_iterator(
            outputs.values(), hyperps.values())
        tree_hist, tree_vs = self._tree_walk(h_it)
        rollout_hist, rollout_vs = self._rollout_walk(h_it)
        vs = tree_vs + rollout_vs
        searcher_eval_token = {
            'tree_hist': tree_hist,
            'rollout_hist': rollout_hist
        }

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
                h.assign_value(v)

                hist.append(i)
                vs.append(v)
            else:
                # NOTE: only implemented for discrete hyperparameters.
                # does the expansion after tree walk.
                if isinstance(h, hp.Discrete):
                    node.expand(len(h.vs))

                    i = np.random.randint(0, len(h.vs))
                    v = h.vs[i]
                    h.assign_value(v)

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
                h.assign_value(v)

                hist.append(i)
                vs.append(v)
            else:
                raise ValueError
        return hist, vs