from six import iteritems
from six.moves import xrange
import numpy as np
import darch.hyperparameters as hp
import darch.core as co
import darch.surrogates as su

# TODO: perhaps change to not have to work until everything is specified.
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
        # print h.get_name()
        v = random_specify_hyperparameter(h)
        vs.append(v)
    return vs

def specify(output_lst, hyperp_lst, vs):
    for i, h in enumerate(unset_hyperparameter_iterator(output_lst, hyperp_lst)):
        h.set_val(vs[i])

class Searcher:
    def sample(self):
        raise NotImplementedError
    
    def update(self, val, cfg_d):
        raise NotImplementedError

class RandomSearcher(Searcher):
    def __init__(self, search_space_fn):
        self.search_space_fn = search_space_fn
        
    def sample(self):
        inputs, outputs, hs = self.search_space_fn()
        vs = random_specify(outputs.values(), hs.values())
        return inputs, outputs, hs, vs, {}
    
    def update(self, val, cfg_d):
        pass

# keeps the statistics and knows how to update information related to a node.
class MCTSTreeNode:
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

        parent_log_nt = np.log(self.num_trials)
        for (i, node) in enumerate(self.children):
            # NOTE: potentially, do a different definition for the scores.
            # especially once the surrogate model is introduced.
            # selection policy may be somewhat biased towards what the 
            # rollout policy based on surrogate functions says.
            # think about how to extend this.
            if node.num_trials > 0:
                score = ( node.sum_scores / node.num_trials + 
                            exploration_bonus * np.sqrt(
                                2.0 * parent_log_nt / node.num_trials) )
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

        return (best_child, best_i)

    # expands a node creating all the placeholders for the children.
    def expand(self, num_children):
        self.children = [MCTSTreeNode(self) for _ in xrange(num_children)]

class MCTSearcher(Searcher):
    def __init__(self, search_space_fn, exploration_bonus=1.0):
        self.search_space_fn = search_space_fn
        self.exploration_bonus = exploration_bonus
        self.mcts_root_node = MCTSTreeNode(None)
        
    # NOTE: this operation changes the state of the tree.
    def sample(self):
        inputs, outputs, hs = self.search_space_fn()

        h_it = unset_hyperparameter_iterator(outputs.values(), hs.values())
        tree_hist, tree_vs = self._tree_walk(h_it)
        rollout_hist, rollout_vs = self._rollout_walk(h_it)
        vs = tree_vs + rollout_vs
        cfg_d = {'tree_hist' : tree_hist, 'rollout_hist' : rollout_hist}

        return inputs, outputs, hs, vs, cfg_d

    def update(self, val, cfg_d):
        node = self.mcts_root_node
        node.update_stats(val)

        for i in cfg_d['tree_hist']:
            node = node.children[i]
            node.update_stats(val)

    def _tree_walk(self, h_it):
        hist = []
        vs = []

        node = self.mcts_root_node
        for h in h_it:
            # print h.get_name(), h.vs
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
                    node.expand( len(h.vs) )

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
            # print h.get_name(), h.vs
            if isinstance(h, hp.Discrete):
                i = np.random.randint(0, len(h.vs))
                v = h.vs[i]
                h.set_val(v)
                
                hist.append(i)
                vs.append(v)
            else:
                raise ValueError
        return hist, vs

class SMBOSearcher(Searcher):
    def __init__(self, search_space_fn, surrogate_model, num_samples, eps_prob):
        self.search_space_fn = search_space_fn
        self.surr_model = surrogate_model
        self.num_samples = num_samples
        self.eps_prob = eps_prob
    
    def sample(self):
        if np.random.rand() < self.eps_prob:
            inputs, outputs, hs = self.search_space_fn()

            best_vs = random_specify(outputs.values(), hs.values())
        else: 
            best_model = None
            best_vs = None
            best_score = - np.inf
            for i in xrange(self.num_samples):
                inputs, outputs, hs = self.search_space_fn()
                vs = random_specify(outputs.values(), hs.values())

                # NOTE: the model may require a compile to do the surrogate 
                # this would be the case if it requires information that is 
                # available only after compilation.
                feats = su.extract_features(inputs, outputs, hs)
                score = self.surr_model.eval(feats)
                if score > best_score:
                    best_model = (inputs, outputs, hs)
                    best_vs = vs
                    best_score = score

            inputs, outputs, hs = best_model
        
        cfg_d = {'vs' : best_vs}
        return inputs, outputs, hs, best_vs, cfg_d    
        
    def update(self, val, cfg_d):
        (inputs, outputs, hs) = self.search_space_fn()
        specify(outputs.values(), hs.values(), cfg_d['vs'])
        feats = su.extract_features(inputs, outputs, hs)
        self.surr_model.update(val, feats)

# surrogate with MCTS optimization.
# TODO: make sure that can keep the tree while the surrogate changes behind me.
# TODO: maybe done with part of the model.
# TODO: I would just compute the std for the scores.
class SMBOSearcherWithMCTSOptimizer(Searcher):
    def __init__(self, search_space_fn, surrogate_model, num_samples, 
        eps_prob, tree_refit_interval):
        self.search_space_fn = search_space_fn
        self.surr_model = surrogate_model
        self.mcts = MCTSearcher(self.search_space_fn)
        self.num_samples = num_samples
        self.eps_prob = eps_prob
        self.tree_refit_interval = tree_refit_interval
        self.cnt = 0

    def sample(self):
        if np.random.rand() < self.eps_prob:
            inputs, outputs, hs = self.search_space_fn()

            best_vs = random_specify(outputs.values(), hs.values())
        # TODO: ignoring the size of the model here.
        # TODO: needs to add the exploration bonus.
        else:
            best_model = None
            best_vs = None
            best_score = - np.inf
            for _ in xrange(self.num_samples):
                (inputs, outputs, hs, vs, m_cfg_d) = self.mcts.sample()
                feats = su.extract_features(inputs, outputs, hs)
                score = self.surr_model.eval(feats)
                if score > best_score:
                    best_model = (inputs, outputs, hs)
                    best_vs = vs
                    best_score = score
                
                self.mcts.update(score, m_cfg_d)
            inputs, outputs, hs = best_model
        
        cfg_d = {'vs' : best_vs}
        return inputs, outputs, hs, best_vs, cfg_d 
        
    def update(self, val, cfg_d):
        (inputs, outputs, hs) = self.search_space_fn()
        specify(outputs.values(), hs.values(), cfg_d['vs'])
        feats = su.extract_features(inputs, outputs, hs)
        self.surr_model.update(val, feats)
        
        self.cnt += 1
        if self.cnt % self.tree_refit_interval == 0:
            self.mcts = MCTSearcher(self.search_space_fn)
