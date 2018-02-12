from six import iteritems
from six.moves import xrange
import numpy as np
import darch.hyperparameters as hp
import darch.core as co

# TODO: perhaps change to not have to work until everything is specified.
# TODO: also perhaps change this for reproducibility.
# seems to be problematic.
def unset_hyperparameter_iterator(module_lst, h_lst=None):

    if h_lst is not None:
        for h in h_lst:
            if not h.is_set():
                yield h

    while not co.is_specified(module_lst):
        hs = co.get_unset_hyperparameters(module_lst)
        for h in hs:
            if not h.is_set():
                yield h
            
def random_specify_hyperparameter(h):
    assert not h.is_set()

    if isinstance(h, hp.Discrete):
        v = h.vs[np.random.randint(len(h.vs))]
        h.set_val(v)
    else:
        raise ValueError

    return v
    
def random_specify(output_or_module_lst, h_lst=None):
    vs = []
    module_lst = co.extract_unique_modules(output_or_module_lst)
    for h in unset_hyperparameter_iterator(module_lst, h_lst):
        # print h.get_name()
        v = random_specify_hyperparameter(h)
        vs.append(v)
    return vs

def specify(output_or_module_lst, h_lst, vs):
    module_lst = co.extract_unique_modules(output_or_module_lst)
    for i, h in enumerate( unset_hyperparameter_iterator(module_lst, h_lst) ):
        # print h.get_name(), h.vs
        # print h.get_name()
        # print h.vs, vs[i]
        h.set_val( vs[i] )

# NOTE: I may have to do some adaptation, but this is kind of correct.
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
        # import darch.utils as ut 
        # ut.draw_graph( outputs.values(), True, True)
        # from pprint import pprint
        # pprint( extract_features(inputs, outputs, hs) )
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

        module_lst = co.extract_unique_modules(outputs.values())
        h_it = unset_hyperparameter_iterator(module_lst, hs.values())
        tree_hist, tree_vs = self._tree_walk(h_it)
        rollout_hist, rollout_vs = self._rollout_walk(h_it)
        vs = tree_vs + rollout_vs
        # print "sample", tree_hist, rollout_hist
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

# NOTE: may be based on features or not.
class SurrogateModel:
    def __init__(self):
        pass
    
    def eval(self, feats):
        raise NotImplementedError
    
    def update(self, val, feats):
        raise NotImplementedError

class DummyModel(SurrogateModel):
    def eval(self, feats):
        return 0.0

    def update(self, val, feats):
        pass

# TODO: perhaps add some regularization.
# TODO: perhaps choose different ways of coming up with an objective. 
# it can be regression, it can be classification, it can be something else.
# TODO: what to do if the model is still not there.
# TODO: SMBO has to call update on the surrogate model.
# TODO: add the regularization type. this is the problem, the question 
# is that I can do many things.
# TODO: can be binary or additive.
# TODO: can be classification or ranking.
import sklearn.linear_model as lm
import scipy.sparse as sp

class HashingSurrogate(SurrogateModel):
    def __init__(self, hash_size, refit_interval, weight_decay_coeff=1e-5):
        self.hash_size = hash_size
        self.refit_interval = refit_interval
        self.vecs = []
        self.vals = []
        # NOTE: I'm going to use something like scikit learn for now.
        self.weight_decay_coeff = weight_decay_coeff
        self.model = None
        # NOTE: regression, classification, ranking.

    def eval(self, feats):
        if self.model == None:
            return 0.0
        else:
            vec = self._feats2vec(feats)
            return self.model.predict(vec)[0]

    def update(self, val, feats):
        vec = self._feats2vec(feats)        
        self.vecs.append(vec)
        self.vals.append(val)

        if len(self.vals) % self.refit_interval == 0:
            self._refit()

    def _feats2vec(self, feats):
        vec = sp.dok_matrix((1, self.hash_size), dtype='float') 
        for fs in feats:
            for f in fs:
                idx = hash(f) % self.hash_size
                vec[0, idx] += 1.0
        
        return vec.tocsr()
        
    def _refit(self):
        if self.model == None:
            self.model = lm.Ridge(self.weight_decay_coeff)
        
        X = sp.vstack(self.vecs, format='csr')
        y = np.array(self.vals)
        self.model.fit(X, y)

# TODO: implement this for a single one.
# TODO: thsi is going to be a model that either predicts 
# NOTE: potentially, this is going to be different from the things that I'm
# doing right now.
# TODO: in progress.
# TODO: I commented this out because tensorflow and pytorch dont work simultaneously https://github.com/tensorflow/tensorflow/issues/14812
# PLUS, it doesnt make sense to be here
# class CharLSTMSurrogate(SurrogateModel):
#     def __init__(self):
#         self.ch_lst = [chr(i) for i in range(ord('A'), ord('Z')) + 1] + [
#             chr(i) for i in range(ord('a'), ord('z') + 1)] + [
#             chr(i) for i in range(ord('0'), ord('9') + 1)] + [
#                 '.', ':', '-', '_', '<', '>', '/', '=', '*']
#         self.ch2idx = {ch : idx for (idx, ch) in enumerate(self.ch_lst)}
#
#         # maybe process them right away. may it easy to process.
#         # proc_feats =
#
#
#         self.hash_size = hash_size
#         self.refit_interval = refit_interval
#         self.feats_lst = []
#         self.vals = []
#         self.weight_decay_coeff = 1e-5
#         self.model = None
#
#         self.hidden_size = 100
#         self.embedding_size = 100
#
#         self.rnn = torch.nn.LSTM(self.embedding_size, self.hidden_size, 1)
#         self.ch_embs = nn.Embedding(len(self.ch_lst), self.embedding_size)
#
#     def eval(self, feats):
#         # TODO: check features here.
#
#
#         if self.model == None:
#             return 0.0
#         else:
#             feats = extract_features(inputs, outputs, hs)
#             vec = self._feats2vec(feats)
#             return self.model.predict(vec)[0]
#
#     def update(self, val, feats):
#         feats = extract_features(inputs, outputs, hs)
#
#
#
#         vec = self._feats2vec(feats)
#         self.feats_lst.append(vec)
#         self.vals.append(val)
#
#         if len(self.vals) % self.refit_interval == 0:
#             self._refit()
#
#     # TODO: it is not clear what is the best way of doing this.
#     def _refit(self):
#         if self.model == None:
#             self.model = lm.Ridge(self.weight_decay_coeff)
#
#         X = sp.vstack(self.feats_lst, format='csr')
#         y = np.array(self.vals)
#         self.model.fit(X, y)


# extract some simple features from the network. useful for smbo surrogate models.
# TODO: this may include features of the composite modules.
# NOTE: probably assume that all modules have the same scope.
def extract_features(inputs, outputs, hs):
    module_memo = co.OrderedSet()

    module_feats = []
    connection_feats = []
    module_hps_feats = []
    other_hps_feats = []

    # getting all the modules
    module_lst = co.extract_unique_modules(outputs.values())
    co.backward_traverse(module_lst, lambda _: None, module_memo)

    for m in module_memo:
        # module features
        m_feats = m.get_name()
        module_feats.append( m_feats ) 

        for ox_localname, ox in iteritems(m.outputs):
            if ox.is_connected():
                ix_lst = ox.get_connected_inputs()
                for ix in ix_lst:
                    # connection features
                    c_feats = "%s |-> %s" % (ox.get_name(), ix.get_name())
                    connection_feats.append( c_feats )
        
        # module hyperparameters
        for h_localname, h in iteritems(m.hs):
            mh_feats = "%s/%s : %s = %s" % (
                m.get_name(), h_localname, h.get_name(), h.val)
            module_hps_feats.append( mh_feats )

    # other features
    for h_localname, h in iteritems(hs):
        oh_feats = "%s : %s = %s" % (h_localname, h.get_name(), h.val)
        other_hps_feats.append( oh_feats ) 
    
    return (module_feats, connection_feats, module_hps_feats, other_hps_feats)

# either samples some models and picks one greedily according to the 
# surrogate model or samples a model randomly. 
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
                feats = extract_features(inputs, outputs, hs)
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
        feats = extract_features(inputs, outputs, hs)
        self.surr_model.update(val, feats)

# surrogate with MCTS optimization.
# TODO: make sure tht can keep the tree while the surrogate changes behind me.
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
                # print vs
                # print
                feats = extract_features(inputs, outputs, hs)
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
        # print(cfg_d['vs'], len(cfg_d['vs']))
        (inputs, outputs, hs) = self.search_space_fn()
        specify(outputs.values(), hs.values(), cfg_d['vs'])
        feats = extract_features(inputs, outputs, hs)
        self.surr_model.update(val, feats)
        
        self.cnt += 1
        if self.cnt % self.tree_refit_interval == 0:
            self.mcts = MCTSearcher(self.search_space_fn)

def run_searcher(searcher, evaluator, num_samples):
    vs_lst = []
    score_lst = []
    cfg_d_lst = []

    for _ in xrange(num_samples):
        inputs, outputs, hs, vs, cfg_d = searcher.sample()
        score = evaluator.eval(inputs, outputs, hs)
        searcher.update(score, cfg_d)

        vs_lst.append(vs)
        score_lst.append(score)
        cfg_d_lst.append(cfg_d)
    
    return vs_lst, score_lst, cfg_d_lst

# TODO: how to handle the case where the surrogate model was to use 
# information that is only available after compilation.
# TODO: write a few use cases for the model.
# TODO: assess how much memory is kept around by using this model.
# TODO: perhaps give the probability of exploration to the sample function in 
# SMBO. same for the number of samples.
# TODO: make sure that it is easy to run some of these models.
# TODO: add some feature functions for the surrogate model.
# TODO: how should searchers keep track of the out standing models.
# TODO: move the features to some other file.
# TODO: how to keep the state of a searcher? how to do a model of architectures.
# TODO: do a directory based distributed architecture. the updates to the 
# surrogate model are done on master. can we do it for distributed cases.
# TODO: do the serialization of these models. 
# TODO: finish the model to do all intermediate aspects.
# TODO: evaluator can be based on taking model and solving some task.
# TODO: it must be easy to start one of these searchers from scratch.
# TODO: add more searchers and more complex searchers.
# TODO: perhaps add dimensions to the model.
# TODO: it is important to come up with a good policy for the update of the 
# surrogate model in the case where it is complicated.
# TODO: I think that this can be in many cases a single file model.
# TODO: how to do start and begin on search spaces, especially on these kinds 
# of models.
# TODO: way of using hyperparameters in the definition of the data 
# augmentation in the dataset.
# TODO: the hashing functions may require a faster alternative.
# TODO: it is possible to implement these models in a different way.
# TODO: perhaps have an efficient way of iterating over the hyperparameters
# that can be used by the different models. 
# TODO: how to write or in hyperparameters. currently not supported.
# for example, currently it is not supported. that is not true. 
# no need for that. can have those attached to some dummy module.
# TODO: do something like split merge and add it to the search space.
# TODO: ability to do contextual parameters.
# TODO: add a way to keep information about the model.
# TODO: check if I can do things more efficiently when evaluationg the evaluation 
# model.
# TODO: keep the history in the case I use multiple GPUs.
# TODO: need to add a rule for the exploration probability.
# TODO: add the binarization of the model. 

# TODO: write tests for MCTS (behavior, convergence)
# NOTE: the surrogate function can do something even if there are no values
# there yet.
# TODO: add some online adaptation to the exploration probability.

# TODO: explore MCTS by plotting some things.
# TODO: it needs a better rollout policy. this is true for MCTS. 
# random is pretty bad.
# possibility of doing MCTS 

# TODO: define better features.

# TODO: I don't know if by moving MCTS to this case, I'm going to have to 
# change things, as there might exist unexpanded nodes.

# TODO: add bissection to MCTS. if it was the maximum, that is a good 
# idea.