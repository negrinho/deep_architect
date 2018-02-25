import darch.core as co
import sklearn.linear_model as lm
import scipy.sparse as sp
import numpy as np
from six import iteritems

class SurrogateModel:
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
    module_memo = []
    def fn(m):
        module_memo.append(m)
    co.backward_traverse(outputs.values(), fn)

    for m in module_memo:
        # module features
        m_feats = m.get_name()
        module_feats.append(m_feats) 

        for ox_localname, ox in iteritems(m.outputs):
            if ox.is_connected():
                ix_lst = ox.get_connected_inputs()
                for ix in ix_lst:
                    # connection features
                    c_feats = "%s |-> %s" % (ox.get_name(), ix.get_name())
                    connection_feats.append( c_feats )
        
        # module hyperparameters
        for h_localname, h in iteritems(m.hyperps):
            mh_feats = "%s/%s : %s = %s" % (
                m.get_name(), h_localname, h.get_name(), h.val)
            module_hps_feats.append( mh_feats )

    # other features
    for h_localname, h in iteritems(hs):
        oh_feats = "%s : %s = %s" % (h_localname, h.get_name(), h.val)
        other_hps_feats.append( oh_feats ) 
    
    return (module_feats, connection_feats, module_hps_feats, other_hps_feats)


