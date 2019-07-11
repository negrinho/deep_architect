import sklearn.linear_model as lm
import scipy.sparse as sp
import numpy as np
from six import iteritems, itervalues
from deep_architect.surrogates.common import SurrogateModel
import deep_architect.utils as ut
import os


class HashingSurrogate(SurrogateModel):
    """Simple hashing surrogate function that simply hashes the strings in the
    feature representation of the architecture to buckets. See
    :func:`deep_architect.surrogates.common.extract_features` for the functions that is used to
    extract string features for the architectures.
    """

    def __init__(self,
                 hash_size,
                 refit_interval,
                 weight_decay_coeff=1e-5,
                 use_module_feats=True,
                 use_connection_feats=True,
                 use_module_hyperp_feats=True):
        self.hash_size = hash_size
        self.refit_interval = refit_interval
        self.weight_decay_coeff = weight_decay_coeff
        self.feats_name_to_use_flag = {
            'connection_feats': use_connection_feats,
            'module_hyperp_feats': use_module_hyperp_feats,
            'module_feats': use_module_feats,
        }
        assert any(itervalues(self.feats_name_to_use_flag))
        self.vecs_lst = []
        self.vals_lst = []
        # NOTE: using scikit learn for now.
        self.model = None

    def eval(self, feats):
        if self.model == None:
            return 0.0
        else:
            vec = self._feats2vec(feats)
            return self.model.predict(vec)[0]

    def update(self, val, feats):
        vec = self._feats2vec(feats)
        self.vecs_lst.append(vec)
        self.vals_lst.append(val)
        if len(self.vals_lst) % self.refit_interval == 0:
            self._refit()

    def _feats2vec(self, feats):
        vec = sp.dok_matrix((1, self.hash_size), dtype='float')
        for name, fs in iteritems(feats):
            if self.feats_name_to_use_flag[name]:
                for f in fs:
                    idx = hash(f) % self.hash_size
                    vec[0, idx] += 1.0
        return vec.tocsr()

    def _refit(self):
        if self.model == None:
            self.model = lm.Ridge(alpha=self.weight_decay_coeff)

        X = sp.vstack(self.vecs_lst, format='csr')
        y = np.array(self.vals_lst)
        self.model.fit(X, y)

    def save_state(self, folderpath):
        state = {
            'num_evals': len(self.vecs_lst),
            'vals_lst': self.vals_lst,
        }
        ut.write_jsonfile(state,
                          ut.join_paths([folderpath, 'hash_model_state.json']))
        for i, vecs in enumerate(self.vecs_lst):
            sp.save_npz(ut.join_paths([folderpath, str(i) + '.npz']), vecs)

    def load_state(self, folderpath):
        state = ut.read_jsonfile(
            ut.join_paths([folderpath, 'hash_model_state.json']))
        self.vals_lst = state['vals_lst']
        num_evals = state['num_evals']
        for i in range(num_evals):
            self.vecs_lst.append(
                sp.load_npz(ut.join_paths([folderpath,
                                           str(i) + '.npz'])))
        if num_evals > 0:
            self._refit()