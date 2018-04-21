import darch.core as co
import sklearn.linear_model as lm
import scipy.sparse as sp
import numpy as np
from six import iteritems, itervalues

class SurrogateModel:
    """Abstract class for a surrogate model.
    """
    def eval(self, feats):
        """ Returns a prediction of performance (or other relevant metrics),
        given a feature representation of the architecture.
        """
        raise NotImplementedError

    def update(self, val, feats):
        """Updates the state of the surrogate function given the feature
        representation for the architecture and the corresponding ground truth
        performance metric.

        .. note::
            The data for the surrogate model may be kept internally. The update
            of the state of the surrogate function can be done periodically, rather
            than with each call to update. This can be done based on values for
            the configuration of the surrogate model instance.
        """
        raise NotImplementedError

class DummySurrogate(SurrogateModel):
    def __init__(self):
        self.val_lst = []

    def eval(self, feats):
        if len(self.val_lst) == 0:
            return 0.0
        else:
            return np.mean(self.val_lst)

    def update(self, val, feats):
        self.val_lst.append(val)

class HashingSurrogate(SurrogateModel):
    """Simple hashing surrogate function that simply hashes the strings in the
    feature representation of the architecture to buckets. See
    :func:`darch.surrogates.extract_features` for the functions that is used to
    extract string features for the architectures.
    """
    def __init__(self, hash_size, refit_interval, weight_decay_coeff=1e-5,
            use_module_feats=True, use_connection_feats=True,
            use_module_hyperp_feats=True, use_other_hyperp_feats=True):
        self.hash_size = hash_size
        self.refit_interval = refit_interval
        self.weight_decay_coeff = weight_decay_coeff
        self.feats_name_to_use_flag = {
            'connection_feats' : use_connection_feats,
            'module_hyperp_feats' : use_module_hyperp_feats,
            'module_feats' : use_module_feats,
            'other_hyperp_feats' : use_other_hyperp_feats}
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

# extract some simple features from the network. useful for smbo surrogate models.
def extract_features(inputs, outputs, hs):
    """Extract a feature representation of a model represented through inputs,
    outputs, and hyperparameters.

    This function has been mostly used for performance prediction on fully
    specified models, i.e., after all the hyperparameters in the search space
    have specified. After this, there is a single model for which we can compute
    an appropriate feature representation containing information about the
    connections that the model makes and the values of the hyperparameters.

    Args:
        inputs (dict[str, darch.core.Input]): Dictionary mapping names
            to inputs of the architecture.
        outputs (dict[str, darch.core.Output]): Dictionary mapping names to outputs
            of the architecture.
        hs (dict[str, darch.core.Hyperparameter]): Dictionary mappings names to
            hyperparameters of the architecture.

    Returns:
        dict[str, list[str]]:
            Representation of the architecture as a dictionary where each
            key is associated to a list with different types of features.
    """
    module_memo = co.OrderedSet()

    module_feats = []
    connection_feats = []
    module_hyperp_feats = []
    other_hyperps_feats = []

    # getting all the modules
    module_memo = []
    def fn(m):
        module_memo.append(m)
    co.traverse_backward(outputs.values(), fn)

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
                    connection_feats.append(c_feats)

        # module hyperparameters
        for h_localname, h in iteritems(m.hyperps):
            mh_feats = "%s/%s : %s = %s" % (
                m.get_name(), h_localname, h.get_name(), h.get_val())
            module_hyperp_feats.append(mh_feats)

    # other features
    for h_localname, h in iteritems(hs):
        oh_feats = "%s : %s = %s" % (h_localname, h.get_name(), h.get_val())
        other_hyperps_feats.append(oh_feats)

    return {'module_feats' : module_feats,
            'connection_feats' : connection_feats,
            'module_hyperp_feats' : module_hyperp_feats,
            'other_hyperp_feats' : other_hyperps_feats }
