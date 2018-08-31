import deep_architect.surrogates as su
from six import iteritems
import scipy.sparse as sp

# NOTE: these are local. these are nice.
# "M.AffineSimplified-0" -> "M.AffineSimplified"
def process_module_feat(s):
    return s[:-s[::-1].index('-') - 1]

# "M.AffineSimplified-1.O.Out |-> M.Nonlinearity-0.I.In" -> "M.AffineSimplified.O.Out |-> M.Nonlinearity.I.In"
def process_connection_feat(s):
    s_proc_lst = []
    for s_it in s.split(' |-> '):
        lst = s_it.split('.')
        lst[1] = process_module_feat(lst[1])
        s_proc_lst.append('.'.join(lst))
    return ' |-> '.join(s_proc_lst)

# "optimizer_type : H.Discrete-7 = adam" -> "optimizer_type = adam"
def process_other_hyperp_feat(s):
    hyperp_name, right = s.split(':')
    hyperp_value = right.split(' = ')[-1]
    return "%s = %s" % (hyperp_name, hyperp_value)

# "M.AffineSimplified-0/m : H.Discrete-5 = 10" -> "M.AffineSimplified/m = 10"
def process_module_hyperp_feat(s):
    s_module, s_hyperp = s.split(' : ')
    # for the module part
    s_module_id, s_module_h = s_module.split('/')
    proc_s_module_id = process_module_feat(s_module_id)
    proc_s_module = '/'.join([proc_s_module_id, s_module_h])
    # for the hyperp part
    s_hyperp_value = s_hyperp.split(' = ')[-1]
    return "%s = %s" % (proc_s_module, s_hyperp_value)

class SimplerHashingSurrogate(su.HashingSurrogate):
    """Same as the su.HashingSurrogate but with a simpler feature function that
    elids identifiers that make the modules and hyperparameters unique.
    """
    def _feats2vec(self, feats):
        name_to_proc_fn = {
            "connection_feats" : process_connection_feat,
            "module_hyperp_feats" : process_module_hyperp_feat,
            "module_feats" : process_module_feat,
            "other_hyperp_feats" : process_other_hyperp_feat,
        }
        vec = sp.dok_matrix((1, self.hash_size), dtype='float')
        for name, fs in iteritems(feats):
            if self.feats_name_to_use_flag[name]:
                proc_fn = name_to_proc_fn[name]
                for f in fs:
                    idx = hash(proc_fn(f)) % self.hash_size
                    vec[0, idx] += 1.0
        return vec.tocsr()