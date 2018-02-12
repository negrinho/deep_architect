### simple tools for randomization
import random

def shuffle(xs):
    idxs = range( len(xs) )
    random.shuffle(idxs)

    return [xs[i] for i in idxs]

def random_permutation(n):
    idxs = range( n )
    random.shuffle(idxs)

    return idxs

def argsort(xs, fns, increasing=True):
    """The functions in fns are used to compute a key which are then used to 
    construct a tuple which is then used to sort. The earlier keys are more 
    important than the later ones.
    """

    def key_fn(x):
        return tuple( [f(x) for f in fns] )
    
    idxs, _ = zip_toggle( 
        sorted( enumerate( xs ), 
            key=lambda x: key_fn( x[1] ), 
            reverse=not increasing ) )

    return idxs

def sort(xs, fns, increasing=True):
    idxs = argsort(xs, fns, increasing)
    return apply_permutation(xs, idxs)

def apply_permutation(xs, idxs):
    assert len( set(idxs).intersection( range( len(xs) ) ) ) == len(xs)

    return [xs[i] for i in idxs]

def apply_inverse_permutation(xs, idxs):
    assert len( set(idxs).intersection( range( len(xs) ) ) ) == len(xs)
    
    out_xs = [None] * len(xs)
    for i_from, i_to in enumerate(idxs):
        out_xs[i_to] = xs[i_from]
    
    return out_xs

def shuffle_tied(xs_lst):
    assert len(xs_lst) > 0 and len(map(len, xs_lst)) == 1

    n = len( xs_lst[0] )
    idxs = random_permutation(n)
    ys_lst = [apply_permutation(xs, idxs) for xs in xs_lst]
    return ys_lst

### simple randomness
import random
import numpy as np

def set_random_seed(x=None):
    np.random.seed(x)
    random.seed(x)

def uniform_sample_product(lst_lst_vals, num_samples):
    n = len(lst_lst_vals)
    components = []
    for i, lst in enumerate(lst_lst_vals):
        idxs = np.random.randint(len(lst), size=num_samples)
        components.append( [lst[j] for j in idxs] )

    samples = []
    for i in xrange(num_samples):
        samples.append([ components[j][i] for j in xrange(n) ])
    return samples

### TODO: not finished
def sample(xs, num_samples, with_replacement=True, p=None):
    if p == None:
        pass
