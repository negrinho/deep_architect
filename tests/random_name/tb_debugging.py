### simple tests and debugging that are often useful.
import itertools

def test_overfit(ds, overfit_val, overfit_key):
    ov_ds = []
    err_ds = []

    for d in ds:
        if d[overfit_key] == overfit_val:
            ov_ds.append( d )
        else:
            err_ds.append( d )
    
    return (ov_ds, err_ds)

def test_with_fn(ds, fn):
    ov_ds = []
    err_ds = []

    for d in ds:
        if fn( d ):
            good_ds.append( d )
        else:
            bad_ds.append( d )
    
    return (good_ds, bad_ds)

# assumes that eq_fn is transitive. can be used to check properties of
# a set of results.
def all_equivalent_with_fn(ds, eq_fn):
    for d1, d2 in itertools.izip(ds[:-1], ds[1:]):
        if not eq_fn(d1, d2):
            return False
    return True

def assert_length_consistency(xs_lst):
    assert len( set(map(len, xs_lst) ) ) == 1

    for i in xrange( len(xs_lst) ):
        assert set( [len(xs[i]) for xs in xs_lst] ) == 1

