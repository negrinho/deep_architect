### auxiliary functions for avoiding boilerplate in in assignments
def set_obj_vars(obj, d, 
        abort_if_exists=False, abort_if_notexists=True):
    d_to = vars(obj)
    assert (not abort_if_exists) or all(
        [k not in d_to for k in d.iterkeys()])
    assert (not abort_if_notexists) or all(
        [k in d_to for k in d.iterkeys()])    

    for k, v in d.iteritems():
        assert not hasattr(obj, k) 
        setattr(obj, k, v)
     
def retrieve_obj_vars(obj, var_names, tuple_fmt=False):
    return retrieve_values(vars(obj), var_names, tuple_fmt)

### partial application and other functional primitives
import functools

def partial_apply(fn, d):
    return functools.partial(fn, **d)

def to_list_fn(f):
    return lambda xs: map(f, xs)

def transform(x, fns):
    for f in fns:
        x = f( x )
    return y

def zip_toggle(xs):
    """[[x1, ...], [x2, ...], [x3, ...]] --> [(x1, x2, .., xn) ... ];
        [(x1, x2, .., xn) ... ] --> [[x1, ...], [x2, ...], [x3, ...]]"""
    assert isinstance(xs, list)
    return zip(*xs)

### useful iterators
import itertools

def iter_product(lst_lst_vals, tuple_fmt=True):
    vs = list(itertools.product(*lst_lst_vals))
    if not tuple_fmt:
        vs = map(list, vs)

    return vs

def iter_ortho_all(lst_lst_vals, ref_idxs, ignore_repeats=True):
    assert len(lst_lst_vals) == len(ref_idxs)
    ref_r = [lst_lst_vals[pos][idx] for (pos, idx) in enumerate(ref_idxs)]

    # put reference first in this case, if ignoring repeats
    rs = [] if not ignore_repeats else [tuple(ref_r)]

    num_lsts = len(lst_lst_vals)
    for i in xrange( num_lsts ):
        num_vals = len(lst_lst_vals[i])
        for j in xrange( num_vals ):
            if ignore_repeats and j == ref_idxs[i]:
                continue

            r = list(ref_r)
            r[i] = lst_lst_vals[i][j]
            rs.append(tuple(r))
    return rs

def iter_ortho_single(lst_lst_vals, ref_idxs, idx_it, ref_first=True):
    assert len(lst_lst_vals) == len(ref_idxs)
    ref_r = [lst_lst_vals[pos][idx] for (pos, idx) in enumerate(ref_idxs)]
    
    rs = [] if not ref_first else [tuple(ref_r)]

    num_vals = len(lst_lst_vals[idx_it])
    for j in xrange( num_vals ):
        if ref_first and j == ref_idxs[idx_it]:
            continue

        r = [lst_lst_vals[pos][idx] for (pos, idx) in enumerate(ref_idxs)]
        r[i] = lst_lst_vals[idx_it][j]
        rs.append(tuple(r))
    return rs

import inspect 

def get_argnames(fn):
    return inspect.getargspec(fn).args

### dictionary manipulation
import pprint
import pandas

def create_dict(ks, vs):
    assert len(ks) == len(vs)
    return dict(zip(ks, vs))

def create_dataframe(ds, abort_if_different_keys=True):
    ks = key_union(ds)
    assert (not abort_if_different_keys) or len( key_intersection(ds) ) == len( ks ) 

    df_d = {k : [] for k in ks }
    for d in ds:
        for k in ks:
            if k not in d:
                df_d[k].append( None )
            else:
                df_d[k].append( d[k] )
    
    df = pandas.DataFrame(df_d)
    return df

def copy_update_dict(d, d_other):
    proc_d = dict(d)
    proc_d.update(d_other)

    return proc_d

def merge_dicts(ds):
    out_d = {}
    for d in ds:
        for (k, v) in d.iteritems():
            assert k not in out_d
            out_d[k] = v
    return out_d

def groupby(xs, fn):
    assert isinstance(xs, list)  
    
    d = {}
    for x in xs:
        fx = fn( x )
        if fx not in d:
            d[fx] = []

        d[fx].append(x)
    return d

def flatten(d):
    assert isinstance(d, dict)

    xs = []
    for (_, k_xs) in d.iteritems():
        assert isinstance(k_xs, list)
        xs.extend(k_xs)
    return xs

def recursive_groupby(p, fn):
    assert isinstance(p, dict) or isinstance(p, list)

    if isinstance(p, list):
        return groupby(p, fn)
    else:
        return {k : recursive_groupby(k_p, fn) for (k, k_p) in p.iteritems()}

def recursive_flatten(p):
    assert isinstance(p, dict) or isinstance(p, list)

    if isinstance(p, list):
        return list(p)
    else:
        xs = []
        for (_, k_p) in p.iteritems():
            xs.extend( recursive_flatten(k_p) )
        return xs

def recursive_map(p, fn):
    assert isinstance(p, dict) or isinstance(p, list)

    if isinstance(p, list):
        return map(fn, p)
    else:
        return {k : recursive_map(k_p, fn) for (k, k_p) in p.iteritems()}

def recursive_index(d, ks):
    for k in ks:
        d = d[k]
    return d

def filter_dict(d, fn):
    return { k : v for (k, v) in d.iteritems() if fn(k, v) }

def map_dict(d, fn):
    return { k : fn(k, v) for (k, v) in d.iteritems() }

def structure(ds, ks):
    get_fn = lambda k: lambda x: x[k]

    for k in ks:
        ds = recursive_groupby(ds, get_fn(k))
    return ds

def structure_with_fns(ds, fns):
    for fn in fns:
        ds = recursive_groupby(ds, fn)
    return ds

def get_subset_indexing_fn(ks, tuple_fmt=True):
    def fn(d):
        assert isinstance(d, dict)
        out = [d[k] for k in ks]
        if tuple_fmt:
            out = tuple(out)
        return out
    return fn

def flatten_nested_list(xs):
    assert isinstance(xs, list)

    xs_res = []
    for x_lst in xs:
        assert isinstance(x_lst, list)
        xs_res.extend(x_lst)
    return xs_res

def key_union(ds):
    ks = []
    for d in ds:
        ks.extend(d.keys())
    return list(set(ks))

def key_intersection(ds):
    assert len(ds) > 0

    ks = set(ds[0].keys())
    for d in ds[1:]:
        ks.intersection_update(d.keys())
    return list(ks)

# NOTE: right now, this is done for dictionaries with hashable values.
def key_to_values(ds):
    out_d = {}
    for d in ds:
        for (k, v) in d.iteritems():
            if k not in out_d:
                out_d[k] = set()
            out_d[k].add(v)
    return out_d

def retrieve_values(d, ks, tuple_fmt=False):
    out_d = {}
    for k in ks:
        out_d[k] = d[k]
    
    if tuple_fmt:
        out_d = tuple([out_d[k] for k in ks])
    return out_d

def set_values(d_to, d_from, 
        abort_if_exists=False, abort_if_notexists=True):
    assert (not abort_if_exists) or all(
        [k not in d_to for k in d_from.iterkeys()])
    assert (not abort_if_notexists) or all(
        [k in d_to for k in d_from.iterkeys()])
    
    d_to.update(d_from)

def sort_items(d, by_key=True, decreasing=False):
    key_fn = (lambda x: x[0]) if by_key else (lambda x: x[1])
    return sorted(d.items(), key=key_fn, reverse=decreasing)

def print_dict(d, width=1):
    pprint.pprint(d, width=width)

def collapse_nested_dict(d, sep='.'):
    assert all([type(k) == str for k in d.iterkeys()]) and (
        all([all([type(kk) == str for kk in d[k].iterkeys()]) for k in d.iterkeys()]))
    
    ps = []
    for k in d.iterkeys():
        for (kk, v) in d[k].iteritems():
            p = (k + sep + kk, v)
            ps.append(p)
    return dict(ps)

def uncollapse_nested_dict(d, sep='.'):
    assert all([type(k) == str for k in d.iterkeys()]) and (
        all([len(k.split()) == 2 for k in d.iterkeys()]))

    out_d = []
    for (k, v) in d.iteritems():
        (k1, k2) = k.split()
        if k1 not in out_d:
            d[k1] = {}
        d[k1][k2] = v

    return out_d

