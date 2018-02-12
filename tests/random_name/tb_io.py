### simple io
import json
import sys
import pickle

def read_textfile(fpath, strip=True):
    with open(fpath, 'r') as f:
        lines = f.readlines()
        if strip:
            lines = [line.strip() for line in lines]
        return lines

def write_textfile(fpath, lines, append=False, with_newline=True):
    mode = 'a' if append else 'w'

    with open(fpath, mode) as f:
        for line in lines:
            f.write(line)
            if with_newline:
                f.write("\n")

def read_dictfile(fpath, sep='\t'):
    """This function is conceived for dicts with string keys."""
    lines = read_textfile(fpath)
    d = dict([line.split(sep, 1) for line in lines])
    return d

def write_dictfile(fpath, d, sep="\t", ks=None):
    """This function is conceived for dicts with string keys. A set of keys 
    can be passed to specify an ordering to write the keys to disk.
    """
    if ks == None:
        ks = d.keys()
    assert all([type(k) == str and sep not in k for k in ks])
    lines = [sep.join([k, str(d[k])]) for k in ks]
    write_textfile(fpath, lines)

def read_jsonfile(fpath):
    with open(fpath, 'r') as f:
        d = json.load(f)
        return d

def write_jsonfile(d, fpath, sort_keys=False):
    with open(fpath, 'w') as f:
        json.dump(d, f, indent=4, sort_keys=sort_keys)

def read_picklefile(fpath):
    with open(fpath, 'rb') as f:
        return pickle.load(f)

def write_picklefile(x, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(x, f)

# NOTE: this function can use some existing functionality from python to 
# make my life easier. I don't want fields manually. 
# perhaps easier when supported in some cases.
def read_csvfile(fpath, sep=',', has_header=True):
    raise NotImplementedError

# TODO: there is also probably some functionality for this.
def write_csvfile(ds, fpath, sep=',',
        write_header=True, abort_if_different_keys=True, sort_keys=False):

    ks = key_union(ds)
    if sort_keys:
        ks.sort()

    assert (not abort_if_different_keys) or len( key_intersection(ds) ) == len(ks)

    lines = []
    if write_header:
        lines.append( sep.join(ks) )
    
    for d in ds:
        lines.append( sep.join([str(d[k]) if k in d else '' for k in ks]) )

    write_textfile(fpath, lines)
