import json


def write_jsonfile(d, fpath, sort_keys=False):
    with open(fpath, 'w') as f:
        json.dump(d, f, indent=4, sort_keys=sort_keys)


def read_jsonfile(fpath):
    with open(fpath, 'r') as f:
        d = json.load(f)
        return d
