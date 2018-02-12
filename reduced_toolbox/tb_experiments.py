from six import iteritems, itervalues
from reduced_toolbox.tb_io import write_jsonfile, read_jsonfile


class SummaryDict:
    def __init__(self, fpath=None, abort_if_different_lengths=False):
        self.abort_if_different_lengths = abort_if_different_lengths
        if fpath != None:
            self.d = self._read(fpath)
        else:
            self.d = {}
        self._check_consistency()

    def append(self, d):
        for k, v in iteritems(d):
            assert type(k) == str and len(k) > 0
            if k not in self.d:
                self.d[k] = []
            self.d[k].append(v)

        self._check_consistency()

    ### NOTE: I don't think that read makes sense anymore because you
    # can keep think as a json file or something else.
    def write(self, fpath):
        write_jsonfile(self.d, fpath)

    def _read(self, fpath):
        return read_jsonfile(fpath)

    def _check_consistency(self):
        assert (not self.abort_if_different_lengths) or (len(
            set([len(v) for v in itervalues(self.d)])) <= 1)

    def get_dict(self):
        return dict(self.d)