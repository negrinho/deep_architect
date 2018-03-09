import sys
sys.path.append('../')

import data as da
import evaluator as ev
import darch.searchers as se
import darch.surrogates as su
import search_space as ss
import search_logging as sl
import darch.visualization as vi
from collections import OrderedDict
from six import iteritems
import numpy as np

def running_max(vs):
    return np.maximum.accumulate(vs)

def evaluate_searcher(data, searcher, num_evals, logger):
    rs = []
    for i in xrange(num_evals):
        (inputs, outputs, hs, vs, cfg_d) = searcher.sample()
        r = ev.evaluate_fn(inputs, outputs, hs, data)
        logger.log(inputs, outputs, hs, cfg_d, vs, r)
        searcher.update(r['val_acc'], cfg_d)
        rs.append(r['val_acc'])
    return rs

def main():
    num_evals = 2
    num_repeats = 2
    ns_surr = 128
    search_space_fn = ss.get_ss1_fn(2)
    data = da.load_data(True)

    label2data = OrderedDict([
        ('rand', []), ('mcts', []), ('smbo_rand', []), ('smbo_mcts', [])])
    for label in label2data:
        sl.delete_folder('./' + label, abort_if_notexists=False, abort_if_nonempty=False)
        sl.create_folder('./' + label)

    for i in xrange(num_repeats):
        logger = sl.SearchLogger('rand', 'rand-%d' % i, True)
        searcher = se.RandomSearcher(search_space_fn)
        label2data['rand'].append(evaluate_searcher(data, searcher, num_evals, logger))

        logger = sl.SearchLogger('mcts', 'mcts-%d' % i, True)
        searcher = se.MCTSearcher(search_space_fn, 10.0)
        label2data['mcts'].append(evaluate_searcher(data, searcher, num_evals, logger))

        logger = sl.SearchLogger('smbo_rand', 'smbo_rand-%d' % i, True)
        surr = su.HashingSurrogate(256, 1)
        searcher = se.SMBOSearcher(search_space_fn, surr, ns_surr, 0.1)
        label2data['smbo_rand'].append(evaluate_searcher(data, searcher, num_evals, logger))

        logger = sl.SearchLogger('smbo_mcts', 'smbo_mcts-%d' % i, True)
        surr = su.HashingSurrogate(256, 1)
        searcher = se.SMBOSearcherWithMCTSOptimizer(search_space_fn, surr, ns_surr, 0.1, 1)
        label2data['smbo_mcts'].append(evaluate_searcher(data, searcher, num_evals, logger))

    plotter = vi.LinePlot()
    for (lab, data) in iteritems(label2data):
        vs = np.array([running_max(rs) for rs in data])
        mean = vs.mean(axis=0)
        err = vs.std(axis=0)
        plotter.add_line(np.arange(len(mean)), mean, err=err / np.sqrt(num_repeats), label=lab)
    plotter.plot(fpath='plot.png')

if __name__ == '__main__':
    main()
