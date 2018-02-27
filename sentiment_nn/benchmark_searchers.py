
import data as da
import evaluator as ev
import darch.searchers as se
import search_space as ss
import search_logging as sl

def evaluate_searcher(searcher, num_evals, logger):
    rs = []
    for i in xrange(num_evals):
        (inputs, outputs, hs, vs, cfg_d) = searcher.sample()
        r = ev.evaluate_fn(inputs, outputs, hs)
        logger.log(inputs, outputs, hs, cfg_d, vs, r))
        searcher.update(r, cfg_d)
        rs.append(r)
    return rs

if __name__ == '__main__':
    sl.cre

    delete_folder('test', abort_if_nonempty=False)
    data = da.load_data(True)
    searcher = se.MCTSearcher(ss.ss1_fn, 0.1)
    logger = SearchLogger('.', 'test')
    for _ in xrange(16):
        (inputs, outputs, hs, vs, cfg_d) = searcher.sample()
        r = ev.evaluate_fn(inputs, outputs, hs, data)
        print vs, r, cfg_d
        searcher.update(r['val_acc'], cfg_d)
        logger.log(inputs, outputs, hs, cfg_d, vs, r)
        # vi.draw_graph(outputs.values(), True)

def main():
    num_evals = 4
    num_repeats = 3
    ns_surr = 128

    label2data = OrderedDict([
        ('rand', []), ('mcts', []), ('smbo_rand', []), ('smbo_mcts', [])])
    for label in label2data:
        sl.delete_folder(label, abort_if_notexists=False, abort_if_nonempty=False)
        sl.create_folder(label)

    for i in xrange(num_repeats):
        logger = sl.SearchLogger('rand', 'rand-%d' % i)
        searcher = se.RandomSearcher(search_space_fn)
        label2data['rand'].append(evaluate_searcher(data, searcher, num_evals, logger))

        logger = sl.SearchLogger('mcts', 'mcts-%d' % i)
        searcher = se.MCTSearcher(search_space_fn, 10.0)
        label2data['mcts'].append(evaluate_searcher(data, searcher, num_evals, logger))

        logger = sl.SearchLogger('smbo_rand', 'smbo_rand-%d' % i)
        surr = su.HashingSurrogate(256, 1)
        searcher = se.SMBOSearcher(search_space_fn, surr, ns_surr, 0.1)
        label2data['smbo_rand'].append(evaluate_searcher(data, searcher, num_evals, logger))

        logger = sl.SearchLogger('smbo_mcts', 'smbo_mcts-%d' % i)
        surr = su.HashingSurrogate(256, 1)
        searcher = se.SMBOSearcherWithMCTSOptimizer(search_space_fn, surr, ns_surr, 0.1, 1)
        label2data['smbo_mcts'].append(evaluate_searcher(data, searcher, num_evals, logger))

    plotter = vi.LinePlot()
    for (lab, data) in iteritems(label2data):
        vs = np.array([running_max(rs) for rs in data])
        mean = vs.mean(axis=0)
        err = vs.std(axis=0)
        plotter.add_line(np.arange(len(mean)), mean, err=err / np.sqrt(num_repeats))
    plotter.plot(fpath='plot.png')