
import data as da
import evaluator as ev
import darch.searchers as se
import search_space as ss

if __name__ == '__main__':
    data = da.load_data(False)
    searcher = se.MCTSearcher(ss.get_ss1_fn(2), 0.1)
    for _ in xrange(128):
        (inputs, outputs, hs, vs, cfg_d) = searcher.sample()
        r = ev.evaluate_fn(inputs, outputs, hs, data)
        print vs, r, cfg_d
        searcher.update(r['val_acc'], cfg_d)