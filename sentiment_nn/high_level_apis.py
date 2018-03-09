import data as da
import evaluator as ev
import darch.searchers as se
import search_space as ss


class SearchSpace:
    def __init__(self):
        pass

    def get_search_space(self):
        pass

class Evaluator:
    def __init__(self):
        pass

    def evaluate(self):
        pass

# do a few evaluators.
# create a sentiment_nn data folder.


def run_searcher(evaluator, searcher):
    pass


# evaluator, search_space, searcher
def run_searcher():
    data = da.load_data(False)
    searcher = se.MCTSearcher(ss.ss1_fn, 0.1)
    for _ in xrange(128):
        (inputs, outputs, hs, vs, cfg_d) = searcher.sample()
        r = ev.evaluate_fn(inputs, outputs, hs, data)
        print vs, r, cfg_d
        searcher.update(r['val_acc'], cfg_d)



if __name__ == '__main__':
    data = da.load_data(False)
    searcher = se.MCTSearcher(ss.ss1_fn, 0.1)
    for _ in xrange(128):
        (inputs, outputs, hs, vs, cfg_d) = searcher.sample()
        r = ev.evaluate_fn(inputs, outputs, hs, data)
        print vs, r, cfg_d
        searcher.update(r['val_acc'], cfg_d)