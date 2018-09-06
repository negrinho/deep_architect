from deep_architect.searchers.common import random_specify, Searcher

class RandomSearcher(Searcher):
    def __init__(self, search_space_fn):
        Searcher.__init__(self, search_space_fn)

        self.best_vs = None
        self.best_acc = 0

    def sample(self):
        inputs, outputs, hs = self.search_space_fn()
        vs = random_specify(outputs.values(), hs.values())
        return inputs, outputs, hs, vs, {'vs': vs}

    def update(self, val, searcher_eval_token):
        if val > self.best_acc:
            self.best_vs = searcher_eval_token['vs']


