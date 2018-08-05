from darch.searchers.common import random_specify, Searcher

class RandomSearcher(Searcher):
    def sample(self):
        inputs, outputs, hs = self.search_space_fn()
        vs = random_specify(outputs.values(), hs.values())
        return inputs, outputs, hs, vs, {}

    def update(self, val, searcher_eval_token):
        pass

