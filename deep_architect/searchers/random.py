from deep_architect.searchers.common import random_specify, Searcher


class RandomSearcher(Searcher):

    def __init__(self, search_space_fn, reset_default_scope_upon_sample=True):
        Searcher.__init__(self, search_space_fn, reset_default_scope_upon_sample)

    def sample(self):
        inputs, outputs = self.search_space_fn()
        vs = random_specify(outputs)
        return inputs, outputs, vs, {}

    def update(self, val, searcher_eval_token):
        pass

    def save_state(self, folderpath):
        pass

    def load_state(self, folderpath):
        pass
