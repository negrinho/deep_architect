from deep_architect.searchers.common import random_specify, Searcher
from deep_architect.utils import join_paths, write_jsonfile, read_jsonfile, file_exists
class RandomSearcher(Searcher):
    def __init__(self, search_space_fn, keep_best=1):
        Searcher.__init__(self, search_space_fn)

        self.keep_best = keep_best
        self.best_models = []

    def sample(self):
        inputs, outputs, hs = self.search_space_fn()
        vs = random_specify(outputs.values(), hs.values())
        return inputs, outputs, hs, vs, {'vs': vs}

    def update(self, val, searcher_eval_token):
        self.best_models.append((val, searcher_eval_token['vs']))
        self.best_models.sort(reverse=True)
        if len(self.best_models) > self.keep_best:
            self.best_models = self.best_models[:self.keep_best]

    def save_state(self, folder_name):
        state = {
            "keep_best": self.keep_best,
            "best_models": self.best_models
        }
        write_jsonfile(state, join_paths([folder_name, 'random_searcher.json']))

    def load(self, folder_name):
        filepath = join_paths([folder_name, 'random_searcher.json'])
        if not file_exists(filepath):
            raise RuntimeError("Load file does not exist")

        state = read_jsonfile(filepath)
        self.keep_best = state["keep_best"]
        self.best_models = state["best_models"]

    def get_best(self, num_models):
        return self.best_models[:min(num_models, len(self.best_models))]
