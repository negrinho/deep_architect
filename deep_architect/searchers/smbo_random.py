from deep_architect.searchers.common import random_specify, specify, Searcher
from deep_architect.surrogates.common import extract_features
import numpy as np


class SMBOSearcher(Searcher):

    def __init__(self, search_space_fn, surrogate_model, num_samples,
                 exploration_prob):
        Searcher.__init__(self, search_space_fn)
        self.surr_model = surrogate_model
        self.num_samples = num_samples
        self.exploration_prob = exploration_prob

    def sample(self):
        if np.random.rand() < self.exploration_prob:
            while True:
                try:
                    inputs, outputs = self.search_space_fn()
                    best_vs = random_specify(outputs.values())
                    break
                except ValueError:
                    pass
        else:
            best_model = None
            best_vs = None
            best_score = -np.inf
            for i in range(self.num_samples):
                while True:
                    try:
                        inputs, outputs = self.search_space_fn()
                        vs = random_specify(outputs.values())
                        break
                    except ValueError:
                        pass

                feats = extract_features(inputs, outputs)
                score = self.surr_model.eval(feats)
                if score > best_score:
                    best_model = (inputs, outputs)
                    best_vs = vs
                    best_score = score

            inputs, outputs = best_model

        searcher_eval_token = {'vs': best_vs}
        return inputs, outputs, best_vs, searcher_eval_token

    def update(self, val, searcher_eval_token):
        (inputs, outputs) = self.search_space_fn()
        specify(outputs.values(), searcher_eval_token['vs'])
        feats = extract_features(inputs, outputs)
        self.surr_model.update(val, feats)

    def save_state(self, folder):
        self.surr_model.save_state(folder)

    def load_state(self, folder):
        self.surr_model.load_state(folder)
