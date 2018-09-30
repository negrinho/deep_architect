from deep_architect.searchers.common import random_specify, specify, Searcher
from deep_architect.surrogates.common import extract_features
import numpy as np


class SMBOSearcher(Searcher):
    def __init__(self, search_space_fn, surrogate_model, num_samples,
                 eps_prob):
        Searcher.__init__(self, search_space_fn)
        self.surr_model = surrogate_model
        self.num_samples = num_samples
        self.eps_prob = eps_prob

    def sample(self):
        if np.random.rand() < self.eps_prob:
            inputs, outputs, hyperps = self.search_space_fn()
            best_vs = random_specify(outputs.values(), hyperps.values())
        else:
            best_model = None
            best_vs = None
            best_score = -np.inf
            for _ in range(self.num_samples):
                inputs, outputs, hyperps = self.search_space_fn()
                vs = random_specify(outputs.values(), hyperps.values())

                feats = extract_features(inputs, outputs, hyperps)
                score = self.surr_model.eval(feats)
                if score > best_score:
                    best_model = (inputs, outputs, hyperps)
                    best_vs = vs
                    best_score = score

            inputs, outputs, hyperps = best_model

        searcher_eval_token = {'vs': best_vs}
        return inputs, outputs, hyperps, best_vs, searcher_eval_token

    def update(self, val, searcher_eval_token):
        (inputs, outputs, hyperps) = self.search_space_fn()
        specify(outputs.values(), hyperps.values(), searcher_eval_token['vs'])
        feats = extract_features(inputs, outputs, hyperps)
        self.surr_model.update(val, feats)
