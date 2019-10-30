"""
Implementation of Regularized Evolution searcher from Zoph et al '18
"""
import random
from collections import deque

import deep_architect.utils as ut
from deep_architect.searchers.common import Searcher, random_specify_hyperparameter
import deep_architect.core as co
import numpy as np


def mutatable(h):
    return len(h.vs) > 1


def mutate(outputs, user_vs, all_vs, mutatable_fn, search_space_fn):
    mutate_candidates = []
    new_vs = list(user_vs)
    for i, h in enumerate(
            co.unassigned_independent_hyperparameter_iterator(outputs)):
        if mutatable_fn(h):
            mutate_candidates.append(h)
        h.assign_value(all_vs[i])

    # mutate a random hyperparameter
    assert len(mutate_candidates) == len(user_vs)
    m_ind = random.randint(0, len(mutate_candidates) - 1)
    m_h = mutate_candidates[m_ind]
    v = m_h.vs[random.randint(0, len(m_h.vs) - 1)]

    # ensure that same value is not chosen again
    while v == user_vs[m_ind]:
        v = m_h.vs[random.randint(0, len(m_h.vs) - 1)]
    new_vs[m_ind] = v
    if 'sub' in m_h.get_name():
        new_vs = new_vs[:m_ind + 1]

    inputs, outputs = search_space_fn()
    all_vs = specify_evolution(outputs, mutatable_fn, new_vs)
    return inputs, outputs, new_vs, all_vs


def random_specify_evolution(outputs, mutatable_fn):
    user_vs = []
    all_vs = []
    for h in co.unassigned_independent_hyperparameter_iterator(outputs):
        v = random_specify_hyperparameter(h)
        if mutatable_fn(h):
            user_vs.append(v)
        all_vs.append(v)
    return user_vs, all_vs


def specify_evolution(outputs, mutatable_fn, user_vs):
    vs_idx = 0
    vs = []
    for i, h in enumerate(
            co.unassigned_independent_hyperparameter_iterator(outputs)):
        if mutatable_fn(h):
            if vs_idx >= len(user_vs):
                user_vs.append(h.vs[random.randint(0, len(h.vs) - 1)])
            h.assign_value(user_vs[vs_idx])
            vs.append(user_vs[vs_idx])
            vs_idx += 1
        else:
            v = random_specify_hyperparameter(h)
            vs.append(v)
    return vs


class EvolutionSearcher(Searcher):

    def __init__(self,
                 search_space_fn,
                 mutatable_fn,
                 P,
                 S,
                 regularized=False,
                 reset_default_scope_upon_sample=True):
        Searcher.__init__(self, search_space_fn,
                          reset_default_scope_upon_sample)

        # Population size
        self.P = P
        # Sample size
        self.S = S

        self.population = deque(maxlen=P)
        # self.processing = []
        self.regularized = regularized
        self.initializing = True
        self.mutatable = mutatable_fn

    def sample(self):
        if self.initializing:
            inputs, outputs = self.search_space_fn()
            user_vs, all_vs = random_specify_evolution(outputs, self.mutatable)
            if len(self.population) >= self.P - 1:
                self.initializing = False
            return inputs, outputs, all_vs, {
                'user_vs': user_vs,
                'all_vs': all_vs
            }
        else:
            sample_inds = sorted(
                random.sample(list(range(len(self.population))),
                              min(self.S, len(self.population))))
            # delete weakest model

            # mutate strongest model
            inputs, outputs = self.search_space_fn()
            user_vs, all_vs, _ = self.population[
                self._get_strongest_model_index(sample_inds)]
            inputs, outputs, new_user_vs, new_all_vs = mutate(
                outputs, user_vs, all_vs, self.mutatable, self.search_space_fn)

            # self.processing.append(self.population[weak_ind])
            # del self.population[weak_ind]
            return inputs, outputs, new_all_vs, {
                'user_vs': new_user_vs,
                'all_vs': new_all_vs
            }

    def update(self, val, searcher_eval_token):
        if not self.initializing:
            weak_ind = self._get_weakest_model_index()
            del self.population[weak_ind]
        self.population.append((searcher_eval_token['user_vs'],
                                searcher_eval_token['all_vs'], val))

    def save_state(self, folderpath):
        filepath = ut.join_paths([folderpath, 'evolution_searcher.json'])
        state = {
            "P": self.P,
            "S": self.S,
            "population": list(self.population),
            "regularized": self.regularized,
            "initializing": self.initializing,
        }
        ut.write_jsonfile(state, filepath)

    def load_state(self, folderpath):
        filepath = ut.join_paths([folderpath, 'evolution_searcher.json'])
        state = ut.read_jsonfile(filepath)
        self.P = state["P"]
        self.S = state["S"]
        self.regularized = state['regularized']
        self.population = deque(state['population'])
        self.initializing = state['initializing']

    def _get_weakest_model_index(self):
        if self.regularized:
            return 0
        else:
            min_score = np.inf
            min_score_ind = None
            for i in range(len(self.population)):
                _, _, score = self.population[i]
                if score < min_score:
                    min_score = score
                    min_score_ind = i
            return min_score_ind

    def _get_strongest_model_index(self, sample_inds):
        max_score = -np.inf
        max_score_ind = -1
        for i in range(len(sample_inds)):
            _, _, score = self.population[sample_inds[i]]
            if score > max_score:
                max_score = score
                max_score_ind = i
        return sample_inds[max_score_ind]

    def get_best(self, num_models):
        ranked_population = sorted(self.population,
                                   reverse=True,
                                   key=lambda tup: tup[2])

        return [(model[2], model[1]) for model in ranked_population[:num_models]
               ]
