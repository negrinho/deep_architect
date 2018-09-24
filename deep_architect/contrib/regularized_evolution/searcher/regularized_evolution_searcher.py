"""
Implementation of Regularized Evolution searcher from Zoph et al '18
"""
from __future__ import print_function

from builtins import range
import random
from collections import deque

from deep_architect.utils import join_paths, write_jsonfile, read_jsonfile, file_exists
from deep_architect.searchers.common import Searcher, random_specify_hyperparameter
from deep_architect.core import unassigned_independent_hyperparameter_iterator

def mutatable(h):
    return h.get_name().startswith('H.Mutatable')

def mutate(output_lst, user_vs, all_vs, mutatable_fn, search_space_fn):
    mutate_candidates = []
    new_vs = list(user_vs)
    for i, h in enumerate(unassigned_independent_hyperparameter_iterator(output_lst)):
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

    inputs, outputs, hs = search_space_fn()
    output_lst = list(outputs.values())
    all_vs = specify_evolution(output_lst, mutatable_fn, new_vs, hs)
    return inputs, outputs, hs, new_vs, all_vs

def random_specify_evolution(output_lst, mutatable_fn, hyperp_lst=None):
    user_vs = []
    all_vs = []
    for h in unassigned_independent_hyperparameter_iterator(output_lst, hyperp_lst):
        v = random_specify_hyperparameter(h)
        if mutatable_fn(h):
            user_vs.append(v)
        all_vs.append(v)
    return user_vs, all_vs

def specify_evolution(output_lst, mutatable_fn, user_vs, hyperp_lst=None):
    vs_idx = 0
    vs = []
    for i, h in enumerate(unassigned_independent_hyperparameter_iterator(output_lst, hyperp_lst)):
        if mutatable_fn(h):
            h.assign_value(user_vs[vs_idx])
            vs.append(user_vs[vs_idx])
            vs_idx += 1
        else:
            v = random_specify_hyperparameter(h)
            vs.append(v)
    return vs

class EvolutionSearcher(Searcher):
    def __init__(self, search_space_fn, mutatable_fn, P, S, regularized=False):
        Searcher.__init__(self, search_space_fn)

        # Population size
        self.P = P
        self.S = S
        # Sample size

        self.population = deque(maxlen=P)
        self.processing = []
        self.regularized = regularized
        self.initializing = True
        self.mutatable = mutatable_fn
        print(self.mutatable)

    def sample(self):
        if self.initializing:
            inputs, outputs, hs = self.search_space_fn()
            user_vs, all_vs = random_specify_evolution(
                list(outputs.values()), self.mutatable, list(hs.values()))
            if len(self.population) >= self.P - 1:
                self.initializing = False
            return inputs, outputs, hs, all_vs, {'user_vs': user_vs, 'all_vs': all_vs}
        else:
            sample_inds = sorted(random.sample(
                list(range(len(self.population))), min(self.S, len(self.population))))
            # delete weakest model
            weak_ind = self.get_weakest_model_index(sample_inds)

            # mutate strongest model
            inputs, outputs, hs = self.search_space_fn()
            user_vs, all_vs, _ = self.population[self.get_strongest_model_index(
                sample_inds)]
            inputs, outputs, hs, new_user_vs, new_all_vs = mutate(
                list(outputs.values()), user_vs, all_vs, self.mutatable, 
                self.search_space_fn)

            self.processing.append(self.population[weak_ind])
            del self.population[weak_ind]
            return inputs, outputs, hs, new_all_vs, {'user_vs': new_user_vs, 'all_vs': new_all_vs}

    def update(self, val, cfg_d):
        arc = (cfg_d['user_vs'], cfg_d['all_vs'], val)
        if arc in self.processing:
            self.processing.remove(arc)
        self.population.append((
            cfg_d['user_vs'], 
            cfg_d['all_vs'], 
            val))

    def get_searcher_state_token(self):
        for arc in self.processing:
            if len(self.population) >= self.P:
                break
            self.population.append(arc)
        return {
            "P": self.P,
            "S": self.S,
            "population": list(self.population),
            "regularized": self.regularized,
            "initializing": self.initializing,
        }

    def save_state(self, folder_name):
        state = self.get_searcher_state_token()
        write_jsonfile(state, join_paths([folder_name, 'evolution_searcher.json']))

    def load(self, folder_name):
        filepath = join_paths([folder_name, 'evolution_searcher.json'])
        if not file_exists(filepath):
            raise RuntimeError("Load file does not exist")
        
        state = read_jsonfile(filepath)
        self.P = state["P"]
        self.S = state["S"]
        self.regularized = state['regularized']
        self.population = deque(state['population'])
        self.initializing = state['initializing']


    def get_weakest_model_index(self, sample_inds):
        if self.regularized:
            return sample_inds[0]
        else:
            min_acc = 1.
            min_acc_ind = -1
            for i in range(len(sample_inds)):
                _, _, acc = self.population[sample_inds[i]]
                if acc < min_acc:
                    min_acc = acc
                    min_acc_ind = i
            return sample_inds[min_acc_ind]

    def get_strongest_model_index(self, sample_inds):
        max_acc = 0.
        max_acc_ind = -1
        for i in range(len(sample_inds)):
            _, _, acc = self.population[sample_inds[i]]
            if acc > max_acc:
                max_acc = acc
                max_acc_ind = i
        return sample_inds[max_acc_ind]