"""
TODO:
(1) Better testing
(2) Some cleaning with search_eval_token
(3) Need better error message:
    - When number of architecture sample exceed the total architectures in the 
    search space
(4) Maybe there's a better way to use lazy evaluation, so don't need to gather
    all configurations up front and just gather up to number of architectures
"""
import copy

import darch.searchers as se 
import darch.hyperparameters as hp
import darch.core as co

from src.search_space import get_search_space

class ExhaustiveSearcher(se.Searcher): 
    """
    """
    def __init__(self, search_space_fn): 
        se.Searcher.__init__(self, search_space_fn)
        self.configs = self._get_all_configs(search_space_fn)
        self.current_config = 0
        self.total_configs = len(self.configs)

    def _get_all_configs(self, search_space_fn): 
        """This is effectively dfs"""
        configs = []
        h_history = []
        def dfs(): 
            inputs, outputs, hs = self.search_space_fn()
            for i, h in enumerate(se.unset_hyperparameter_iterator(outputs.values(), hs.values())):
                if (i < len(h_history)): 
                    self.specify_hyperparameter(h, h_history[i])
                else: 
                    for val in h.vs: 
                        h_history.append(val)
                        all_specified = dfs()
                        if (all_specified): 
                            cur_config = copy.copy(h_history)
                            configs.append(cur_config)
                        h_history.pop()
                    return False if len(h_history) > 0 else True
            return True

        finished = dfs()
        assert finished
        return configs

    def specify_hyperparameter(self, h, v): 
        assert not h.is_set()
        if (isinstance(h, hp.Discrete)): 
            h.set_val(v)
        else: 
            raise ValueError

    def sample(self): 
        """
        TODO: 
        (1) Put self.current_config to update function, make more sense
        """
        inputs, outputs, hs = self.search_space_fn()
        vs = self.configs[self.current_config]
        se.specify(outputs.values(), hs, vs)
        self.current_config += 1
        return inputs, outputs, hs, vs, {'current_config_index': self.current_config-1,
                                         'total_configs': self.total_configs}

    def update(self, val, searcher_eval_token):
        # NOTE: I think searcher_eval_token can be the number of architectures 
        pass 

def test_exhaustive_search(): 

    # test case 1
    search_space_fn1 = get_search_space('cnn_baseline', 125) # NOTE: this is hard-coded, need clean up
    searcher = ExhaustiveSearcher(search_space_fn1)
    assert(searcher.total_configs == 16)
    # print(searcher.configs)
    searcher.sample()
    assert(searcher.current_config == 1)

    # test case 2
    search_space_fn2 = get_search_space('residual_cnn_baseline', 125) # NOTE: this is hard-coded, need clean up
    searcher = ExhaustiveSearcher(search_space_fn2)
    # print(searcher.total_configs)
    # assert(searcher.total_configs == 24)
    # print(searcher.configs)
    searcher.sample()
    assert(searcher.current_config == 1)

    # test case 3
    search_space_fn2 = get_search_space('interleave', 125) # NOTE: this is hard-coded, need clean up
    searcher = ExhaustiveSearcher(search_space_fn2)
    print(searcher.total_configs)
    # assert(searcher.total_configs == 24)
    print(searcher.configs)
    searcher.sample()
    assert(searcher.current_config == 1)
    

# test_exhaustive_search()