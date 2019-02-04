from deep_architect.searchers.random import RandomSearcher
from deep_architect.searchers.mcts import MCTSSearcher
from deep_architect.searchers.smbo_random import SMBOSearcher
from deep_architect.searchers.smbo_mcts import SMBOSearcherWithMCTSOptimizer
from deep_architect.surrogates.hashing import HashingSurrogate

name_to_get_searcher_fn = {
    'random':
    lambda ssf: RandomSearcher(ssf),
    'smbo_rand_256':
    lambda ssf: SMBOSearcher(ssf, HashingSurrogate(2048, 1), 256, 0.1),
    'smbo_rand_512':
    lambda ssf: SMBOSearcher(ssf, HashingSurrogate(2048, 1), 512, 0.1),
    'smbo_mcts_256':
    lambda ssf: SMBOSearcherWithMCTSOptimizer(ssf, HashingSurrogate(2048, 1),
                                              256, 0.1, 1),
    'smbo_mcts_512':
    lambda ssf: SMBOSearcherWithMCTSOptimizer(ssf, HashingSurrogate(2048, 1),
                                              512, 0.1, 1)
}