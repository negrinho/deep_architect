
import darch.searchers as se
import darch.surrogates as su

name_to_get_searcher_fn = {
    'random' : lambda ssf: se.RandomSearcher(ssf),
    'smbo_rand_256' : lambda ssf: se.SMBOSearcher(ssf, su.HashingSurrogate(2048, 1), 256, 0.1),
    'smbo_rand_512' : lambda ssf: se.SMBOSearcher(ssf, su.HashingSurrogate(2048, 1), 512, 0.1),
    'smbo_mcts_256' : lambda ssf: se.SMBOSearcherWithMCTSOptimizer(ssf, su.HashingSurrogate(2048, 1), 256, 0.1, 1),
    'smbo_mcts_512' : lambda ssf: se.SMBOSearcherWithMCTSOptimizer(ssf, su.HashingSurrogate(2048, 1), 512, 0.1, 1)
}