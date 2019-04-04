from __future__ import absolute_import
from dev.regularized_evolution.searcher.regularized_evolution_searcher import EvolutionSearcher, mutatable
# from dev.enas.searcher.enas_searcher import ENASSearcher
import deep_architect.searchers.random as ra
from deep_architect.searchers.smbo_mcts import SMBOSearcherWithMCTSOptimizer
from deep_architect.searchers.smbo_random import SMBOSearcher
from deep_architect.surrogates.hashing import HashingSurrogate
import deep_architect.surrogates as su

name_to_searcher_fn = {
    'random':
    lambda ssf: ra.RandomSearcher(ssf),
    'evolution_pop=100_samp=25_reg=t':
    lambda ssf: EvolutionSearcher(ssf, mutatable, 100, 25, regularized=True),
    'evolution_pop=64_samp=16_reg=t':
    lambda ssf: EvolutionSearcher(ssf, mutatable, 64, 16, regularized=True),
    'evolution_pop=20_samp=20_reg=t':
    lambda ssf: EvolutionSearcher(ssf, mutatable, 20, 20, regularized=True),
    'evolution_pop=100_samp=50_reg=t':
    lambda ssf: EvolutionSearcher(ssf, mutatable, 100, 50, regularized=True),
    'evolution_pop=100_samp=2_reg=t':
    lambda ssf: EvolutionSearcher(ssf, mutatable, 100, 2, regularized=True),
    'evolution_pop=100_samp=25_reg=f':
    lambda ssf: EvolutionSearcher(ssf, mutatable, 100, 25, regularized=False),
    'evolution_pop=64_samp=16_reg=f':
    lambda ssf: EvolutionSearcher(ssf, mutatable, 64, 16, regularized=False),
    'evolution_pop=20_samp=20_reg=f':
    lambda ssf: EvolutionSearcher(ssf, mutatable, 20, 20, regularized=False),
    'evolution_pop=100_samp=50_reg=f':
    lambda ssf: EvolutionSearcher(ssf, mutatable, 100, 50, regularized=False),
    'evolution_pop=100_samp=2_reg=f':
    lambda ssf: EvolutionSearcher(ssf, mutatable, 100, 2, regularized=False),
    'smbo_optimizer=rand_samples=256':
    lambda ssf: SMBOSearcher(ssf, HashingSurrogate(2048, 1), 256, 0.1),
    'smbo_optimizer=rand_samples=512':
    lambda ssf: SMBOSearcher(ssf, HashingSurrogate(2048, 1), 512, 0.1),
    'smbo_optimizer=mcts_samples=256':
    lambda ssf: SMBOSearcherWithMCTSOptimizer(ssf, HashingSurrogate(2048, 1),
                                              256, 0.1, 1),
    'smbo_optimizer=mcts_samples=512':
    lambda ssf: SMBOSearcherWithMCTSOptimizer(ssf, HashingSurrogate(2048, 1),
                                              512, 0.1, 1),
    # 'enas_searcher':
    # lambda ssf: ENASSearcher(ssf)
}