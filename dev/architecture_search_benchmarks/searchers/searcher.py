from .regularized_evolution_searcher import EvolutionSearcher, mutatable
from enas.enas_searcher import ENASSearcher
from enas.enas_searcher_eager import ENASEagerSearcher
import darch.searchers as se
import darch.surrogates as su

name_to_searcher_fn = {
    'random': lambda ssf: se.RandomSearcher(ssf),
    'evolution_pop=100_samp=25_reg=t': lambda ssf: EvolutionSearcher(ssf, mutatable, 100, 25, regularized=True),
    'evolution_pop=64_samp=16_reg=t': lambda ssf: EvolutionSearcher(ssf, mutatable, 64, 16, regularized=True),
    'evolution_pop=20_samp=20_reg=t': lambda ssf: EvolutionSearcher(ssf, mutatable, 20, 20, regularized=True),
    'evolution_pop=100_samp=50_reg=t': lambda ssf: EvolutionSearcher(ssf, mutatable, 100, 50, regularized=True),
    'evolution_pop=100_samp=2_reg=t': lambda ssf: EvolutionSearcher(ssf, mutatable, 100, 2, regularized=True),
    'evolution_pop=100_samp=25_reg=f': lambda ssf: EvolutionSearcher(ssf, mutatable, 100, 25, regularized=False),
    'evolution_pop=64_samp=16_reg=f': lambda ssf: EvolutionSearcher(ssf, mutatable, 64, 16, regularized=False),
    'evolution_pop=20_samp=20_reg=f': lambda ssf: EvolutionSearcher(ssf, mutatable, 20, 20, regularized=False),
    'evolution_pop=100_samp=50_reg=f': lambda ssf: EvolutionSearcher(ssf, mutatable, 100, 50, regularized=False),
    'evolution_pop=100_samp=2_reg=f': lambda ssf: EvolutionSearcher(ssf, mutatable, 100, 2, regularized=False),
    'smbo_optimizer=rand_samples=256' : lambda ssf: se.SMBOSearcher(ssf, su.HashingSurrogate(2048, 1), 256, 0.1),
    'smbo_optimizer=rand_samples=512' : lambda ssf: se.SMBOSearcher(ssf, su.HashingSurrogate(2048, 1), 512, 0.1),
    'smbo_optimizer=mcts_samples=256' : lambda ssf: se.SMBOSearcherWithMCTSOptimizer(ssf, su.HashingSurrogate(2048, 1), 256, 0.1, 1),
    'smbo_optimizer=mcts_samples=512' : lambda ssf: se.SMBOSearcherWithMCTSOptimizer(ssf, su.HashingSurrogate(2048, 1), 512, 0.1, 1),
    'enas_searcher': lambda ssf: ENASSearcher(ssf),
    'enas_searcher_eager': lambda ssf: ENASEagerSearcher(ssf)
}