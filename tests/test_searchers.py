from __future__ import print_function
from six.moves import xrange
from six import itervalues, iteritems

import darch.hyperparameters as hp
import darch.searchers as se
import darch.modules as mo

import tests.search_space as ss
import tests.dataset as ds
import tests.evaluator as ev

import pickle


D = hp.Discrete

# should eventually choose the best configuration.
def test_mcts():
    def search_space_fn():
        m = mo.Empty()
        
        k = 20
        n = 20
        hs = { 'H' + str(i) : D( range(k) ) for i in xrange(n) }

        return m.inputs, m.outputs, hs

    mcts = se.MCTSearcher(search_space_fn, .01)

    for _ in xrange(4 * 1024):
        (inputs, outputs, hs, _, cfg_d) = mcts.sample()
        print([v.val for v in itervalues(hs)])
        v = sum([v.val for v in itervalues(hs)])
        mcts.update(v, cfg_d)
    print(v)

# NOTE: time, and benchmark of different aspects.
def test_benchmark():
    num_evals = 128

    # loading the dataset and evaluator
    (train_dataset, val_dataset, test_dataset, in_d, num_classes) = ds.load_data(
        'mnist')

    evaluator = ev.ClassifierEvaluator(train_dataset, val_dataset, in_d, num_classes, 
        '.', 'mnist', 
        output_to_terminal=True, test_dataset=test_dataset,
        max_minutes_per_model=0.01, batch_size_init=128)

    rand_se = se.RandomSearcher(ss.search_space_fn)
    mcts_se = se.MCTSearcher(ss.search_space_fn)
    # simple surrogate model.
    num_samples_surr = 128
    eps_prob = 0.1
    surr_model = se.HashingSurrogate(1024, 1)
    smbo_se = se.SMBOSearcher(ss.search_space_fn, surr_model, 
        num_samples_surr, eps_prob)
    
    tree_refit_interval = 1
    surr_model = se.HashingSurrogate(1024, 1)
    smbo_mcts_se = se.SMBOSearcherWithMCTSOptimizer(ss.search_space_fn, 
        surr_model, num_samples_surr, eps_prob, tree_refit_interval)

    d_searcher = {
        # 'rand' : rand_se, 
        # 'mcts' : mcts_se, 
        'smbo' : smbo_se, 
        'smbo_mcts' : smbo_mcts_se
        }
    
    d_res = {}
    d_hist = {}
    for k, searcher in iteritems(d_searcher):
        vs_lst = []
        vs_hist_lst = []
        for _ in xrange(num_evals):
            (inputs, outputs, hs, vs_hist, cfg_d) = searcher.sample()
            v = evaluator.eval(inputs, outputs, hs)
            searcher.update(v, cfg_d)
     
            vs_lst.append( v )
            vs_hist_lst.append( vs_hist )
            
        d_res[k] = vs_lst
        d_hist[k] = vs_hist_lst
    
        with open('benchmark.pkl', 'wb') as f:
            pickle.dump({
                'searchers' : d_searcher, 
                'results' : d_res,
                'hist' : d_hist }, f)

if __name__ == '__main__':
    # test_mcts()
    test_benchmark()
