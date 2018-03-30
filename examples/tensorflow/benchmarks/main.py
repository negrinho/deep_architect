# Run configs to make it easier to run the code.
name_to_cfg = {
    'local_debug' : {
        'max_eval_time_in_minutes' : [0.1],
        'num_samples' : 3,
        'num_repetitions' : 3,
        'search_name' : 'benchmark_debug',
        'use_gpu' : False,
        'delete_if_exists' : True
    },
    'server_debug' : {
        'max_eval_time_in_minutes' : [0.5, 2.0],
        'num_samples' : 3,
        'num_repetitions' : 3,
        'search_name' : 'benchmark_debug',
        'use_gpu' : True,
        'delete_if_exists' : True
    },
    'benchmark_full' : {
        'max_eval_time_in_minutes' : [0.1, 1.0],
        'num_samples' : 256,
        'num_repetitions' : 3,
        'search_name' : 'benchmark',
        'use_gpu' : True,
        'delete_if_exists' : False
    },
}
cfg_name = 'server_debug'
cfg = name_to_cfg[cfg_name]

# Make sure that only one GPU is visible.
use_gpu = cfg['use_gpu']
if use_gpu:
    import darch.contrib.gpu_utils as gpu_utils
    gpu_id = gpu_utils.get_available_gpu(0.1, 5.0)
    print "Using GPU %d" % gpu_id
    assert gpu_id is not None
    gpu_utils.set_visible_gpus([gpu_id])
    # assert gpu_utils.get_total_num_gpus() == 1

from darch.contrib.datasets.loaders import load_mnist
from darch.contrib.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from darch.contrib.datasets.dataset import InMemoryDataset
import darch.contrib.search_spaces.tensorflow.dnn as css_dnn
import darch.contrib.search_spaces.tensorflow.cnn2d as css_cnn2d
import darch.searchers as se
import darch.surrogates as su
import darch.search_logging as sl
import darch.visualization as vi
from darch.contrib.search_spaces.tensorflow.common import D
import darch.modules as mo
from six import iteritems

def get_hyperps():
    return {
        'learning_rate_init' : D([1e-2, 1e-3, 1e-4, 1e-5]),
        'optimizer_type' : D(['adam', 'sgd_mom'])}

class SSF0(mo.SearchSpaceFactory):
    def __init__(self, num_classes):
        mo.SearchSpaceFactory.__init__(self)
        self.num_classes = num_classes

    def _get_search_space(self):
        inputs, outputs = css_dnn.dnn_net(self.num_classes)
        return inputs, outputs, get_hyperps()

class SSF1(mo.SearchSpaceFactory):
    def __init__(self, num_classes):
        mo.SearchSpaceFactory.__init__(self)
        self.num_classes = num_classes

    def _get_search_space(self):
        h_num_spatial_reductions = D([2, 3, 4])
        h_pool_op = D(['max', 'avg'])
        inputs, outputs = mo.siso_sequential([
            css_cnn2d.conv_net(h_num_spatial_reductions),
            css_cnn2d.spatial_squeeze(h_pool_op, D([self.num_classes]))])
        return inputs, outputs, get_hyperps()

def run_searcher(searcher, evaluator, num_samples, search_logger):
    for _ in xrange(num_samples):
        evaluation_logger = search_logger.get_current_evaluation_logger()
        inputs, outputs, hs, hyperp_value_lst, searcher_eval_token = searcher.sample()
        evaluation_logger.log_config(hyperp_value_lst, searcher_eval_token)
        evaluation_logger.log_features(inputs, outputs, hs)
        results = evaluator.eval(inputs, outputs, hs)
        evaluation_logger.log_results(results)
        vi.draw_graph(outputs.values(), True, True, print_to_screen=False,
            out_folderpath=evaluation_logger.get_user_data_folderpath())
        searcher.update(results['validation_accuracy'], searcher_eval_token)

# NOTE: This is wrong because of the creation of the models.
def main():
    num_classes = 10
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = load_mnist('data/mnist')
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)

    name_to_search_space_fn = {
        'dnn' : SSF0(num_classes).get_search_space,
        'conv' : SSF1(num_classes).get_search_space,
    }

    name_to_get_searcher_fn = {
        'random' : lambda ssf: se.RandomSearcher(ssf),
        'smbo_rand_256' : lambda ssf: se.SMBOSearcher(ssf, su.HashingSurrogate(2048, 1), 256, 0.1),
        'smbo_rand_512' : lambda ssf: se.SMBOSearcher(ssf, su.HashingSurrogate(2048, 1), 512, 0.1),
        'smbo_mcts_256' : lambda ssf: se.SMBOSearcherWithMCTSOptimizer(ssf, su.HashingSurrogate(2048, 1), 256, 0.1, 1),
        'smbo_mcts_512' : lambda ssf: se.SMBOSearcherWithMCTSOptimizer(ssf, su.HashingSurrogate(2048, 1), 512, 0.1, 1)
    }
    ### this may not be necessary
    name_to_get_evaluator_fn = {
        'simple' : lambda max_eval_time, model_path: SimpleClassifierEvaluator(
                train_dataset, val_dataset, num_classes,
                model_path, max_eval_time_in_minutes=max_eval_time, # change path
                log_output_to_terminal=True, test_dataset=test_dataset)
    }

    # basic benchmark based on MNIST.
    searcher_name_lst = ['random', 'smbo_rand_256', 'smbo_mcts_256',
        # 'smbo_rand_512', 'smbo_mcts_512'
        ]
    search_space_name_lst = ['dnn',
    # 'conv'
    ]
    for rep_i in xrange(cfg['num_repetitions']):
        for search_space_name in search_space_name_lst:
            for searcher_name in searcher_name_lst:
                for max_eval_time in cfg['max_eval_time_in_minutes']:
                    evaluator_name = "ev%0.2f" % max_eval_time
                    folderpath = sl.join_paths(['logs', cfg['search_name'],
                        search_space_name, searcher_name, evaluator_name])

                    search_logger = sl.SearchLogger(folderpath, 'rep%d' % rep_i,
                        create_parent_folders=True, delete_if_exists=cfg['delete_if_exists'])

                    search_space_fn = name_to_search_space_fn[search_space_name]
                    searcher = name_to_get_searcher_fn[searcher_name](search_space_fn)
                    evaluator = name_to_get_evaluator_fn['simple'](
                        max_eval_time, search_logger.get_search_data_folderpath() + '/current_model')

                    run_searcher(searcher, evaluator, cfg['num_samples'], search_logger)

if __name__ == '__main__':
    main()