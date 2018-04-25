# Run configs to make it easier to run the code.
import darch.search_logging as sl

# Make sure that only one GPU is visible.
if __name__ == '__main__':
    cmd = sl.CommandLineArgs()
    cmd.add('config_filepath', 'str', 'examples/tensorflow/benchmarks/config.json', True)
    cmd.add('key', 'str')
    d_cmd = cmd.parse()
    cfg = sl.read_jsonfile(d_cmd['config_filepath'])[d_cmd['key']]

    if cfg['use_gpu']:
        import darch.contrib.gpu_utils as gpu_utils
        gpu_id = gpu_utils.get_available_gpu(0.1, 5.0)
        print "Using GPU %d" % gpu_id
        assert gpu_id is not None
        gpu_utils.set_visible_gpus([gpu_id])

from darch.contrib.datasets.loaders import load_mnist
from darch.contrib.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from darch.contrib.datasets.dataset import InMemoryDataset
import darch.visualization as vi

import searchers as local_se
import search_spaces as local_ss

def run_searcher(searcher, evaluator, num_samples, search_logger):
    for _ in xrange(num_samples):
        evaluation_logger = search_logger.get_current_evaluation_logger()
        inputs, outputs, hs, hyperp_value_lst, searcher_eval_token = searcher.sample()
        results = evaluator.eval(inputs, outputs, hs)
        evaluation_logger.log_config(hyperp_value_lst, searcher_eval_token)
        evaluation_logger.log_features(inputs, outputs, hs)
        evaluation_logger.log_results(results)
        vi.draw_graph(outputs.values(), True, True, print_to_screen=False,
            out_folderpath=evaluation_logger.get_user_data_folderpath())
        searcher.update(results['validation_accuracy'], searcher_eval_token)

def main():
    num_classes = 10
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = load_mnist('data/mnist')
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)

    evaluator = SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes,
        sl.join_paths(['temp', 'benchmarks', cfg['search_name']]),
        max_eval_time_in_minutes=cfg['max_eval_time_in_minutes'],
        log_output_to_terminal=True, test_dataset=test_dataset)

    for rep_i in xrange(cfg['num_repetitions']):
        for search_space_name in cfg['search_space_name_lst']:
            for searcher_name in cfg['searcher_name_lst']:

                folderpath = sl.join_paths([cfg['logs_folderpath'],
                    cfg['search_name'], search_space_name, searcher_name])

                search_logger = sl.SearchLogger(folderpath, 'rep%d' % rep_i,
                    create_parent_folders=True, delete_if_exists=cfg['delete_if_exists'],
                    resume_if_exists=cfg['resume_if_exists'])

                search_space_fn = local_ss.name_to_search_space_fn[search_space_name](num_classes)
                searcher = local_se.name_to_get_searcher_fn[searcher_name](search_space_fn)
                run_searcher(searcher, evaluator,
                    cfg['num_samples'] - search_logger.get_current_evaluation_id(), search_logger)

if __name__ == '__main__':
    main()