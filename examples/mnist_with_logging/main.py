import deep_architect.utils as ut

# Make sure that only one GPU is visible.
if __name__ == '__main__':
    cfg = ut.get_config()
    if cfg['use_gpu']:
        import deep_architect.contrib.misc.gpu_utils as gpu_utils
        gpu_id = gpu_utils.get_available_gpu(0.1, 5.0)
        print("Using GPU %d" % gpu_id)
        assert gpu_id is not None
        gpu_utils.set_visible_gpus([gpu_id])

from deep_architect.contrib.misc.datasets.loaders import load_mnist
from deep_architect.contrib.misc.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from deep_architect.contrib.misc.datasets.dataset import InMemoryDataset
import deep_architect.contrib.misc.search_spaces.tensorflow.dnn as css_dnn
import deep_architect.search_logging as sl
import deep_architect.visualization as vi
import deep_architect.modules as mo
import deep_architect.utils as ut
from deep_architect.searchers.random import RandomSearcher


def main():
    # Loading the config file.
    cfg = ut.get_config()
    num_classes = 10
    num_samples = cfg['num_samples']
    # Loading the data.
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = load_mnist('data/mnist')
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)
    # Creating up the evaluator.
    evaluator = SimpleClassifierEvaluator(
        train_dataset,
        val_dataset,
        num_classes,
        './temp',
        max_eval_time_in_minutes=cfg['max_eval_time_in_minutes'],
        log_output_to_terminal=True,
        test_dataset=test_dataset)
    # Creating the search space.
    search_space_fn = lambda: css_dnn.dnn_net(num_classes)

    sl.create_search_folderpath(
        cfg["folderpath"],
        cfg["search_name"],
        delete_if_exists=cfg['delete_if_exists'],
        abort_if_exists=False,
        create_parent_folders=True)

    # Creating the searcher.
    searcher = RandomSearcher(search_space_fn)
    # Search loop.
    for evaluation_id in range(num_samples):
        eval_logger = sl.EvaluationLogger(cfg["folderpath"], cfg["search_name"],
                                          evaluation_id)
        if not eval_logger.config_exists():

            inputs, outputs, hyperp_value_lst, eval_token = searcher.sample()
            results = evaluator.eval(inputs, outputs)
            # Logging results (including graph).
            eval_logger.log_config(hyperp_value_lst, eval_token)
            eval_logger.log_results(results)
            vi.draw_graph(
                outputs,
                draw_module_hyperparameter_info=True,
                print_to_screen=False,
                out_folderpath=eval_logger.get_evaluation_data_folderpath())
            # Updating the searcher given the results of logging.
            searcher.update(results['validation_accuracy'], eval_token)


if __name__ == '__main__':
    main()