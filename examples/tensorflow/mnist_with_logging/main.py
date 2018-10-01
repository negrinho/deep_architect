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
    search_space_factory = mo.SearchSpaceFactory(search_space_fn)
    # Creating the searcher.
    searcher = RandomSearcher(search_space_factory.get_search_space)
    # Search loop.
    for _ in xrange(num_samples - search_logger.get_current_evaluation_id()):
        inputs, outputs, hyperp_value_lst, searcher_eval_token = searcher.sample(
        )
        results = evaluator.eval(inputs, outputs)
        # Logging results (including graph).
        evaluation_logger = search_logger.get_current_evaluation_logger()
        evaluation_logger.log_config(hyperp_value_lst, searcher_eval_token)
        evaluation_logger.log_features(inputs, outputs)
        evaluation_logger.log_results(results)
        vi.draw_graph(
            outputs.values(),
            True,
            True,
            print_to_screen=False,
            out_folderpath=evaluation_logger.get_user_data_folderpath())
        # Updating the searcher given the results of logging.
        searcher.update(results['validation_accuracy'], searcher_eval_token)


if __name__ == '__main__':
    main()