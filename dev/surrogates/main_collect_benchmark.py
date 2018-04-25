
from darch.contrib.useful.datasets.loaders import load_mnist
from darch.contrib.useful.datasets.dataset import InMemoryDataset
from darch.contrib.useful.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from darch.contrib.useful.search_spaces.tensorflow.common import D
import darch.modules as mo
import darch.contrib.useful.search_spaces.tensorflow.dnn as css_dnn
import darch.search_logging as sl
import darch.searchers as se

# TODO: perhaps add more datasets and the dataset type to the config.
name_to_cfg = {
    'small' : {
        'benchmark_size' : 128,
        'max_eval_time_in_minutes' : 0.1
    },
    'full' : {
        'benchmark_size' : 4096,
        'max_eval_time_in_minutes' : 4.0
    }
}
cfg_name = 'small'
cfg = name_to_cfg[cfg_name]

class SSF0(mo.SearchSpaceFactory):
    def __init__(self, num_classes):
        mo.SearchSpaceFactory.__init__(self, True)
        self.num_classes = num_classes

    def _get_search_space(self):
        inputs, outputs = css_dnn.dnn_net(self.num_classes)
        return inputs, outputs, {'learning_rate_init' : D([1e-2, 1e-3, 1e-4, 1e-5])}

def main():
    ## Params:
    dataset_size = cfg['benchmark_size'] # Initial dataset size

    # Steal MNIST Tensorflow example data:
    (X_train, y_train, X_valid, y_valid, X_test, y_test) = load_mnist('.temp/data/mnist')
    # Define datasets for contrib tensorflow evaluator
    train_dataset = InMemoryDataset(X_train, y_train, True)
    val_dataset = InMemoryDataset(X_valid, y_valid, False)
    test_dataset = InMemoryDataset(X_test, y_test, False)

    num_classes = 10 # Change this per dataset

    ## Declare the model evaluator
    evaluator = SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes,
        './temp', max_eval_time_in_minutes=cfg['max_eval_time_in_minutes'],
        log_output_to_terminal=True, test_dataset=test_dataset)

    ## Define our search space
    search_space_fn = SSF0(num_classes).get_search_space

    # We use a random searcher to populate our initial dataset
    searcher = se.RandomSearcher(search_space_fn)

    ## Define our Logger
    search_logger = sl.SearchLogger('./logs',
        'benchmark_surrogates.%s.mnist' % cfg_name)

    # We want to populate our dataset with some initial configurations and evaluations
    if search_logger.current_evaluation_id < dataset_size:
        print('Not enough data found, training models.')
        while search_logger.current_evaluation_id < dataset_size:
            # Sample from our searcher
            (inputs, outputs, hyperps, hyperp_value_lst, searcher_eval_token) = searcher.sample()
                # Get the true score by training the model sampled and log them
            results = evaluator.eval(inputs, outputs, hyperps)
            # Get the logger for this iteration and log configurations, features, and results
            eval_logger = search_logger.get_current_evaluation_logger()
            eval_logger.log_config(hyperp_value_lst, searcher_eval_token)
            eval_logger.log_features(inputs, outputs, hyperps)
            eval_logger.log_results(results)

if __name__ == '__main__':
    main()