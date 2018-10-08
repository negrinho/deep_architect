import argparse
from pprint import pprint
import deep_architect.utils as ut

from deep_architect.contrib.misc.datasets.loaders import (
    load_cifar10, load_mnist, load_fashion_mnist)
from deep_architect.contrib.misc.datasets.dataset import InMemoryDataset

from deep_architect.searchers import common as se
from deep_architect import search_logging as sl
from deep_architect.contrib.misc.evaluators.tensorflow.classification import SimpleClassifierEvaluator

from search_space_factory import name_to_search_space_factory_fn


def train_best(config, searcher_file):
    folderpath = config['search_folder']
    search_name = config['search_name']
    search_data_folder = sl.get_search_data_folderpath(folderpath, search_name)
    save_filepath = ut.join_paths((search_data_folder, searcher_file))
    searcher_data = ut.read_jsonfile(save_filepath)
    best_models = searcher_data
    datasets = {
        'cifar10': lambda: (load_cifar10('data/cifar10/'), 10),
        'mnist': lambda: (load_mnist('data/mnist/'), 10),
        'fashion_mnist': lambda: (load_fashion_mnist(), 10)
    }

    (Xtrain, ytrain, Xval, yval, Xtest, ytest), num_classes = datasets[config['dataset']]()
    search_space_factory = name_to_search_space_factory_fn[config['search_space']](num_classes)
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)
    evaluator = SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes,
                                            './temp', test_dataset=test_dataset)
    for ix, model in enumerate(best_models):
        inputs, outputs, hs = search_space_factory.get_search_space()
        se.specify(outputs.values(), hs, model[1])
        results = evaluator.eval(inputs, outputs, hs)
        pprint(results)
        eval_logger = sl.EvaluationLogger(folderpath, search_name + '_final', ix)
        eval_logger.log_config(model[1], model[1])
        eval_logger.log_features(inputs, outputs, hs)
        eval_logger.log_results(results)

def main():
    parser = argparse.ArgumentParser("MPI Job for architecture search")
    parser.add_argument('--config', '-c', action='store', dest='config_name',
        default='normal')
    parser.add_argument('--searcher-file', '-s', action='store', dest='searcher_file',
        default='normal')
    options = parser.parse_args()
    configs = ut.read_jsonfile("./examples/tensorflow/full_benchmarks/experiment_config.json")
    config = configs[options.config_name]
    train_best(config, options.searcher_file)

if __name__ == '__main__':
    main()
