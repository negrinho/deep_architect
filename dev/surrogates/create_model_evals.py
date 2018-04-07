from darch import searchers as se, surrogates as su, core as co, search_logging as sl

import darch.contrib.search_spaces.tensorflow.dnn as search_dnn
from darch.contrib.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from darch.contrib.search_spaces.tensorflow.common import D
from darch.contrib.datasets.dataset import InMemoryDataset
from darch.contrib.datasets.loaders import load_mnist
import dev.surrogates.datasets as datasets
import torchvision

import sys

class SearchSpaceFactory:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def get_search_space(self):
        co.Scope.reset_default_scope()
        inputs, outputs = search_dnn.dnn_net(self.num_classes)
        return inputs, outputs, {'learning_rate_init' : D([1e-2, 1e-3, 1e-4, 1e-5])}

# Samples and evaluates a model from a searcher. Logs the result
def sample_and_evaluate(searcher, evaluator, search_logger):
    # Sample from our searcher
    (inputs, outputs, hs, hyperp_value_lst, searcher_eval_token) = searcher.sample()
        # Get the true score by training the model sampled and log them
    results = evaluator.eval(inputs, outputs, hs)
    # Get the logger for this iteration and log configurations, features, and results
    eval_logger = search_logger.get_current_evaluation_logger()
    eval_logger.log_config(hyperp_value_lst, searcher_eval_token)
    eval_logger.log_features(inputs, outputs, hs)
    eval_logger.log_results(results)


def create_eval_dataset(train_dataset, test_dataset, dataset_name, num_classes):
    train_dataset, val_dataset = datasets.train_val_split(train_dataset)
    train_dataset = datasets.TorchInMemoryDataset(train_dataset, True)
    val_dataset = datasets.TorchInMemoryDataset(val_dataset, False)
    test_dataset = datasets.TorchInMemoryDataset(test_dataset, False)

    ## Define our search space
    search_space_fn = SearchSpaceFactory(num_classes).get_search_space
    ## Define our searcher
    searcher_rand = se.RandomSearcher(search_space_fn)
    ## Declare the model evaluator
    evaluator = SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes,
        'temp/', max_eval_time_in_minutes=1.0, log_output_to_terminal=True)

    ## Define our Logger
    search_logger = sl.SearchLogger('./logs', dataset_name, resume_if_exists=True)

    while search_logger.current_evaluation_id < 2048: #TODO: parameterize this
            sample_and_evaluate(searcher_rand, evaluator, search_logger)


config = {
    'WINE': {
        'TRAIN': datasets.WINE('temp/', train=True),
        'TEST': datasets.WINE('temp/', train=False),
        'CLASSES': 3,
    },
    'CANCER': {
        'TRAIN': datasets.CANCER('temp/', train=True),
        'TEST': datasets.CANCER('temp/', train=False),
        'CLASSES': 2,
    },
    'IRIS': {
        'TRAIN': datasets.IRIS('temp/', train=True),
        'TEST': datasets.IRIS('temp/', train=False),
        'CLASSES': 3,
    },
}

# Not Tested Yet: Need to find out if its one-hot encoded already
config_Large = {
    'CIFAR': {
        'TRAIN': torchvision.datasets.CIFAR10('temp/', download=True, train=True, transform=datasets.ToNumpy(), target_transform=datasets.ToOneHot(10)),
        'TEST': torchvision.datasets.CIFAR10('temp/', download=True, train=False, transform=datasets.ToNumpy(), target_transform=datasets.ToOneHot(10)),
        'CLASSES': 10,
    },
    'SVHN': {
        'TRAIN': torchvision.datasets.SVHN('temp/', download=True, split='train', transform=datasets.ToNumpy(), target_transform=datasets.ToOneHot(10)),
        'TEST': torchvision.datasets.SVHN('temp/', download=True, split='test', transform=datasets.ToNumpy(), target_transform=datasets.ToOneHot(10)),
        'CLASSES': 10,
    },
    'MNIST': {
        'TRAIN': torchvision.datasets.MNIST('temp/', download=True, train=True, transform=datasets.ToNumpy(), target_transform=datasets.ToOneHot(10)),
        'TEST': torchvision.datasets.MNIST('temp/', download=True, train=False, transform=datasets.ToNumpy(), target_transform=datasets.ToOneHot(10)),
        'CLASSES': 10,
    },
    'FASHION-MNIST': {
        'TRAIN': datasets.FashionMNIST('temp/', download=True, train=True, transform=datasets.ToNumpy(), target_transform=datasets.ToOneHot(10)),
        'TEST': datasets.FashionMNIST('temp/', download=True, train=False, transform=datasets.ToNumpy(), target_transform=datasets.ToOneHot(10)),
        'CLASSES': 10,
    },
}

def main(args):
    for name, info in config_Large.items():
        create_eval_dataset(info['TRAIN'], info['TEST'], name, info['CLASSES'])

    

if __name__ == '__main__':
    main(sys.argv)