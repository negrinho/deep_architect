

import argparse
from pprint import pprint
import deep_architect.utils as ut

from deep_architect.contrib.misc.datasets.loaders import (
    load_cifar10, load_mnist, load_fashion_mnist)
from deep_architect.contrib.misc.datasets.dataset import InMemoryDataset
import deep_architect.modules as mo
from deep_architect.hyperparameters import D
import deep_architect.contrib.deep_learning_backend.tfe_ops as tfe_ops
from deep_architect.searchers import common as se
from deep_architect import search_logging as sl
from deep_architect.contrib.misc.evaluators.tensorflow.estimator_classification import AdvanceClassifierEvaluator

from search_space_factory import name_to_search_space_factory_fn
import os

def search_space():
    return mo.siso_sequential([
        mo.empty(),
        tfe_ops.conv2d(D([36]), D([3]), D([1]), D([1]), D([True])),
        tfe_ops.batch_normalization(),
        tfe_ops.relu(),
        tfe_ops.conv2d(D([36]), D([3]), D([1]), D([1]), D([True])),
        tfe_ops.dropout(D([.6])),
        tfe_ops.global_pool2d(),
        tfe_ops.fc_layer(D([10]))
    ])
def main():
    scratchdir = os.environ['SCRATCH']
    config = ut.read_jsonfile(ut.join_paths([scratchdir, 'bench/logs/bench_random_zoph_sp1/evaluations/x7/config.json']))
    # config = ut.read_jsonfile('./logs/config.json')
    vs = config['hyperp_value_lst']
    datasets = {
        'cifar10': lambda: (load_cifar10('data/cifar10/'), 10),
        'mnist': lambda: (load_mnist('data/mnist/'), 10),
        'fashion_mnist': lambda: (load_fashion_mnist(), 10)
    }
    scratch_folder = './scratch_data'
    ut.create_folder(scratch_folder, abort_if_exists=False)
    (Xtrain, ytrain, Xval, yval, Xtest, ytest), num_classes = datasets['cifar10']()
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)
    search_space_factory = name_to_search_space_factory_fn['zoph_sp1'](num_classes)

    evaluator = AdvanceClassifierEvaluator(train_dataset, val_dataset, num_classes,
                        max_num_training_epochs=600, stop_patience=25, whiten=True,
                        test_dataset=test_dataset, base_dir=scratch_folder)

    inputs, outputs, hs = search_space_factory.get_search_space()
    se.specify(outputs.values(), hs, vs)
    # inputs, outputs = search_space()
    # se.random_specify(outputs.values())
    results = evaluator.eval(inputs, outputs)
    pprint(results)

if __name__ == '__main__':
    main()
