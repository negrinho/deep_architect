from time import sleep
import argparse
import random
import os
import tensorflow as tf
from darch.contrib.useful.datasets.loaders import load_cifar10
from darch.contrib.useful.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from darch.contrib.useful.datasets.dataset import InMemoryDataset
from darch.contrib.useful import gpu_utils
from darch import searchers as se
import darch.search_logging as sl

from file_utils import consume_file, write_file
from search_spaces.search_space_factory import name_to_search_space_factory_fn
from searchers.searcher import name_to_searcher_fn
from evaluators.enas_evaluator import ENASEvaluator
from evaluators.enas_evaluator_eager import ENASEagerEvaluator

def start_worker(rank, evaluator, search_space_factory, 
    worker_queue_file='worker_queue', worker_results_prefix='worker_results_',):
    # set the available gpu for process
    print 'WORKER %d' % rank
    if len(gpu_utils.get_gpu_information()) != 0:
        #https://github.com/tensorflow/tensorflow/issues/1888
        gpu_utils.set_visible_gpus([rank % gpu_utils.get_total_num_gpus()])

    while(True):
        file_data = consume_file(worker_queue_file)
        if file_data is None:
            continue
        elif file_data is 'kill':
            write_file(worker_results_prefix + str(rank), 'done')
            break
        vs, evaluation_id, searcher_eval_token, kill = file_data

        inputs, outputs, hs = search_space_factory.get_search_space()
        se.specify(outputs.values(), hs, vs)
        # results = evaluator.eval(inputs, outputs, hs)
        results = {'validation_accuracy': random.random()}
        write_file(
            worker_results_prefix + str(rank), 
            (results, evaluation_id, searcher_eval_token))

def main():
    configs = sl.read_jsonfile("./dev/architecture_search_benchmarks/experiment_config.json")

    parser = argparse.ArgumentParser("Architecture search job, communication based on filesystem")
    parser.add_argument('--config', '-c', action='store', dest='config_name',
    default='normal')
    parser.add_argument('--eager', '-e', action='store_true', dest='eager', default=False)
    parser.add_argument('--rank', '-r', type=int, action='store', dest='rank', required=True)

    options = parser.parse_args()
    config = configs[options.config_name]

    if options.eager:
        tf.enable_eager_execution()

    dirname = './comm_dir'
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    
    datasets = {
        'cifar10': lambda: (load_cifar10('data/cifar10/cifar-10-batches-py/'), 10)
    }

    (Xtrain, ytrain, Xval, yval, Xtest, ytest), num_classes = datasets[config['dataset']]()
    search_space_factory = name_to_search_space_factory_fn[config['search_space']](num_classes)

    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)

    evaluators = {
        'simple_classification': lambda: SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes,
                    './temp' + str(rank), max_num_training_epochs=config['epochs'], log_output_to_terminal=options.display_output,
                    test_dataset=test_dataset),
        'enas_evaluator': lambda: ENASEvaluator(train_dataset, val_dataset, num_classes,
                    search_space_factory.weight_sharer),
        'enas_eager_evaluator': lambda: ENASEagerEvaluator(train_dataset, val_dataset, num_classes,
                    search_space_factory.weight_sharer)
    }

    assert not config['evaluator'].startswith('enas') or hasattr(search_space_factory, 'weight_sharer')
    evaluator = evaluators[config['evaluator']]()


    start_worker(options.rank, evaluator, search_space_factory, 
        worker_queue_file=dirname + '/worker_queue', 
        worker_results_prefix=dirname + '/worker_results_')

if __name__ == "__main__":
    main()