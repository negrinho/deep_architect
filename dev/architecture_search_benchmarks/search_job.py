from time import sleep
import argparse
import os
import tensorflow as tf
from darch.contrib.useful.datasets.loaders import load_cifar10
from darch.contrib.useful import gpu_utils
from darch import searchers as se
import darch.search_logging as sl

from file_utils import consume_file, write_file, read_file
from search_spaces.search_space_factory import name_to_search_space_factory_fn
from searchers.searcher import name_to_searcher_fn

def start_searcher(num_workers, searcher, resume_if_exists,
    searcher_load_path, worker_queue_file='worker_queue', 
    worker_results_prefix='worker_results_', num_samples = -1, num_epochs= - 1):
    assert num_samples != -1 or num_epochs != -1

    print 'SEARCHER'
    search_logger = sl.SearchLogger('./logs', 'test',
        resume_if_exists=resume_if_exists, delete_if_exists=not resume_if_exists)
    search_data_path = sl.join_paths([search_logger.search_data_folderpath, searcher_load_path])

    if sl.file_exists(search_data_path):
        state = sl.read_jsonfile(search_data_path)
        searcher.load(state)

    eval_loggers = {}

    models_evaluated = search_logger.current_evaluation_id
    epochs = 0
    finished = 0

    while(finished < num_workers):
        # check if search should continue
        cont = num_samples == -1 or models_evaluated < num_samples
        cont = cont and (num_epochs == -1 or epochs < num_epochs)
        
        # check if model on worker queue has been consumed yet
        file_data = read_file(worker_queue_file)

        if file_data is None:
            if cont:
                # create evaluation logger for model
                evaluation_logger = search_logger.get_current_evaluation_logger()
                print search_logger.current_evaluation_id
                eval_loggers[search_logger.current_evaluation_id] = evaluation_logger
                
                inputs, outputs, hs, vs, searcher_eval_token = searcher.sample()
                
                # log model
                evaluation_logger.log_config(vs, searcher_eval_token)
                evaluation_logger.log_features(inputs, outputs, hs)
                
                # write model description to worker queue
                write_file(worker_queue_file, (vs, search_logger.current_evaluation_id, searcher_eval_token, False))
                models_evaluated += 1
            else:
                write_file(worker_queue_file, 'kill')
        
        for i in range(num_workers):
            # check if worker has finished or has put results on their results queue
            worker_result = consume_file(worker_results_prefix + str(i))
            if worker_result is 'done':
                finished += 1
            elif worker_result is not None:
                results, model_id, searcher_eval_token = worker_result
                evaluation_logger = eval_loggers.pop(model_id)
                evaluation_logger.log_results(results)
                if 'epoch' in results:
                    epochs = max(epochs, results['epoch'])

                searcher.update(results['validation_accuracy'], searcher_eval_token)
                searcher.save_state(search_logger.search_data_folderpath)

def main():
    configs = sl.read_jsonfile("./dev/architecture_search_benchmarks/experiment_config.json")

    parser = argparse.ArgumentParser("Architecture search job, communication based on filesystem")
    parser.add_argument('--config', '-c', action='store', dest='config_name',
    default='normal')

    # Other arguments
    parser.add_argument('--resume', '-r', action='store_true', dest='resume',
                        default=False)
    parser.add_argument('--load_searcher', '-l', action='store', dest='searcher_file_name',
                        default='searcher_state.json')
    parser.add_argument('--eager', '-e', action='store_true', dest='eager', default=False)
    parser.add_argument('--num-workers', '-n', type=int, action='store', dest='num_workers', required=True)

    options = parser.parse_args()
    config = configs[options.config_name]

    if options.eager:
        tf.enable_eager_execution()

    dirname = './comm_dir'
    print 'first'
    if not os.path.isdir(dirname):
        print 'here'
        os.mkdir(dirname)
    
    datasets = {
        'cifar10': lambda: (load_cifar10('data/cifar10/cifar-10-batches-py/'), 10)
    }
    _, num_classes = datasets[config['dataset']]()
    search_space_factory = name_to_search_space_factory_fn[config['search_space']](num_classes)

    searcher = name_to_searcher_fn[config['searcher']](search_space_factory.get_search_space)
    num_samples = -1 if 'samples' not in config else config['samples']
    num_epochs = -1 if 'epochs' not in config else config['epochs']
    start_searcher(
        options.num_workers, searcher, options.resume, 
        config['searcher_file_name'], num_samples=num_samples, 
        num_epochs=num_epochs, worker_queue_file=dirname + '/worker_queue', 
        worker_results_prefix=dirname + '/worker_results_')

if __name__ == "__main__":
    main()