"""
MPI Run script
"""
from time import sleep
import argparse
from mpi4py import MPI
from darch.contrib.datasets.loaders import load_cifar10
from darch.contrib.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from darch.contrib.datasets.dataset import InMemoryDataset
from darch.contrib.search_spaces.tensorflow.search_space_factory import SearchSpaceFactory
#import darch.contrib.search_spaces.tensorflow.dnn as css_dnn
import darch.contrib.searchers.searchers as se
import darch.contrib.searchers.searcher_utils as sut
from darch.contrib import gpu_utils
import darch.search_logging as sl

READY_REQ = 0
MODEL_REQ = 1
RESULTS_REQ = 2

def start_searcher(comm, num_workers, num_samples, searcher, resume_if_exists, 
    searcher_load_path):
    search_logger = sl.SearchLogger('./logs', 'test', resume_if_exists=resume_if_exists, delete_if_exists=not resume_if_exists)
    search_data_path = sl.join_paths([search_logger.search_data_folderpath, searcher_load_path])
    
    if sl.file_exists(search_data_path):
        state = sl.read_jsonfile(search_data_path)
        searcher.load(state)

    ready_requests = [comm.irecv(source=i+1, tag=READY_REQ) for i in range(num_workers)]
    eval_requests = [comm.irecv(source=i+1, tag=RESULTS_REQ) for i in range(num_workers)]
    eval_loggers = [None] * num_workers
    models_evaluated = search_logger.current_evaluation_id
    finished = [False] * num_workers

    while(not all(finished)):

        # See which workers are ready for a new model
        for idx, req in enumerate(ready_requests):
            test, msg = req.test()
            if test:
                if models_evaluated < num_samples:
                    eval_loggers[idx] = search_logger.get_current_evaluation_logger()
                    inputs, outputs, hs, vs, searcher_eval_token = searcher.sample()
                    
                    eval_loggers[idx].log_config(vs, searcher_eval_token)
                    eval_loggers[idx].log_features(inputs, outputs, hs)
                    
                    comm.isend((vs, search_logger.current_evaluation_id, searcher_eval_token, False), dest=idx + 1, tag=MODEL_REQ)
                    ready_requests[idx] = comm.irecv(source=idx + 1, tag=READY_REQ)
                    
                    models_evaluated += 1
                elif not finished[idx]:
                    comm.isend((None, None, None, True), 
                                dest=idx + 1, tag=MODEL_REQ)
                    finished[idx] = True
        
        # See which workers have finished evaluation
        for idx, req in enumerate(eval_requests):
            test, msg = req.test()
            if test:
                results, model_id, searcher_eval_token = msg
                evaluation_logger = eval_loggers[idx]
                evaluation_logger.log_results(results)      
                print('Sample %d: %f' % (model_id, results['validation_accuracy']))
                searcher.update(results['validation_accuracy'], searcher_eval_token)
                searcher.save_state(search_logger.search_data_folderpath)
                eval_requests[idx] = comm.irecv(source=idx + 1, tag=RESULTS_REQ)
        sleep(1)


def start_worker(comm, rank, evaluator, search_space_factory):
    # set the available gpu for process
    if len(gpu_utils.get_gpu_information()) != 0:
        gpu_utils.set_visible_gpus([rank % gpu_utils.get_total_num_gpus()])

    while(True):
        comm.ssend([rank], dest=0, tag=READY_REQ)
        vs, evaluation_id, searcher_eval_token, kill = comm.recv(source=0, tag=MODEL_REQ)
        if kill:
            break
        
        inputs, outputs, hs = search_space_factory.get_search_space()
        sut.specify(outputs.values(), vs)
        
        results = evaluator.eval(inputs, outputs, hs)
        comm.ssend((results, evaluation_id, searcher_eval_token), dest=0, tag=RESULTS_REQ)

def main():
    configs = sl.read_jsonfile("experiment_config.json")
    
    parser = argparse.ArgumentParser("MPI Job for architecture search")
    parser.add_argument('--config', '-c', action='store', dest='config_name', 
    default='evolution', choices=['evolution'])

    # Other arguments
    parser.add_argument('--display-output', '-o', action='store_true', dest='display_output', 
                        default=False)
    parser.add_argument('--resume', '-r', action='store_true', dest='resume', 
                        default=False)
    parser.add_argument('--load_searcher', '-l', action='store', dest='searcher_file_name',
                        default='searcher_state.json')

    options = parser.parse_args()
    config = configs[options.config_name]
        
    datasets = {
        'cifar10': lambda: (load_cifar10('data/cifar10'), 10)
    }

    (Xtrain, ytrain, Xval, yval, Xtest, ytest), num_classes = datasets[config['dataset']]()
    search_space_factory = SearchSpaceFactory(config['search_space'], num_classes)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        if config['searcher'] == 'evolution':
            assert config['P'] >= config['S']
        searchers = {
            'evolution': lambda: se.EvolutionSearcher(search_space_factory.get_search_space, se.mutatable, config['P'], config['S'], regularized=config['regularized'])
        }
        searcher = searchers[config['searcher']]()
        start_searcher(comm, comm.Get_size() - 1, config['samples'], searcher, options.resume, options.searcher_file_name)
    else:

        train_dataset = InMemoryDataset(Xtrain, ytrain, True)
        val_dataset = InMemoryDataset(Xval, yval, False)
        test_dataset = InMemoryDataset(Xtest, ytest, False)

        evaluators = {
            'simple_classification': lambda: SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes,
                        './temp' + str(rank), max_num_training_epochs=config['epochs'], log_output_to_terminal=options.display_output, 
                        test_dataset=test_dataset)
        }
        evaluator = evaluators[config['evaluator']]()

        start_worker(comm, rank, evaluator, search_space_factory)

if __name__ == "__main__":
    main()