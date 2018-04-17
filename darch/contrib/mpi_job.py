"""
MPI Run script
"""
from time import sleep
from mpi4py import MPI
from darch.contrib.datasets.loaders import load_cifar10
from darch.contrib.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from darch.contrib.datasets.dataset import InMemoryDataset
#import darch.contrib.search_spaces.tensorflow.dnn as css_dnn
import darch.searchers as se
import darch.contrib.search_spaces.tensorflow.evolution_search_space as ss
from darch.contrib import gpu_utils
import darch.search_logging as sl
import darch.core as co
import random
READY_REQ = 0
MODEL_REQ = 1
RESULTS_REQ = 2

class SearchSpaceFactory:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def get_search_space(self):
        co.Scope.reset_default_scope()
        inputs, outputs = ss.get_search_space_1(self.num_classes)
        return inputs, outputs, {}

def mutatable(h):
    return h.get_name().startswith('H.Mutatable')

def specify(output_lst, vs):
    try:
        for i, h in enumerate(unset_hyperparameter_iterator(output_lst)):
            h.set_val(vs[i])
    except AssertionError as a:
        print h.vs
        print vs[i]
    

# should be moved to utils file
def unset_hyperparameter_iterator(output_lst, hyperp_lst=None):
    if hyperp_lst is not None:
        for h in hyperp_lst:
            if not h.is_set():
                yield h

    while not co.is_specified(output_lst):
        hs = co.get_unset_hyperparameters(output_lst)
        for h in hs:
            if not h.is_set():
                yield h

def start_searcher(comm, num_workers):
    num_classes = 10
    num_samples = 200
    num_classes = 10
    search_space_factory = SearchSpaceFactory(num_classes)
    searcher = se.EvolutionSearcher(search_space_factory.get_search_space, mutatable, 20, 20, regularized=True)

    search_logger = sl.SearchLogger('./logs', 'test', resume_if_exists=True)
    search_data_path = sl.join_paths([search_logger.search_data_folderpath, "searcher_state.json"])
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


def start_worker(comm, rank):
    # set the available gpu for process
    if len(gpu_utils.get_gpu_information()) != 0:
        gpu_utils.set_visible_gpus([rank % gpu_utils.get_total_num_gpus()])

    # set up evaluator
    num_classes = 10
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = load_cifar10('data/cifar10')
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)
    evaluator = SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes, 
                    './temp' + str(rank), max_num_training_epochs=4, log_output_to_terminal=True, 
                    test_dataset=test_dataset)

    # set up search space factory
    search_space_factory = SearchSpaceFactory(num_classes)

    while(True):
        comm.ssend([rank], dest=0, tag=READY_REQ)
        vs, evaluation_id, searcher_eval_token, kill = comm.recv(source=0, tag=MODEL_REQ)
        if kill:
            break
        
        inputs, outputs, hs = search_space_factory.get_search_space()
        specify(outputs.values(), vs)
        
        results = evaluator.eval(inputs, outputs, hs)
        comm.ssend((results, evaluation_id, searcher_eval_token), dest=0, tag=RESULTS_REQ)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        start_searcher(comm, comm.Get_size() - 1)
    else:
        start_worker(comm, rank)

if __name__ == "__main__":
    main()