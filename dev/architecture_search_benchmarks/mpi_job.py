from time import sleep
import argparse
from mpi4py import MPI
import tensorflow as tf
from deep_architect.contrib.useful.datasets.loaders import load_cifar10
from deep_architect.contrib.useful.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from deep_architect.contrib.useful.datasets.dataset import InMemoryDataset
from deep_architect.contrib.useful import gpu_utils
from deep_architect import searchers as se
import deep_architect.search_logging as sl

from search_spaces.search_space_factory import name_to_search_space_factory_fn
from searchers.searcher import name_to_searcher_fn
from evaluators.enas_evaluator import ENASEvaluator
from evaluators.enas_evaluator_eager import ENASEagerEvaluator

READY_REQ = 0
MODEL_REQ = 1
RESULTS_REQ = 2

def start_searcher(comm, num_workers, searcher, resume_if_exists,
    searcher_load_path, num_samples = -1, num_epochs= - 1, save_every=1):
    assert num_samples != -1 or num_epochs != -1

    print('SEARCHER')
    search_logger = sl.SearchLogger('./logs', 'test',
        resume_if_exists=resume_if_exists, delete_if_exists=not resume_if_exists)

    if resume_if_exists:
        searcher.load(search_logger.search_data_folderpath, searcher_load_path)

    ready_requests = [comm.irecv(source=i+1, tag=READY_REQ) for i in range(num_workers)]
    eval_requests = [comm.irecv(source=i+1, tag=RESULTS_REQ) for i in range(num_workers)]
    eval_loggers = [None] * num_workers
    models_evaluated = search_logger.current_evaluation_id
    epochs = 0
    finished = [False] * num_workers

    while(not all(finished)):
        # See which workers are ready for a new model
        for idx, req in enumerate(ready_requests):
            test, msg = req.test()
            if test:
                cont = num_samples == -1 or models_evaluated < num_samples
                cont = cont and (num_epochs == -1 or epochs < num_epochs)
                if cont:
                    eval_loggers[idx] = search_logger.get_current_evaluation_logger()
                    inputs, outputs, hs, vs, searcher_eval_token = searcher.sample()

                    eval_loggers[idx].log_config(vs, searcher_eval_token)
                    eval_loggers[idx].log_features(inputs, outputs, hs)

                    comm.isend(
                        (vs, search_logger.current_evaluation_id, searcher_eval_token, 
                            search_logger.search_data_folderpath, False),
                        dest=idx + 1, tag=MODEL_REQ)
                    ready_requests[idx] = comm.irecv(source=idx + 1, tag=READY_REQ)

                    models_evaluated += 1
                    if models_evaluated % save_every == 0:
                        searcher.save_state(search_logger.search_data_folderpath)
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

                if 'epoch' in results:
                    epochs = max(epochs, results['epoch'])

                searcher.update(results['validation_accuracy'], searcher_eval_token)
                eval_requests[idx] = comm.irecv(source=idx + 1, tag=RESULTS_REQ)


def start_worker(comm, rank, evaluator, search_space_factory, resume=True, save_every=1):
    # set the available gpu for process
    print('WORKER %d' % rank)
    step = 0
    if len(gpu_utils.get_gpu_information()) != 0:
        #https://github.com/tensorflow/tensorflow/issues/1888
        gpu_utils.set_visible_gpus([rank % gpu_utils.get_total_num_gpus()])


    while(True):
        comm.ssend([rank], dest=0, tag=READY_REQ)
        (vs, evaluation_id, searcher_eval_token, 
            search_data_folderpath, kill) = comm.recv(source=0, tag=MODEL_REQ)
        if kill:
            break

        if resume:
            evaluator.load_state(search_data_folderpath)
            resume = False

        inputs, outputs, hs = search_space_factory.get_search_space()
        se.specify(outputs.values(), hs, vs)
        results = evaluator.eval(inputs, outputs, hs)
        step += 1
        if step % save_every == 0:
            evaluator.save_state(search_data_folderpath)
        comm.ssend((results, evaluation_id, searcher_eval_token), dest=0, tag=RESULTS_REQ)

def main():
    configs = sl.read_jsonfile("./dev/architecture_search_benchmarks/experiment_config.json")

    parser = argparse.ArgumentParser("MPI Job for architecture search")
    parser.add_argument('--config', '-c', action='store', dest='config_name',
    default='normal')

    # Other arguments
    parser.add_argument('--display-output', '-o', action='store_true', dest='display_output',
                        default=False)
    parser.add_argument('--resume', '-r', action='store_true', dest='resume',
                        default=False)
    parser.add_argument('--load_searcher', '-l', action='store', dest='searcher_file_name',
                        default='searcher_state.json')
    parser.add_argument('--eager', '-e', action='store_true', dest='eager', default=False)

    options = parser.parse_args()
    config = configs[options.config_name]

    if options.eager:
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth=True
        tf.enable_eager_execution(tfconfig, device_policy=tf.contrib.eager.DEVICE_PLACEMENT_SILENT)

    datasets = {
        'cifar10': lambda: (load_cifar10('data/cifar10/cifar-10-batches-py/'), 10)
    }

    (Xtrain, ytrain, Xval, yval, Xtest, ytest), num_classes = datasets[config['dataset']]()
    search_space_factory = name_to_search_space_factory_fn[config['search_space']](num_classes)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    save_every = 1 if 'save_every' not in config else config['save_every']
    if rank == 0:
        searcher = name_to_searcher_fn[config['searcher']](search_space_factory.get_search_space)
        num_samples = -1 if 'samples' not in config else config['samples']
        num_epochs = -1 if 'epochs' not in config else config['epochs']
        start_searcher(
            comm, comm.Get_size() - 1, searcher, options.resume, 
            config['searcher_file_name'], num_samples=num_samples, 
            num_epochs=num_epochs, save_every=save_every)
    else:
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

        start_worker(comm, rank, evaluator, search_space_factory, resume=options.resume, save_every=save_every)

if __name__ == "__main__":
    main()