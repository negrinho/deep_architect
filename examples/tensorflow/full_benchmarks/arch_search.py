import argparse
import tensorflow as tf

from deep_architect.contrib.misc.datasets.loaders import load_cifar10, load_mnist
from deep_architect.contrib.misc.datasets.dataset import InMemoryDataset

from deep_architect.searchers import common as se
from deep_architect.contrib.misc import gpu_utils
from deep_architect import search_logging as sl
from deep_architect import utils as ut

from search_space_factory import name_to_search_space_factory_fn
from searcher import name_to_searcher_fn

from deep_architect.contrib.enas.evaluator.enas_evaluator import ENASEvaluator
from deep_architect.contrib.misc.evaluators.tensorflow.classification import SimpleClassifierEvaluator

from deep_architect.communicators.communicator import get_communicator


def start_searcher(comm, searcher, resume_if_exists, folderpath, search_name,
    searcher_load_path, num_samples = -1, num_epochs= - 1, save_every=1):
    assert num_samples != -1 or num_epochs != -1

    print('SEARCHER')

    search_data_folder = sl.get_search_data_folderpath(folderpath, search_name)

    # Load previous searcher
    if resume_if_exists:
        searcher.load(search_data_folder, searcher_load_path)

    models_sampled = 0
    epochs = 0
    finished = 0
    killed = 0

    while(finished < models_sampled or killed < comm.num_workers):
        # See whether workers are ready to consume architectures
            # Search end conditions
        cont = num_samples == -1 or models_sampled < num_samples
        cont = cont and (num_epochs == -1 or epochs < num_epochs)
        if cont:
            if comm.is_ready_to_publish_architecture():
                eval_logger = sl.EvaluationLogger(folderpath, search_name, models_sampled)
                inputs, outputs, hs, vs, searcher_eval_token = searcher.sample()

                eval_logger.log_config(vs, searcher_eval_token)
                eval_logger.log_features(inputs, outputs, hs)
                
                comm.publish_architecture_to_worker(vs, models_sampled, 
                                       searcher_eval_token)

                models_sampled += 1

                if models_sampled % save_every == 0:
                    searcher.save_state(search_data_folder)

        else:
            if comm.is_ready_to_publish_architecture():
                comm.kill_worker()
                killed += 1

        # See which workers have finished evaluation
        for worker in range(comm.num_workers):
            msg = comm.receive_results_in_master(worker)
            if msg is not None:
                results, model_id, searcher_eval_token = msg
                eval_logger = sl.EvaluationLogger(folderpath, search_name, model_id)
                eval_logger.log_results(results)

                if 'epoch' in results:
                    epochs = max(epochs, results['epoch'])

                searcher.update(results['validation_accuracy'], searcher_eval_token)
                finished += 1

def start_worker(comm, evaluator, search_space_factory, folderpath, 
    search_name, resume=True, save_every=1):
    # set the available gpu for process
    print('WORKER %d' % comm.get_rank())
    step = 0
    if len(gpu_utils.get_gpu_information()) != 0:
        #https://github.com/tensorflow/tensorflow/issues/1888
        gpu_utils.set_visible_gpus([comm.get_rank() % gpu_utils.get_total_num_gpus()])

    search_data_folder = sl.get_search_data_folderpath(folderpath, search_name)

    if resume:
        evaluator.load_state(search_data_folder)

    while(True):
        arch = comm.receive_architecture_in_worker()

        # if a kill signal is received        
        if arch is None:
            break
        
        vs, evaluation_id, searcher_eval_token = arch

        inputs, outputs, hs = search_space_factory.get_search_space()
        se.specify(outputs.values(), hs, vs)
        results = evaluator.eval(inputs, outputs, hs)
        step += 1
        if step % save_every == 0:
            evaluator.save_state(search_data_folder)
        comm.publish_results_to_master(results, evaluation_id, searcher_eval_token)

def main():
    configs = ut.read_jsonfile("./dev/architecture_search_benchmarks/experiment_config.json")

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

    options = parser.parse_args()
    config = configs[options.config_name]

    if 'eager' in config and config['eager']:
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth=True
        tf.enable_eager_execution(tfconfig, device_policy=tf.contrib.eager.DEVICE_PLACEMENT_SILENT)

    datasets = {
        'cifar10': lambda: (load_cifar10('data/cifar10/'), 10),
        'mnist': lambda: (load_mnist('data/mnist/'), 10)
    }

    (Xtrain, ytrain, Xval, yval, Xtest, ytest), num_classes = datasets[config['dataset']]()
    search_space_factory = name_to_search_space_factory_fn[config['search_space']](num_classes)

    num_procs = config['num_procs'] if 'num_procs' in config else 0
    comm = get_communicator(config['communicator'], num_procs)

    save_every = 1 if 'save_every' not in config else config['save_every']
    if comm.get_rank() == 0:
        searcher = name_to_searcher_fn[config['searcher']](search_space_factory.get_search_space)
        num_samples = -1 if 'samples' not in config else config['samples']
        num_epochs = -1 if 'epochs' not in config else config['epochs']
        start_searcher(
            comm, searcher, options.resume, config['search_folder'], 
            config['search_name'], config['searcher_load_file_name'], 
            num_samples=num_samples, num_epochs=num_epochs, save_every=save_every)
    else:
        train_dataset = InMemoryDataset(Xtrain, ytrain, True)
        val_dataset = InMemoryDataset(Xval, yval, False)
        test_dataset = InMemoryDataset(Xtest, ytest, False)

        evaluators = {
            'simple_classification': lambda: SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes,
                        './temp' + str(comm.get_rank()), max_num_training_epochs=config['epochs'], log_output_to_terminal=options.display_output,
                        test_dataset=test_dataset),
            'enas_evaluator': lambda: ENASEvaluator(train_dataset, val_dataset, num_classes,
                        search_space_factory.weight_sharer)
        }

        assert not config['evaluator'].startswith('enas') or hasattr(search_space_factory, 'weight_sharer')
        evaluator = evaluators[config['evaluator']]()

        start_worker(comm, evaluator, search_space_factory, config['search_folder'], 
            config['search_name'], resume=options.resume, save_every=save_every)

if __name__ == "__main__":
    main()
