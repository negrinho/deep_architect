import argparse
import deep_architect.utils as ut

from deep_architect.contrib.misc.datasets.loaders import (load_cifar10,
                                                          load_mnist)
from deep_architect.contrib.misc.datasets.dataset import InMemoryDataset

from deep_architect.searchers import common as se
from deep_architect.contrib.misc import gpu_utils
from deep_architect import search_logging as sl

from search_space_factory import name_to_search_space_factory_fn
from searcher import name_to_searcher_fn

from deep_architect.contrib.misc.evaluators.tensorflow.classification import SimpleClassifierEvaluator

from deep_architect.contrib.communicators.communicator import get_communicator


def start_searcher(comm,
                   searcher,
                   resume_if_exists,
                   folderpath,
                   search_name,
                   searcher_load_path,
                   num_samples=-1,
                   num_epochs=-1,
                   save_every=1):
    assert num_samples != -1 or num_epochs != -1

    print('SEARCHER')

    sl.create_search_folderpath(folderpath, search_name)
    search_data_folder = sl.get_search_data_folderpath(folderpath, search_name)
    save_filepath = ut.join_paths((search_data_folder, searcher_load_path))

    models_sampled = 0
    epochs = 0
    finished = 0
    killed = 0
    best_accuracy = 0.

    # Load previous searcher
    if resume_if_exists:
        searcher.load(search_data_folder)
        state = ut.read_jsonfile(save_filepath)
        epochs = state['epochs']
        killed = state['killed']
        models_sampled = state['models_finished']
        finished = state['models_finished']

    while (finished < models_sampled or killed < comm.num_workers):
        # Search end conditions
        cont = num_samples == -1 or models_sampled < num_samples
        cont = cont and (num_epochs == -1 or epochs < num_epochs)
        if cont:
            # See whether workers are ready to consume architectures
            if comm.is_ready_to_publish_architecture():
                eval_logger = sl.EvaluationLogger(folderpath, search_name,
                                                  models_sampled)
                _, _, vs, searcher_eval_token = searcher.sample()

                eval_logger.log_config(vs, searcher_eval_token)
                comm.publish_architecture_to_worker(vs, models_sampled,
                                                    searcher_eval_token)

                models_sampled += 1
        else:
            if comm.is_ready_to_publish_architecture():
                comm.kill_worker()
                killed += 1

        # See which workers have finished evaluation
        for worker in range(comm.num_workers):
            msg = comm.receive_results_in_master(worker)
            if msg is not None:
                results, model_id, searcher_eval_token = msg
                eval_logger = sl.EvaluationLogger(folderpath, search_name,
                                                  model_id)
                eval_logger.log_results(results)

                if 'epoch' in results:
                    epochs = max(epochs, results['epoch'])

                searcher.update(results['validation_accuracy'],
                                searcher_eval_token)
                best_accuracy = max(best_accuracy,
                                    results['validation_accuracy'])
                finished += 1
                if finished % save_every == 0:
                    print('Models sampled: %d Best Accuracy: %f' %
                          (finished, best_accuracy))
                    best_accuracy = 0.

                    searcher.save_state(search_data_folder)
                    state = {
                        'models_finished': finished,
                        'epochs': epochs,
                        'killed': killed
                    }
                    ut.write_jsonfile(state, save_filepath)


def start_worker(comm,
                 evaluator,
                 search_space_factory,
                 folderpath,
                 search_name,
                 resume=True,
                 save_every=1):
    # set the available gpu for process
    print('WORKER %d' % comm.get_rank())
    step = 0

    sl.create_search_folderpath(folderpath, search_name)
    search_data_folder = sl.get_search_data_folderpath(folderpath, search_name)
    save_filepath = ut.join_paths(
        (search_data_folder, 'worker' + str(comm.get_rank()) + '.json'))

    if resume:
        evaluator.load_state(search_data_folder)
        state = ut.read_jsonfile(save_filepath)
        step = state['step']

    while (True):
        arch = comm.receive_architecture_in_worker()

        # if a kill signal is received
        if arch is None:
            break

        vs, evaluation_id, searcher_eval_token = arch

        inputs, outputs = search_space_factory.get_search_space()
        se.specify(outputs, vs)
        results = evaluator.eval(inputs, outputs)
        step += 1
        if step % save_every == 0:
            evaluator.save_state(search_data_folder)
            state = {'step': step}
            ut.write_jsonfile(state, save_filepath)
        comm.publish_results_to_master(results, evaluation_id,
                                       searcher_eval_token)


def main():
    configs = ut.read_jsonfile(
        "./examples/tensorflow/full_benchmarks/experiment_config.json")

    parser = argparse.ArgumentParser("MPI Job for architecture search")
    parser.add_argument('--config',
                        '-c',
                        action='store',
                        dest='config_name',
                        default='normal')

    # Other arguments
    parser.add_argument('--display-output',
                        '-o',
                        action='store_true',
                        dest='display_output',
                        default=False)
    parser.add_argument('--resume',
                        '-r',
                        action='store_true',
                        dest='resume',
                        default=False)

    options = parser.parse_args()
    config = configs[options.config_name]

    num_procs = config['num_procs'] if 'num_procs' in config else 0
    comm = get_communicator(config['communicator'], num_procs)
    if len(gpu_utils.get_gpu_information()) != 0:
        #https://github.com/tensorflow/tensorflow/issues/1888
        gpu_utils.set_visible_gpus(
            [comm.get_rank() % gpu_utils.get_total_num_gpus()])

    if 'eager' in config and config['eager']:
        import tensorflow as tf
        tf.logging.set_verbosity(tf.logging.ERROR)
        tf.enable_eager_execution()
    datasets = {
        'cifar10': lambda: (load_cifar10('data/cifar10/', one_hot=False), 10),
        'mnist': lambda: (load_mnist('data/mnist/'), 10),
    }

    (Xtrain, ytrain, Xval, yval, Xtest,
     ytest), num_classes = datasets[config['dataset']]()
    search_space_factory = name_to_search_space_factory_fn[
        config['search_space']](num_classes)

    save_every = 1 if 'save_every' not in config else config['save_every']
    if comm.get_rank() == 0:
        searcher = name_to_searcher_fn[config['searcher']](
            search_space_factory.get_search_space)
        num_samples = -1 if 'samples' not in config else config['samples']
        num_epochs = -1 if 'epochs' not in config else config['epochs']
        start_searcher(comm,
                       searcher,
                       options.resume,
                       config['search_folder'],
                       config['search_name'],
                       config['searcher_file_name'],
                       num_samples=num_samples,
                       num_epochs=num_epochs,
                       save_every=save_every)
    else:
        train_d_advataset = InMemoryDataset(Xtrain, ytrain, True)
        val_dataset = InMemoryDataset(Xval, yval, False)
        test_dataset = InMemoryDataset(Xtest, ytest, False)

        search_path = sl.get_search_folderpath(config['search_folder'],
                                               config['search_name'])
        ut.create_folder(ut.join_paths([search_path, 'scratch_data']),
                         create_parent_folders=True)
        scratch_folder = ut.join_paths(
            [search_path, 'scratch_data', 'eval_' + str(comm.get_rank())])
        ut.create_folder(scratch_folder)

        evaluators = {
            'simple_classification':
            lambda: SimpleClassifierEvaluator(
                train_dataset,
                val_dataset,
                num_classes,
                './temp' + str(comm.get_rank()),
                max_num_training_epochs=config['eval_epochs'],
                log_output_to_terminal=options.display_output,
                test_dataset=test_dataset),
        }

        assert not config['evaluator'].startswith('enas') or hasattr(
            search_space_factory, 'weight_sharer')
        evaluator = evaluators[config['evaluator']]()

        start_worker(comm,
                     evaluator,
                     search_space_factory,
                     config['search_folder'],
                     config['search_name'],
                     resume=options.resume,
                     save_every=save_every)


if __name__ == "__main__":
    main()
