###${MARKDOWN}
# This tutorial is for a basic architecture search using a random searcher.
#
# If running this tutorial with an MPI based communicator, use the
# [launch_mpi_based_search.sh](examples/tutorials/xx.full_search/launch_mpi_based_search.sh)
# script to run the search. Run the script as
# `./examples/tutorials/xx.full_search/launch_mpi_based_search.sh NP` where NP
# is the number of evaluator workers + 1 (to account for the searcher process)
#
# If running this tutorial with a file based communicator, use the
# [launch_file_based_search.sh](examples/tutorials/xx.full_search/launch_file_based_search.sh)
# script to run the search. Run the script as
# `./examples/tutorials/xx.full_search/launch_file_based_search.sh NP`

import argparse

from deep_architect.contrib.misc.datasets.loaders import load_mnist
from deep_architect.contrib.misc.datasets.dataset import InMemoryDataset
from deep_architect.searchers.random import RandomSearcher
from deep_architect.contrib.misc.evaluators.tensorflow.classification import SimpleClassifierEvaluator

import deep_architect.modules as mo
import deep_architect.contrib.misc.search_spaces.tensorflow.dnn as dnn
import deep_architect.searchers.common as se

from deep_architect.contrib.communicators.communicator import get_communicator

parser = argparse.ArgumentParser(description='Run an architecture search.')
parser.add_argument('--comm',
                    '-c',
                    choices=['mpi', 'file'],
                    required=True,
                    default='mpi')
parser.add_argument('--num-procs', '-n', type=int, default=2)
args = parser.parse_args()

# First, create the communicator. This communicator is used by by to master to
# send candidate architectures to the workers to evaluate, and by the workers
# to send back the results for the architectures they evaluated. Currently,
# the communicator can be MPI based or file based (file based requires the
# processes to share a filesystem).
comm = get_communicator(args.comm, num_procs=args.num_procs)

# This is the number of total models to be evaluated in search
num_total_models = 25

# Now we set up the datasets and the search space factory.
X_train, y_train, X_val, y_val, _, _ = load_mnist('data/mnist',
                                                  normalize_range=True)
train_dataset = InMemoryDataset(X_train, y_train, True)
val_dataset = InMemoryDataset(X_val, y_val, False)
ssf = mo.SearchSpaceFactory(lambda: dnn.dnn_net(10))

# Each process should have a unique rank. The process with rank 0 will act as the
# master process that is in charge of the searcher. Every other process acts
# as a worker that evaluates architectures sent to them.
if comm.get_rank() == 0:
    searcher = RandomSearcher(ssf.get_search_space)

    models_sampled = 0
    killed = 0
    finished = 0

    # This process keeps going as long as we have not received results for all sampled
    # models and not all the worker processes have been killed. Kill signals start
    # being sent out once the searcher has finished sampling the number of models
    # specified by the `num_total_models` parameter
    while finished < models_sampled or killed < comm.num_workers:
        if models_sampled < num_total_models:

            # Now, we check the communicator to see if worker queue is ready for a new
            # architecture. If so, we publish an architecture to the worker queue.
            if comm.is_ready_to_publish_architecture():
                _, _, vs, se_token = searcher.sample()
                comm.publish_architecture_to_worker(vs, models_sampled,
                                                    se_token)
                models_sampled += 1

# If we are over the specified number of models to be sampled, we
# send a kill signal to each worker. Each worker should only consume
# one kill signal, so if the number of kill signals the searcher
# sends is equal to the number of workers, all workers should have
# received a kill signal
        else:
            if comm.is_ready_to_publish_architecture(
            ) and killed < comm.num_workers:
                comm.kill_worker()
                killed += 1

# After sending the appropriate messages to the workers, the master process
# needs to go through each worker and see if it has received any new results to
# update the searcher with
        for worker in range(comm.num_workers):
            msg = comm.receive_results_in_master(worker)
            if msg is not None:
                results, model_id, searcher_eval_token = msg
                searcher.update(results['validation_accuracy'],
                                searcher_eval_token)
                finished += 1
                print('Model %d accuracy: %f' %
                      (model_id, results['validation_accuracy']))

# At this point, all of the workers should be killed, and the searcher should
# have evaluated all the architectures it needed to finish its search.
# print('Best architecture accuracy: %f' % searcher.best_acc)
# print('Best architecture params: %r' % searcher.best_vs)
else:
    evaluator = SimpleClassifierEvaluator(train_dataset,
                                          val_dataset,
                                          10,
                                          './temp',
                                          max_num_training_epochs=2)

    # This process keeps going until it receives a kill signal from the master
    # process. At that point, it breaks out of its loop and ends.
    while (True):
        arch = comm.receive_architecture_in_worker()

        if arch is None:
            break

        vs, evaluation_id, searcher_eval_token = arch

        # In order to evaluate the architecture sent by the searcher, we create a new
        # unspecified search space, and recreate the architecture using the values of
        # the hyperparameters received by the worker.
        inputs, outputs = ssf.get_search_space()
        se.specify(outputs.values(), vs)
        results = evaluator.eval(inputs, outputs)
        comm.publish_results_to_master(results, evaluation_id,
                                       searcher_eval_token)
