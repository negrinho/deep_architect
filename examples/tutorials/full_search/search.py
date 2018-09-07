"""
This tutorial is for a basic architecture search using an MPI communicator
and a random searcher.

To run this tutorial, simply call 'mpiexec -np NP python search.py' where NP is
the number of evaluator workers + 1 (to account for the searcher process).

If running this tutorial with a file based communicator, change the argument of
the get_communicator function to 'file' and number of processes that are going
to be used. Then, run 'python search.py' for as many processes as you
want (eg run 5 concurrent calls of 'python search.py' if you want 4 workers and
1 searcher)
"""

from deep_architect.contrib.misc.datasets.loaders import load_mnist
from deep_architect.contrib.misc.datasets.dataset import InMemoryDataset
from deep_architect.searchers.random import RandomSearcher
from deep_architect.contrib.misc.evaluators.tensorflow.classification import SimpleClassifierEvaluator

import deep_architect.modules as mo
import deep_architect.contrib.misc.search_spaces.tensorflow.dnn as dnn
import deep_architect.searchers.common as se

from dev.architecture_search_benchmarks.communicators.communicator import get_communicator

"""
This is simply a wrapper for the function that returns a new search space.
"""
class SSF(mo.SearchSpaceFactory):
    def __init__(self, num_classes):
        mo.SearchSpaceFactory.__init__(self)
        self.num_classes = num_classes

    def _get_search_space(self):
        inputs, outputs = dnn.dnn_net(self.num_classes)
        return inputs, outputs, {}

def main():
    # create the communicator
    comm = get_communicator('file', 2)

    # number of total models to be evaluated in search
    num_total_models = 25

    # Set up the datasets and the search space
    X_train, y_train, X_val, y_val, _, _ = load_mnist('data/mnist', normalize_range=True)
    train_dataset = InMemoryDataset(X_train, y_train, True)
    val_dataset = InMemoryDataset(X_val, y_val, False)
    ssf = SSF(10)

    # If the rank of this process is 0, it is the searcher process.
    if comm.get_rank() == 0:

        # Create the searcher
        searcher = RandomSearcher(ssf.get_search_space)

        models_sampled = 0
        killed = 0
        finished = 0

        # Keep looping as long as we have not received results for all sampled
        # models
        while finished < num_total_models:
            if models_sampled < num_total_models:

                # Check by the communicator to see if worker queue is ready
                # for a new architecture
                if comm.is_ready_to_publish_architecture():
                    _, _, _, vs, se_token = searcher.sample()
                    comm.publish_architecture_to_worker(vs, models_sampled, se_token)
                    models_sampled += 1

            # If we are over the specified number of models to be sampled, we
            # send a kill signal to each worker. Each worker should only consume
            # one kill signal, so if the number of kill signals the searcher
            # sends is equal to the number of workers, all workers should have
            # received a kill signal
            else:
                if comm.is_ready_to_publish_architecture() and killed < comm.num_workers:
                    comm.kill_worker()
                    killed += 1

            # Go through each worker and see if they have any new results to
            # update the searcher with
            for worker in range(comm.num_workers):
                msg = comm.receive_results_in_master(worker)
                if msg is not None:
                    results, model_id, searcher_eval_token = msg
                    searcher.update(results['validation_accuracy'], searcher_eval_token)
                    finished += 1
                    print('Model %d accuracy: %f' % (model_id, results['validation_accuracy']))
        print('Best architecture accuracy: %f' % searcher.best_acc)
        print('Best architecture params: %r' % searcher.best_vs)

    # This is a worker process
    else:
        # Initialize an evaluator
        evaluator = SimpleClassifierEvaluator(train_dataset, val_dataset, 10,
            './temp', max_num_training_epochs=2)

        while(True):
            arch = comm.receive_architecture_in_worker()

            # if a kill signal is received break out of the loop and end the
            # worker process
            if arch is None:
                break

            vs, evaluation_id, searcher_eval_token = arch

            # Create a new unspecified search space, and specify it using the
            # values sent by the searcher
            inputs, outputs, hs = ssf.get_search_space()
            se.specify(outputs.values(), hs, vs)

            # Evaluate the specified search space and send it back to the
            # searcher
            results = evaluator.eval(inputs, outputs, hs)
            comm.publish_results_to_master(results, evaluation_id, searcher_eval_token)

if __name__ == '__main__':
    main()
