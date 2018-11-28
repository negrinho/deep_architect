import argparse
import pickle
import json
import time
import subprocess

from google.cloud import pubsub_v1
from google.cloud import storage


import deep_architect.utils as ut
from deep_architect.contrib.misc.datasets.loaders import (
    load_cifar10, load_mnist, load_fashion_mnist)
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

PROJECT_ID = 'deeparchitect-219016'
BUCKET_NAME = 'deep_architect'
publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()
results_subscription = publisher.subscription_path(PROJECT_ID, 'results')
arch_topic = subscriber.topic_path(PROJECT_ID, 'architectures')


configs = ut.read_jsonfile("./examples/tensorflow/full_benchmarks/experiment_config.json")

parser = argparse.ArgumentParser("MPI Job for architecture search")
parser.add_argument('--config', '-c', action='store', dest='config_name',
default='normal')
parser.add_argument('--project-id', '-p', action='store', dest='project_id',
default='normal')
parser.add_argument('--bucket', '-b', action='store', dest='bucket',
default='normal')


# Other arguments
parser.add_argument('--resume', '-r', action='store_true', dest='resume',
                    default=False)

options = parser.parse_args()
config = configs[options.config_name]

# num_procs = config['num_procs'] if 'num_procs' in config else 0
# if len(gpu_utils.get_gpu_information()) != 0:
#     #https://github.com/tensorflow/tensorflow/issues/1888
#     gpu_utils.set_visible_gpus([comm.get_rank() % gpu_utils.get_total_num_gpus()])

# if 'eager' in config and config['eager']:
#     import tensorflow as tf
#     tf.logging.set_verbosity(tf.logging.ERROR)
#     tfconfig = tf.ConfigProto()
#     tfconfig.gpu_options.allow_growth=True
#     tf.enable_eager_execution(tfconfig, device_policy=tf.contrib.eager.DEVICE_PLACEMENT_SILENT)

datasets = {
    'cifar10': lambda: (load_cifar10('data/cifar10/'), 10),
    'mnist': lambda: (load_mnist('data/mnist/'), 10),
    'fashion_mnist': lambda: (load_fashion_mnist(), 10)
}

(Xtrain, ytrain, Xval, yval, Xtest, ytest), num_classes = datasets[config['dataset']]()
search_space_factory = name_to_search_space_factory_fn[config['search_space']](num_classes)

save_every = 1 if 'save_every' not in config else config['save_every']
searcher = name_to_searcher_fn[config['searcher']](search_space_factory.get_search_space)
num_epochs = -1 if 'epochs' not in config else config['epochs']
sl.create_search_folderpath(config['search_folder'], config['search_name'])
search_data_folder = sl.get_search_data_folderpath(config['search_folder'], config['search_name'])

save_filepath = ut.join_paths((search_data_folder, config['searcher_file_name']))
eval_epochs = config['eval_epochs']
models_sampled = 0
epochs = 0
finished = 0
killed = 0
best_accuracy = 0.

# Load previous searcher
if options.resume:
    searcher.load(search_data_folder)
    state = ut.read_jsonfile(save_filepath)
    epochs = state['epochs']
    models_sampled = state['finished']
    finished = state['finished']
    # killed = state['killed']

def update_searcher(message):
    global finished
    global best_accuracy
    data = json.loads(message.data.decode('utf-8'))
    results, evaluation_id, searcher_eval_token = data
    eval_logger = sl.EvaluationLogger(
        config['search_folder'], config['search_name'], evaluation_id)
    eval_logger.log_results(results)

    searcher.update(results['validation_accuracy'], searcher_eval_token)
    best_accuracy = max(best_accuracy, results['validation_accuracy'])
    finished += 1
    if epochs % save_every == 0:
        print('Models sampled: %d Best Accuracy: %f' % (finished, best_accuracy))
        best_accuracy = 0.

        searcher.save_state(search_data_folder)
        state = {
            'models_finished': finished,
            'epochs': epochs,
        }
        ut.write_jsonfile(state, save_filepath)
    message.ack()

    publish_new_arch()

def publish_new_arch():
    global models_sampled
    global epochs
    if epochs >= num_epochs:
        arch = 'kill'
    else:
        eval_logger = sl.EvaluationLogger(
            config['search_folder'], config['search_name'], models_sampled)
        inputs, outputs, hs, vs, searcher_eval_token = searcher.sample()

        eval_logger.log_config(vs, searcher_eval_token)
        # eval_logger.log_features(inputs, outputs, hs)
        arch = (vs, models_sampled, searcher_eval_token)

    encoded_results = json.dumps(arch).encode('utf-8')
    future = publisher.publish(arch_topic, encoded_results)
    future.result()
    epochs += eval_epochs
    models_sampled += 1


subscriber.subscribe(results_subscription, callback=update_searcher)
for i in range(len(config['num_workers'])):
    if epochs < num_epochs:
        publish_new_arch()

while not epochs < num_epochs:
    time.sleep(30)


subprocess.check_call([
    'gsutil', '-m', 'cp', '-r',
    './' + config['search_folder'] + '/' + config['search_name'] + '/', 'gs://' + BUCKET_NAME + '/' + config['search_folder'] + '/' + config['search_name']])

# if __name__ == "__main__":
#     main()
