#TODO Comments + README
import argparse
import pickle
import json
import time
import subprocess
import base64
import logging
from google.cloud import pubsub_v1

import deep_architect.utils as ut
from deep_architect.contrib.misc.datasets.loaders import (load_cifar10,
                                                          load_mnist)
from deep_architect.contrib.misc.datasets.dataset import InMemoryDataset

from deep_architect.searchers import common as se
from deep_architect.contrib.misc import gpu_utils
from deep_architect import search_logging as sl
from deep_architect import utils as ut

from search_space_factory import name_to_search_space_factory_fn
from searcher import name_to_searcher_fn

from dev.enas.evaluator.enas_evaluator import ENASEvaluator
from deep_architect.contrib.misc.evaluators.tensorflow.classification import SimpleClassifierEvaluator

from deep_architect.contrib.communicators.communicator import get_communicator
logging.basicConfig()

configs = ut.read_jsonfile(
    "/deep_architect/dev/google_communicator/experiment_config.json")

parser = argparse.ArgumentParser("MPI Job for architecture search")
parser.add_argument('--config',
                    '-c',
                    action='store',
                    dest='config_name',
                    default='normal')
parser.add_argument('--project-id',
                    '-p',
                    action='store',
                    dest='project_id',
                    default='normal')
parser.add_argument('--bucket',
                    '-b',
                    action='store',
                    dest='bucket',
                    default='normal')

# Other arguments
parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    dest='resume',
                    default=False)

options = parser.parse_args()
config = configs[options.config_name]

PROJECT_ID = options.project_id
BUCKET_NAME = options.bucket
publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()
results_subscription = subscriber.subscription_path(PROJECT_ID, 'results-sub')
arch_topic = publisher.topic_path(PROJECT_ID, 'architectures')
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
    'cifar10': ('data/cifar10/', 10),
}

data_dir, num_classes = datasets[config['dataset']]
search_space_factory = name_to_search_space_factory_fn[config['search_space']](
    num_classes)

save_every = 1 if 'save_every' not in config else config['save_every']
searcher = name_to_searcher_fn[config['searcher']](
    search_space_factory.get_search_space)
num_epochs = -1 if 'epochs' not in config else config['epochs']
num_samples = -1 if 'samples' not in config else config['samples']
eval_epochs = config['eval_epochs']

# SET UP GOOGLE STORE FOLDER
search_logger = sl.SearchLogger(config['search_folder'], config['search_name'])
# sl.create_search_folderpath(config['search_folder'], config['search_name'])
search_data_folder = search_logger.get_search_data_folderpath()
save_filepath = ut.join_paths(
    (search_data_folder, config['searcher_file_name']))
eval_path = sl.get_all_evaluations_folderpath(config['search_folder'],
                                              config['search_name'])
search_folder = sl.get_search_folderpath(config['search_folder'],
                                         config['search_name'])
models_sampled = 0
epochs = 0
finished = 0
killed = 0
best_accuracy = 0.

# Load previous searcher
if options.resume:
    download_folder(search_folder)
    searcher.load(search_data_folder)
    state = ut.read_jsonfile(save_filepath)
    epochs = state['epochs']
    models_sampled = state['finished']
    finished = state['finished']
    # killed = state['killed']


def download_folder(folder):
    subprocess.check_call([
        'gsutil', '-m', 'cp', '-r', 'gs://' + BUCKET_NAME + '/' + folder + '/',
        folder + '/'
    ])


def upload_folder(folder, location):
    subprocess.check_call([
        'gsutil', '-m', 'cp', '-r', folder,
        'gs://' + BUCKET_NAME + '/' + location + '/'
    ])


def update_searcher(message):
    global finished
    global best_accuracy
    global epochs
    # print(message.data)
    # print(message.data.decode('utf-8'))
    # print(json.loads(message.data.decode('utf-8')))
    # data = json.loads(message.data.decode('utf-8'))
    # data = base64.b64decode(message['data']).decode('utf-8')
    data = json.loads(message.data.decode('utf-8'))
    # data = json.loads(message.data)
    if not data == 'publish':
        print('Updating')
        results, vs, evaluation_id, searcher_eval_token = data
        eval_logger = search_logger.get_evaluation_logger(evaluation_id)
        eval_logger.log_config(vs, searcher_eval_token)
        eval_logger.log_results(results)
        upload_folder(eval_logger.get_evaluation_folderpath(), eval_path)

        searcher.update(results['validation_accuracy'], searcher_eval_token)
        print('Updated')
        best_accuracy = max(best_accuracy, results['validation_accuracy'])
        finished += 1
        epochs += eval_epochs
        if finished % save_every == 0:
            print('Models sampled: %d Best Accuracy: %f' %
                  (finished, best_accuracy))
            best_accuracy = 0.

            searcher.save_state(search_data_folder)
            state = {'models_finished': finished, 'epochs': epochs}
            ut.write_jsonfile(state, save_filepath)
            upload_folder(search_logger.get_search_data_folderpath(),
                          search_folder)

    message.ack()
    print('Sending new architecture')
    publish_new_arch()


search_finished = False


def publish_new_arch():
    global models_sampled
    global search_finished
    kill = num_samples != -1 and finished >= num_samples
    kill = kill or (num_epochs != -1 and epochs >= num_epochs)
    if kill:
        arch = 'kill'
        encoded_results = json.dumps(arch).encode('utf-8')
        future = publisher.publish(arch_topic, encoded_results)
        future.result()
        search_finished = True
    else:
        cont = num_samples == -1 or models_sampled < num_samples
        cont = cont and (num_epochs == -1 or epochs < num_epochs)
        if cont:
            eval_logger = search_logger.get_evaluation_logger(models_sampled)
            inputs, outputs, vs, searcher_eval_token = searcher.sample()

            # eval_logger.log_config(vs, searcher_eval_token)
            # eval_logger.log_features(inputs, outputs, hs)
            arch = (vs, models_sampled, searcher_eval_token)

            encoded_results = json.dumps(arch).encode('utf-8')
            future = publisher.publish(arch_topic, encoded_results)
            future.result()
            models_sampled += 1


subscriber.subscribe(results_subscription, callback=update_searcher)
while (not search_finished):
    time.sleep(30)

# subprocess.check_call([
#     'gsutil', '-m', 'cp', '-r',
#     './' + config['search_folder'] + '/' + config['search_name'] + '/', 'gs://'
#     + BUCKET_NAME + '/' + config['search_folder'] + '/' + config['search_name']
# ])

# if __name__ == "__main__":
#     main()
