import argparse
import pickle
import json
import time
import threading
import pprint
import deep_architect.utils as ut
from google.cloud import pubsub_v1

from deep_architect.contrib.misc.datasets.loaders import (load_cifar10,
                                                          load_mnist)
from deep_architect.contrib.misc.datasets.dataset import InMemoryDataset

from deep_architect.searchers import common as se
from deep_architect.contrib.misc import gpu_utils
from deep_architect import search_logging as sl
from deep_architect import utils as ut

from dev.google_communicator.search_space_factory import name_to_search_space_factory_fn

from deep_architect.contrib.misc.evaluators.tensorflow.tpu_estimator_classification import TPUEstimatorEvaluator

from deep_architect.communicators.communicator import get_communicator
import logging

logging.basicConfig()

publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()
results_topic = None
arch_subscription = None

specified = False
evaluated = False
arch_data = None
started = False


def retrieve_message(message):
    global specified
    global evaluated
    global arch_data
    global started
    # print(type(message.data))
    # print(message.data)
    # print(message.data.decode('utf-8'))
    # print(json.loads(message.data.decode('utf-8')))
    started = True
    data = json.loads(message.data.decode('utf-8'))
    if data == 'kill':
        arch_data = None
        specified = True
        message.nack()
    else:
        arch_data = data
        evaluated = False
        specified = True
        last_refresh = 0
        while not evaluated:
            if last_refresh > 300:
                message.modify_ack_deadline(600)
                last_refresh = 0
            time.sleep(5)
            last_refresh += 5
        message.ack()


def main():
    global specified
    global evaluated
    global results_topic, arch_subscription
    configs = ut.read_jsonfile(
        "/darch/dev/google_communicator/experiment_config.json")

    parser = argparse.ArgumentParser("MPI Job for architecture search")
    parser.add_argument('--config',
                        '-c',
                        action='store',
                        dest='config_name',
                        default='search_evol')

    # Other arguments
    parser.add_argument('--display-output',
                        '-o',
                        action='store_true',
                        dest='display_output',
                        default=False)
    parser.add_argument('--project-id',
                        action='store',
                        dest='project_id',
                        default='deep-architect')
    parser.add_argument('--bucket',
                        '-b',
                        action='store',
                        dest='bucket',
                        default='normal')
    parser.add_argument('--resume',
                        '-r',
                        action='store_true',
                        dest='resume',
                        default=False)

    options = parser.parse_args()
    config = configs[options.config_name]

    PROJECT_ID = options.project_id
    BUCKET_NAME = options.bucket
    results_topic = publisher.topic_path(PROJECT_ID, 'results')
    arch_subscription = subscriber.subscription_path(PROJECT_ID,
                                                     'architectures-sub')

    datasets = {
        'cifar10': ('/data/cifar10/', 10),
    }

    data_dir, num_classes = datasets[config['dataset']]
    search_space_factory = name_to_search_space_factory_fn[
        config['search_space']](num_classes)

    save_every = 1 if 'save_every' not in config else config['save_every']

    evaluators = {
        'tpu_classification':
        lambda: TPUEstimatorEvaluator(
            'gs://' + BUCKET_NAME + data_dir,
            max_num_training_epochs=config['eval_epochs'],
            log_output_to_terminal=options.display_output,
            base_dir='gs://' + BUCKET_NAME + '/scratch_dir'),
    }

    evaluator = evaluators[config['evaluator']]()

    search_data_folder = sl.get_search_data_folderpath(config['search_folder'],
                                                       config['search_name'])
    subscription = subscriber.subscribe(arch_subscription,
                                        callback=retrieve_message)
    thread = threading.Thread(target=nudge_master)
    thread.start()
    step = 0
    while True:
        while not specified:
            time.sleep(5)
        if arch_data:
            vs, evaluation_id, searcher_eval_token = arch_data
            inputs, outputs = search_space_factory.get_search_space()
            se.specify(outputs.values(), vs)
            print('Evaluating architecture')
            results = evaluator.eval(inputs, outputs)
            print('Evaluated architecture')
            step += 1
            if step % save_every == 0:
                evaluator.save_state(search_data_folder)
            encoded_results = json.dumps((results, vs, evaluation_id,
                                          searcher_eval_token)).encode('utf-8')
            future = publisher.publish(results_topic, encoded_results)
            future.result()
            evaluated = True
            specified = False
        else:
            break
    thread.join()
    subscription.cancel()


def nudge_master():
    global started
    time.sleep(10)
    if not started:
        encoded_results = json.dumps('publish').encode('utf-8')
        future = publisher.publish(results_topic, encoded_results)
        future.result()


if __name__ == "__main__":
    main()
