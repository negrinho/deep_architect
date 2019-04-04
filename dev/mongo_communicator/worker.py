import argparse
import time
import threading
import logging
import os

from deep_architect.searchers import common as se
from deep_architect import search_logging as sl
from deep_architect import utils as ut

from deep_architect.contrib.misc.evaluators.tensorflow.tpu_estimator_classification import AdvanceClassifierEvaluator

from dev.mongo_communicator.search_space_factory import name_to_search_space_factory_fn
from dev.mongo_communicator.master import (
    RESULTS_TOPIC, ARCH_TOPIC, KILL_SIGNAL, PUBLISH_SIGNAL, get_topic_name)
from dev.mongo_communicator.mongo_communicator import MongoCommunicator

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
BUCKET_NAME = 'deep_architect'


def process_config_and_args():
    parser = argparse.ArgumentParser("Worker Job for architecture search")
    parser.add_argument(
        '--config',
        '-c',
        action='store',
        dest='config_name',
        default='search_evol')
    parser.add_argument(
        '--config-file',
        action='store',
        dest='config_file',
        default='/darch/dev/mongo_communicator/experiment_config.json')
    # Other arguments
    parser.add_argument(
        '--display-output',
        '-o',
        action='store_true',
        dest='display_output',
        default=False)
    parser.add_argument(
        '--resume', '-r', action='store_true', dest='resume', default=False)
    parser.add_argument(
        '--bucket', '-b', action='store', dest='bucket', default=BUCKET_NAME)
    parser.add_argument(
        '--mongo-host',
        '-m',
        action='store',
        dest='mongo_host',
        default='127.0.0.1')
    parser.add_argument(
        '--tpu-name', '-t', action='store', dest='tpu_name', default='')
    parser.add_argument(
        '--mongo-port', '-p', action='store', dest='mongo_port', default=27017)

    options = parser.parse_args()
    configs = ut.read_jsonfile(options.config_file)
    config = configs[options.config_name]

    config['bucket'] = options.bucket

    datasets = {
        'cifar10': ('/data/cifar10/', 10),
    }

    data_dir, num_classes = datasets[config['dataset']]
    search_space_factory = name_to_search_space_factory_fn[
        config['search_space']](num_classes)

    config['save_every'] = 1 if 'save_every' not in config else config[
        'save_every']

    evaluators = {
        'tpu_classification':
        lambda: AdvanceClassifierEvaluator(
            'gs://' + config['bucket'] + data_dir,
            options.tpu_name,
            max_num_training_epochs=config['eval_epochs'],
            log_output_to_terminal=options.display_output,
            base_dir='gs://' + config['bucket'] + '/' + config['search_folder']
            + '/' + config['search_name'] + '/scratch_dir'),
    }

    comm = MongoCommunicator(options.mongo_host, options.mongo_port)
    evaluator = evaluators[config['evaluator']]()

    return comm, search_space_factory, evaluator, config


def retrieve_message(message, comm, config, state):
    state['started'] = True
    data = message['data']
    if data == KILL_SIGNAL:
        logger.info('Killing worker')
        state['arch_data'] = None
        state['specified'] = True
        comm.finish_processing(
            get_topic_name(ARCH_TOPIC, config), message, success=False)
    else:
        logger.info('Specifying architecture data %s', str(data))
        state['arch_data'] = data
        state['evaluated'] = False
        state['specified'] = True
        while not state['evaluated']:
            time.sleep(5)
        comm.finish_processing(get_topic_name(ARCH_TOPIC, config), message)


def nudge_master(comm, config, state):
    time.sleep(10)
    if not state['started']:
        comm.publish(get_topic_name(RESULTS_TOPIC, config), PUBLISH_SIGNAL)


def main():
    comm, search_space_factory, evaluator, config = process_config_and_args()

    search_data_folder = sl.get_search_data_folderpath(config['search_folder'],
                                                       config['search_name'])
    state = {
        'specified': False,
        'evaluated': False,
        'arch_data': None,
        'started': False
    }

    comm.subscribe(
        get_topic_name(ARCH_TOPIC, config),
        callback=lambda message: retrieve_message(message, comm, config, state))
    thread = threading.Thread(target=nudge_master, args=(comm, config, state))
    thread.start()
    step = 0
    while True:
        while not state['specified']:
            time.sleep(5)
        if state['arch_data']:
            vs, evaluation_id, searcher_eval_token = state['arch_data']
            logger.info('Evaluating architecture %d', evaluation_id)
            inputs, outputs = search_space_factory.get_search_space()
            se.specify(outputs.values(), vs)
            results = evaluator.eval(inputs, outputs)
            logger.info('Finished evaluating architecture %d', evaluation_id)
            step += 1
            if step % config['save_every'] == 0:
                logger.info('Saving evaluator state')
                evaluator.save_state(search_data_folder)

            encoded_results = [results, vs, evaluation_id, searcher_eval_token]
            comm.publish(get_topic_name(RESULTS_TOPIC, config), encoded_results)
            state['evaluated'] = True
            state['specified'] = False
        else:
            break
    thread.join()
    comm.unsubscribe(get_topic_name(ARCH_TOPIC, config))


if __name__ == "__main__":
    main()
