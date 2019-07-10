import argparse
import time
import subprocess
import logging

from deep_architect import search_logging as sl
from deep_architect import utils as ut
from deep_architect.contrib.communicators.mongo_communicator import MongoCommunicator

from search_space_factory import name_to_search_space_factory_fn
from searcher import name_to_searcher_fn

logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
BUCKET_NAME = 'deep_architect'
RESULTS_TOPIC = 'results'
ARCH_TOPIC = 'architectures'
KILL_SIGNAL = 'kill'
PUBLISH_SIGNAL = 'publish'


def process_config_and_args():

    parser = argparse.ArgumentParser("MPI Job for architecture search")
    parser.add_argument('--config',
                        '-c',
                        action='store',
                        dest='config_name',
                        default='normal')
    parser.add_argument(
        '--config-file',
        action='store',
        dest='config_file',
        default='/darch/examples/contrib/kubernetes/experiment_config.json')
    parser.add_argument('--bucket',
                        '-b',
                        action='store',
                        dest='bucket',
                        default=BUCKET_NAME)

    # Other arguments
    parser.add_argument('--resume',
                        '-r',
                        action='store_true',
                        dest='resume',
                        default=False)
    parser.add_argument('--mongo-host',
                        '-m',
                        action='store',
                        dest='mongo_host',
                        default='127.0.0.1')
    parser.add_argument('--mongo-port',
                        '-p',
                        action='store',
                        dest='mongo_port',
                        default=27017)
    parser.add_argument('--log',
                        choices=['debug', 'info', 'warning', 'error'],
                        default='info')
    parser.add_argument('--repetition', default=0)
    options = parser.parse_args()

    numeric_level = getattr(logging, options.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % options.log)
    logging.getLogger().setLevel(numeric_level)

    configs = ut.read_jsonfile(options.config_file)
    config = configs[options.config_name]

    config['bucket'] = options.bucket

    comm = MongoCommunicator(host=options.mongo_host,
                             port=options.mongo_port,
                             data_refresher=True,
                             refresh_period=10)

    datasets = {
        'cifar10': ('data/cifar10/', 10),
    }

    _, num_classes = datasets[config['dataset']]
    search_space_factory = name_to_search_space_factory_fn[
        config['search_space']](num_classes)

    config['save_every'] = 1 if 'save_every' not in config else config[
        'save_every']
    searcher = name_to_searcher_fn[config['searcher']](
        search_space_factory.get_search_space)
    config['num_epochs'] = -1 if 'epochs' not in config else config['epochs']
    config['num_samples'] = -1 if 'samples' not in config else config['samples']

    # SET UP GOOGLE STORE FOLDER
    config['search_name'] = config['search_name'] + '_' + str(
        options.repetition)
    search_logger = sl.SearchLogger(config['search_folder'],
                                    config['search_name'])
    search_data_folder = search_logger.get_search_data_folderpath()
    config['save_filepath'] = ut.join_paths(
        (search_data_folder, config['searcher_file_name']))
    config['eval_path'] = sl.get_all_evaluations_folderpath(
        config['search_folder'], config['search_name'])
    config['full_search_folder'] = sl.get_search_folderpath(
        config['search_folder'], config['search_name'])
    config['eval_hparams'] = {} if 'eval_hparams' not in config else config[
        'eval_hparams']

    state = {
        'epochs': 0,
        'models_sampled': 0,
        'finished': 0,
        'best_accuracy': 0.0
    }
    if options.resume:
        try:
            download_folder(search_data_folder, config['full_search_folder'],
                            config['bucket'])
            searcher.load_state(search_data_folder)
            if ut.file_exists(config['save_filepath']):
                old_state = ut.read_jsonfile(config['save_filepath'])
                state['epochs'] = old_state['epochs']
                state['models_sampled'] = old_state['models_sampled']
                state['finished'] = old_state['finished']
                state['best_accuracy'] = old_state['best_accuracy']
        except:
            pass

    return comm, search_logger, searcher, state, config


def download_folder(folder, location, bucket):
    logger.info('Downloading gs://%s/%s to %s/', bucket, folder, location)
    subprocess.check_call([
        'gsutil', '-m', 'cp', '-r', 'gs://' + bucket + '/' + folder,
        location + '/'
    ])


def upload_folder(folder, location, bucket):
    subprocess.check_call([
        'gsutil', '-m', 'cp', '-r', folder,
        'gs://' + bucket + '/' + location + '/'
    ])


def get_topic_name(topic, config):
    return config['search_folder'] + '_' + config['search_name'] + '_' + topic


def update_searcher(message, comm, search_logger, searcher, state, config):
    data = message['data']
    if not data == PUBLISH_SIGNAL:
        results = data['results']
        vs = data['vs']
        evaluation_id = data['evaluation_id']
        searcher_eval_token = data['searcher_eval_token']

        log_results(results, vs, evaluation_id, searcher_eval_token,
                    search_logger, config)

        searcher.update(results['validation_accuracy'], searcher_eval_token)
        update_searcher_state(state, config, results)
        save_searcher_state(searcher, state, config, search_logger)

    publish_new_arch(comm, searcher, state, config)
    comm.finish_processing(get_topic_name(RESULTS_TOPIC, config), message)


def save_searcher_state(searcher, state, config, search_logger):
    logger.info('Models finished: %d Best Accuracy: %f', state['finished'],
                state['best_accuracy'])
    searcher.save_state(search_logger.get_search_data_folderpath())
    state = {
        'finished': state['finished'],
        'models_sampled': state['models_sampled'],
        'epochs': state['epochs'],
        'best_accuracy': state['best_accuracy']
    }
    ut.write_jsonfile(state, config['save_filepath'])
    upload_folder(search_logger.get_search_data_folderpath(),
                  config['full_search_folder'], config['bucket'])
    return state


def update_searcher_state(state, config, results):
    state['best_accuracy'] = max(state['best_accuracy'],
                                 results['validation_accuracy'])
    state['finished'] += 1
    state['epochs'] += config['eval_epochs']


def log_results(results, vs, evaluation_id, searcher_eval_token, search_logger,
                config):
    logger.info("Updating searcher with evaluation %d and results %s",
                evaluation_id, str(results))
    eval_logger = search_logger.get_evaluation_logger(evaluation_id)
    eval_logger.log_config(vs, searcher_eval_token)
    eval_logger.log_results(results)
    upload_folder(eval_logger.get_evaluation_folderpath(), config['eval_path'],
                  config['bucket'])


def publish_new_arch(comm, searcher, state, config):
    while comm.check_data_exists(get_topic_name(ARCH_TOPIC, config),
                                 'evaluation_id', state['models_sampled']):
        state['models_sampled'] += 1
    if should_end_searcher(state, config):
        logger.info('Search finished, sending kill signal')
        comm.publish(get_topic_name(ARCH_TOPIC, config), KILL_SIGNAL)
        state['search_finished'] = True
    elif should_continue(state, config):
        logger.info('Publishing architecture number %d',
                    state['models_sampled'])
        _, _, vs, searcher_eval_token = searcher.sample()
        arch = {
            'vs': vs,
            'evaluation_id': state['models_sampled'],
            'searcher_eval_token': searcher_eval_token,
            'eval_hparams': config['eval_hparams']
        }
        comm.publish(get_topic_name(ARCH_TOPIC, config), arch)
        state['models_sampled'] += 1


def should_continue(state, config):
    cont = config[
        'num_samples'] == -1 or state['models_sampled'] < config['num_samples']
    cont = cont and (config['num_epochs'] == -1 or
                     state['epochs'] < config['num_epochs'])
    return cont


def should_end_searcher(state, config):
    kill = config['num_samples'] != -1 and state['finished'] >= config[
        'num_samples']
    kill = kill or (config['num_epochs'] != -1 and
                    state['epochs'] >= config['num_epochs'])
    return kill


def main():
    comm, search_logger, searcher, state, config = process_config_and_args()
    logger.info('Using config %s', str(config))
    logger.info('Current state %s', str(state))
    state['search_finished'] = False
    comm.subscribe(get_topic_name(RESULTS_TOPIC, config),
                   callback=lambda message: update_searcher(
                       message, comm, search_logger, searcher, state, config))
    while not state['search_finished']:
        time.sleep(30)
    comm.unsubscribe(get_topic_name(RESULTS_TOPIC, config))


if __name__ == "__main__":
    main()
