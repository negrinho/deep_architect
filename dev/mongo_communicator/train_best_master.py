import os
import argparse
import time
import subprocess
import logging

from deep_architect import search_logging as sl
from deep_architect import utils as ut

from dev.mongo_communicator.mongo_communicator import MongoCommunicator

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
        default='/darch/dev/mongo_communicator/train_best_config.json')
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
    parser.add_argument('--repetition', default=0)
    options = parser.parse_args()
    configs = ut.read_jsonfile(options.config_file)
    config = configs[options.config_name]

    config['bucket'] = options.bucket

    comm = MongoCommunicator(host=options.mongo_host,
                             port=options.mongo_port,
                             data_refresher=True)

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
    config['results_file'] = os.path.join(
        config['results_prefix'] + '_' + str(options.repetition),
        config['results_file'])
    state = {'finished': 0, 'best_accuracy': 0.0}
    if options.resume:
        try:
            download_folder(search_data_folder, config['full_search_folder'],
                            config['bucket'])
            searcher.load_state(search_data_folder)
            if ut.file_exists(config['save_filepath']):
                old_state = ut.read_jsonfile(config['save_filepath'])
                state['finished'] = old_state['finished']
                state['best_accuracy'] = old_state['best_accuracy']
        except:
            pass

    return comm, search_logger, state, config


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


def update_searcher(message, comm, search_logger, state, config):
    data = message['data']
    if not data == PUBLISH_SIGNAL:
        results = data['results']
        vs = data['vs']
        evaluation_id = data['evaluation_id']
        searcher_eval_token = data['searcher_eval_token']
        log_results(results, vs, evaluation_id, searcher_eval_token,
                    search_logger, config)

        update_searcher_state(state, config, results)
        save_searcher_state(state, config, search_logger)

    maybe_publish_kill(comm, state, config)
    comm.finish_processing(get_topic_name(RESULTS_TOPIC, config), message)


def save_searcher_state(state, config, search_logger):
    logger.info('Models sampled: %d Best Accuracy: %f', state['finished'],
                state['best_accuracy'])
    state = {
        'finished': state['finished'],
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


def log_results(results, vs, evaluation_id, searcher_eval_token, search_logger,
                config):
    logger.info("Updating searcher with evaluation %d and results %s",
                evaluation_id, str(results))
    eval_logger = search_logger.get_evaluation_logger(evaluation_id)
    eval_logger.log_config(vs, searcher_eval_token)
    eval_logger.log_results(results)
    upload_folder(eval_logger.get_evaluation_folderpath(), config['eval_path'],
                  config['bucket'])


def maybe_publish_kill(comm, state, config):
    if should_end_searcher(state, config):
        logger.info('Search finished')
        # comm.publish(get_topic_name(ARCH_TOPIC, config), KILL_SIGNAL)
        state['search_finished'] = True


def should_end_searcher(state, config):
    return state['finished'] == state['models_sampled']


def train_best(comm, state, config):
    search_results = ut.read_jsonfile(config['results_file'])
    best_k = list(
        reversed(
            sorted(
                zip(search_results['validation_accuracies'],
                    search_results['configs']))))[:config['num_architectures']]
    model_id = 0
    for rank, (_, vs) in enumerate(best_k):
        for trial in range(config['num_trials']):
            for lr in [.1, .05, .025, .01, .005, .001]:
                for wd in [.0001, .0003, .0005]:
                    if not comm.check_data_exists(
                            get_topic_name(ARCH_TOPIC, config), 'evaluation_id',
                            model_id):
                        logger.info(
                            'Publishing trial %d for rank %d architecture',
                            trial, rank)
                        arch = {
                            'vs': vs,
                            'evaluation_id': model_id,
                            'searcher_eval_token': {},
                            'eval_hparams': {
                                'init_lr': lr,
                                'weight_decay': wd,
                                'lr_decay_method': 'cosine',
                                'optimizer_type': 'sgd_mom'
                            }
                        }
                        comm.publish(get_topic_name(ARCH_TOPIC, config), arch)
                    model_id += 1
    state['models_sampled'] = model_id
    comm.publish(get_topic_name(ARCH_TOPIC, config), KILL_SIGNAL)


def main():
    comm, search_logger, state, config = process_config_and_args()
    logger.info('Using config %s', str(config))
    logger.info('Current state %s', str(state))
    state['search_finished'] = False
    train_best(comm, state, config)
    comm.subscribe(
        get_topic_name(RESULTS_TOPIC, config),
        callback=lambda message: update_searcher(message, comm, search_logger,
                                                 state, config),
    )
    while not state['search_finished']:
        time.sleep(30)
    comm.unsubscribe(get_topic_name(RESULTS_TOPIC, config))


if __name__ == "__main__":
    main()
