import argparse
import pickle
import json
import time
import deep_architect.utils as ut
from google.cloud import pubsub_v1

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
publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()
results_topic = publisher.topic_path(PROJECT_ID, 'results')
arch_subscription = subscriber.subscription_path(PROJECT_ID, 'architectures')


configs = ut.read_jsonfile("./examples/tensorflow/full_benchmarks/experiment_config.json")

parser = argparse.ArgumentParser("MPI Job for architecture search")
parser.add_argument('--config', '-c', action='store', dest='config_name',
default='normal')

# Other arguments
parser.add_argument('--display-output', '-o', action='store_true', dest='display_output',
                    default=False)
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
train_dataset = InMemoryDataset(Xtrain, ytrain, True)
val_dataset = InMemoryDataset(Xval, yval, False)
test_dataset = InMemoryDataset(Xtest, ytest, False)

evaluators = {
    'simple_classification': lambda: SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes,
                './temp', max_num_training_epochs=config['eval_epochs'], log_output_to_terminal=options.display_output,
                test_dataset=test_dataset),
    'enas_evaluator': lambda: ENASEvaluator(train_dataset, val_dataset, num_classes,
                search_space_factory.weight_sharer)
}

assert not config['evaluator'].startswith('enas') or hasattr(search_space_factory, 'weight_sharer')
evaluator = evaluators[config['evaluator']]()

# start_worker(comm, evaluator, search_space_factory, config['search_folder'],
#     config['search_name'], resume=options.resume, save_every=save_every)
step = 0
sl.create_search_folderpath(config['search_folder'], config['search_name'])
search_data_folder = sl.get_search_data_folderpath(config['search_folder'], config['search_name'])
finished = False
working = False
def eval(message):
    global step
    global finished
    global working = True
    data = json.loads(message.data.decode('utf-8'))
    if data is 'kill':
        finished = True
        message.nack()
        subscription.cancel()
    else:
        vs, evaluation_id, searcher_eval_token = data
        inputs, outputs, hs = search_space_factory.get_search_space()
        se.specify(outputs.values(), hs, vs)
        results = evaluator.eval(inputs, outputs, hs)
        step += 1
        # if 'max_steps_for_evaluator' in config and step == config['max_steps_for_evaluator']:
        #     finished = True
        if step % save_every == 0:
            evaluator.save_state(search_data_folder)

        encoded_results = json.dumps((results, evaluation_id, searcher_eval_token)).encode('utf-8')
        future = publisher.publish(results_topic, encoded_results)
        future.result()
        message.ack()

subscription = subscriber.subscribe(arch_subscription, callback=eval)

time.sleep(10)
if not working:
    encoded_results = json.dumps('publish').encode('utf-8')
    future = publisher.publish(results_topic, encoded_results)
    future.result()

while not finished:
    time.sleep(30)


# if __name__ == "__main__":
#     main()
