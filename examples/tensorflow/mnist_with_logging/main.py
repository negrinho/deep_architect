
from deep_architect.contrib.useful.datasets.loaders import load_mnist
from deep_architect.contrib.useful.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from deep_architect.contrib.useful.datasets.dataset import InMemoryDataset
import deep_architect.contrib.useful.search_spaces.tensorflow.dnn as css_dnn
import deep_architect.searchers as se
import deep_architect.search_logging as sl
import deep_architect.visualization as vi
import deep_architect.modules as mo

class SSF(mo.SearchSpaceFactory):
    def __init__(self, num_classes):
        mo.SearchSpaceFactory.__init__(self)
        self.num_classes = num_classes

    def _get_search_space(self):
        inputs, outputs = css_dnn.dnn_net(self.num_classes)
        return inputs, outputs, {}

def main(config_filepath, key):
    # Loading the config file.
    cfg = sl.read_jsonfile(config_filepath)[key]
    num_classes = 10
    num_samples = cfg['num_samples']
    # Loading the data.
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = load_mnist('data/mnist')
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)
    # Creating up the evaluator.
    evaluator = SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes,
        './temp', max_eval_time_in_minutes=cfg['max_eval_time_in_minutes'],
        log_output_to_terminal=True, test_dataset=test_dataset)
    # Creating the search space.
    search_space_factory = SSF(num_classes)
    # Creating the searcher.
    searcher = se.RandomSearcher(search_space_factory.get_search_space)
    # Creating the search logger to log the results of the experiment.
    search_logger = sl.SearchLogger(cfg['folderpath'], cfg['search_name'],
        delete_if_exists=cfg['delete_if_exists'], resume_if_exists=cfg['resume_if_exists'],
        create_parent_folders=True)
    # Search loop.
    for _ in xrange(num_samples - search_logger.get_current_evaluation_id()):
        inputs, outputs, hs, hyperp_value_lst, searcher_eval_token = searcher.sample()
        results = evaluator.eval(inputs, outputs, hs)
        # Logging results (including graph).
        evaluation_logger = search_logger.get_current_evaluation_logger()
        evaluation_logger.log_config(hyperp_value_lst, searcher_eval_token)
        evaluation_logger.log_features(inputs, outputs, hs)
        evaluation_logger.log_results(results)
        vi.draw_graph(outputs.values(), True, True, print_to_screen=False,
            out_folderpath=evaluation_logger.get_user_data_folderpath())
        # Updating the searcher given the results of logging.
        searcher.update(results['validation_accuracy'], searcher_eval_token)

if __name__ == '__main__':
    cmd = sl.CommandLineArgs()
    cmd.add('config_filepath', 'str', 'examples/tensorflow/mnist_with_logging/config.json', True)
    cmd.add('key', 'str')
    d_cmd = cmd.parse()
    main(**d_cmd)