
from darch.contrib.datasets.loaders import load_mnist
from darch.contrib.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from darch.contrib.datasets.dataset import InMemoryDataset
import darch.contrib.search_spaces.tensorflow.dnn as dnn
import darch.contrib.search_spaces.tensorflow.cnn2d as cnn2d
import darch.searchers as se
import darch.surrogates as su
import darch.search_logging as sl
import darch.visualization as vi
import darch.modules as mo
from six import iteritems

class SSF0(mo.SearchSpaceFactory):
    def __init__(self, num_classes):
        mo.SearchSpaceFactory.__init__(self)
        self.num_classes = num_classes

    def _get_search_space(self):
        inputs, outputs = dnn.dnn_net(self.num_classes)
        return inputs, outputs, {}

def run_searcher(searcher, evaluator, num_samples, search_logger):
    for _ in xrange(num_samples):
        evaluation_logger = search_logger.get_current_evaluation_logger()
        inputs, outputs, hs, hyperp_value_lst, searcher_eval_token = searcher.sample()
        evaluation_logger.log_config(hyperp_value_lst, searcher_eval_token)
        evaluation_logger.log_features(inputs, outputs, hs)
        results = evaluator.eval(inputs, outputs, hs)
        evaluation_logger.log_results(results)
        vi.draw_graph(outputs.values(), True, True, print_to_screen=False,
            out_folderpath=evaluation_logger.get_user_data_folderpath())
        searcher.update(results['validation_accuracy'], searcher_eval_token)

# NOTE: This is wrong because of the creation of the models.
def main():
    num_classes = 10
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = load_mnist('data/mnist')
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)

    num_samples = 128
    num_repetitions = 3
    # basic benchmark based on MNIST.
    for rep_i in xrange(num_repetitions):
        ssf_lst = [SSF0(num_classes).get_search_space]
        ssf_name_lst = ['dnn_net']

        for ssf, ssf_name in zip(ssf_lst, ssf_name_lst):
            evaluator_lst = []
            evaluator_name_lst = []
            for max_eval_time in [0.5, 2.0]:
                evl = SimpleClassifierEvaluator(
                    train_dataset, val_dataset, num_classes,
                    './temp', max_eval_time_in_minutes=max_eval_time,
                    log_output_to_terminal=True, test_dataset=test_dataset)
                evl_name = 'ev:%0.1f' % max_eval_time
                evaluator_lst.append(evl)
                evaluator_name_lst.append(evl_name)

            for evaluator, evaluator_name in zip(evaluator_lst, evaluator_name_lst):
                searcher_lst = [
                    se.RandomSearcher(ssf),
                    se.SMBOSearcher(ssf, su.HashingSurrogate(2048, 1), 256, 0.1),
                    se.SMBOSearcher(ssf, su.HashingSurrogate(2048, 1), 512, 0.1),
                    se.SMBOSearcherWithMCTSOptimizer(ssf, su.HashingSurrogate(2048, 1), 256, 0.1, 1),
                    se.SMBOSearcherWithMCTSOptimizer(ssf, su.HashingSurrogate(2048, 1), 512, 0.1, 1)]
                searcher_name_lst = [
                    'random', 'smbo_rand_256', 'smbo_rand_512', 'smbo_mcts_256', 'smbo_mcts_512']

                for searcher, searcher_name in zip(searcher_lst, searcher_name_lst):
                    search_name = "to_rm_%s-%s-%s-%d" % (ssf_name, searcher_name, evaluator_name, rep_i)
                    search_logger = sl.SearchLogger('./logs', search_name)
                    run_searcher(searcher, evaluator, num_samples, search_logger)

if __name__ == '__main__':
    main()