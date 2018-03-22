
from darch.contrib.datasets.loaders import load_cifar10
from darch.contrib.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from darch.contrib.datasets.dataset import InMemoryDataset
import darch.contrib.search_spaces.tensorflow.cnn2d as css_cnn2d
import darch.contrib.search_spaces.tensorflow.dnn as css_dnn
import darch.modules as mo
import darch.searchers as se
import darch.hyperparameters as hp
import darch.search_logging as sl
import darch.visualization as vi

D = hp.Discrete

class SSF0(mo.SearchSpaceFactory):
    def __init__(self, num_classes):
        mo.SearchSpaceFactory.__init__(self)
        self.num_classes = num_classes

    def _get_search_space(self):
        # h_num_spatial_reductions = D([2, 3, 4])
        # h_pool_op = D(['max', 'avg'])
        # # inputs, outputs = css_dnn.dnn_net(self.num_classes)
        # inputs, outputs = mo.siso_sequential([
        #     css_cnn2d.conv_net(h_num_spatial_reductions),
        #     css_cnn2d.spatial_squeeze(h_pool_op, D([self.num_classes]))])
        inputs, outputs = css_dnn.dnn_net(self.num_classes)
        hyperps = {
            'learning_rate_init' : D([1e-2, 1e-3, 1e-4, 1e-5]),
            'optimizer_type' : D(['adam', 'sgd_mom'])}
        return inputs, outputs, hyperps

def main():
    num_classes = 10
    num_samples = 16
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = load_cifar10('data/cifar10/cifar-10-batches-py/')
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)
    evaluator = SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes,
        './temp', max_eval_time_in_minutes=1.0, log_output_to_terminal=True,
        test_dataset=test_dataset)
    search_space_factory = SSF0(num_classes)

    searcher = se.RandomSearcher(search_space_factory.get_search_space)
    search_logger = sl.SearchLogger('./logs', 'cifar10_conv', delete_if_exists=True)
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

if __name__ == '__main__':
    main()
