
from deep_architect.contrib.misc.datasets.loaders import load_mnist
from deep_architect.contrib.misc.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from deep_architect.contrib.misc.datasets.dataset import InMemoryDataset
import deep_architect.contrib.misc.search_spaces.tensorflow.dnn as css_dnn
import deep_architect.modules as mo
from deep_architect.searchers.random import RandomSearcher

class SSF0(mo.SearchSpaceFactory):
    def __init__(self, num_classes):
        mo.SearchSpaceFactory.__init__(self)
        self.num_classes = num_classes

    def _get_search_space(self):
        inputs, outputs = css_dnn.dnn_net(self.num_classes)
        return inputs, outputs, {}

def main():
    num_classes = 10
    num_samples = 16
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = load_mnist('data/mnist')
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)
    evaluator = SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes,
        './temp', max_eval_time_in_minutes=1.0, log_output_to_terminal=True)
    search_space_factory = SSF0(num_classes)

    searcher = RandomSearcher(search_space_factory.get_search_space)
    for _ in xrange(num_samples):
        inputs, outputs, hs, _, searcher_eval_token = searcher.sample()
        val_acc = evaluator.eval(inputs, outputs, hs)['validation_accuracy']
        searcher.update(val_acc, searcher_eval_token)

if __name__ == '__main__':
    main()