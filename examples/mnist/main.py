from deep_architect.contrib.misc.datasets.loaders import load_mnist
from deep_architect.contrib.misc.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from deep_architect.contrib.misc.datasets.dataset import InMemoryDataset
import deep_architect.contrib.misc.search_spaces.tensorflow.dnn as css_dnn
import deep_architect.modules as mo
from deep_architect.searchers.random import RandomSearcher


def main():
    num_classes = 10
    num_samples = 8
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = load_mnist('data/mnist')
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)
    evaluator = SimpleClassifierEvaluator(
        train_dataset,
        val_dataset,
        num_classes,
        './temp',
        max_eval_time_in_minutes=1.0,
        log_output_to_terminal=True)

    search_space_fn = lambda: css_dnn.dnn_net(num_classes)
    searcher = RandomSearcher(search_space_fn)
    for _ in range(num_samples):
        inputs, outputs, searcher_eval_token, _ = searcher.sample()
        val_acc = evaluator.eval(inputs, outputs)['validation_accuracy']
        searcher.update(val_acc, searcher_eval_token)


if __name__ == '__main__':
    main()
