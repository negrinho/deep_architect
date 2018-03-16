import darch.searchers as se
import darch.contrib.search_spaces.tensorflow.dnn as css_dnn
from create_sentiment_featuresets import create_feature_sets_and_labels
from darch.contrib.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from darch.contrib.datasets.dataset import InMemoryDataset
from darch.contrib.search_spaces.tensorflow.common import D
import darch.core as co
import numpy as np

def ss1_fn():
    co.Scope.reset_default_scope()
    inputs, outputs = css_dnn.dnn_net(2)
    return inputs, outputs, {'learning_rate_init' : D([1e-2, 1e-3, 1e-4, 1e-5])}

def main():
    num_classes = 2
    Xtrain, ytrain, Xtest, ytest = create_feature_sets_and_labels(
        'data/sentiment/small_pos.txt', 'data/sentiment/small_neg.txt')
    Xtrain, Xtest = np.array(Xtrain, dtype=np.float), np.array(Xtest, dtype=np.float)
    ytrain, ytest = np.array(ytrain, dtype=np.int), np.array(ytest, dtype=np.int)
    train_size = int(len(Xtrain)*.8)

    Xtrain, Xval = Xtrain[:train_size], Xtrain[train_size:]
    ytrain, yval = ytrain[:train_size], ytrain[train_size:]
    
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)
    evaluator = SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes, 
        './temp', max_eval_time_in_minutes=1.0, log_output_to_terminal=True,
        batch_size=256)

    searcher = se.RandomSearcher(ss1_fn)
    for _ in xrange(128):
        (inputs, outputs, hs, hyperp_value_hist, searcher_eval_token) = searcher.sample()
        val_acc = evaluator.eval(inputs, outputs, hs)['val_acc']
        print hyperp_value_hist, val_acc, searcher_eval_token
        searcher.update(val_acc, searcher_eval_token)

if __name__ == '__main__':
    main()
