from darch.contrib.datasets.loaders import load_cifar10
from darch.contrib.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from darch.contrib.datasets.dataset import InMemoryDataset
#import darch.contrib.search_spaces.tensorflow.dnn as css_dnn
import darch.searchers as se
import darch.search_space as ss
import darch.core as co

class SearchSpaceFactory:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def get_search_space(self):
        co.Scope.reset_default_scope()
        inputs, outputs = ss.get_search_space_1(self.num_classes)
        return inputs, outputs, {}

def mutatable(h):
    return h.get_name().startswith('H.Mutatable')
        
def main():
    num_classes = 10
    num_samples = 10
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = load_cifar10('data/cifar10')
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)
    evaluator = SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes, 
                    './temp', max_num_training_epochs=4, log_output_to_terminal=True, 
                    test_dataset=test_dataset)
    search_space_factory = SearchSpaceFactory(num_classes)

    searcher = se.EvolutionSearcher(search_space_factory.get_search_space, mutatable, 20, 20, regularized=True)
    for i in xrange(num_samples):
        inputs, outputs, hs, _, searcher_eval_token = searcher.sample()
        val_acc = evaluator.eval(inputs, outputs, hs)['val_acc']
        print('Sample %d: %f', (i, val_acc))
        searcher.update(val_acc, searcher_eval_token)

if __name__ == '__main__':
    main()
