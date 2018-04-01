from darch.contrib.datasets.loaders import load_cifar10
from darch.contrib.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from darch.contrib.datasets.dataset import InMemoryDataset
#import darch.contrib.search_spaces.tensorflow.dnn as css_dnn
import darch.searchers as se
import darch.contrib.search_spaces.tensorflow.evolution_search_space as ss
import darch.search_logging as sl
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
    num_samples = 200
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = load_cifar10('data/cifar10')
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)
    evaluator = SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes, 
                    './temp', max_num_training_epochs=4, log_output_to_terminal=True, 
                    test_dataset=test_dataset)

    search_logger = sl.SearchLogger('./logs', 'test')
    search_data_path = sl.join_paths([search_logger.search_data_folderpath, "searcher_state.json"])

    search_space_factory = SearchSpaceFactory(num_classes)
    searcher = se.EvolutionSearcher(search_space_factory.get_search_space, mutatable, 20, 20, regularized=True)
    
    if sl.file_exists(search_data_path):
        state = sl.read_jsonfile(search_data_path)
        searcher.load(state)
    
    for i in xrange(num_samples):
        evaluation_logger = search_logger.get_current_evaluation_logger()
        inputs, outputs, hs, vs, searcher_eval_token = searcher.sample()
        evaluation_logger.log_config(vs, searcher_eval_token)
        evaluation_logger.log_features(inputs, outputs, hs)
        results = evaluator.eval(inputs, outputs, hs)
        results['epochs'] = 4
        evaluation_logger.log_results(results)      
        print('Sample %d: %f', (i, results['val_acc']))
        searcher.update(results['val_acc'], searcher_eval_token)
        searcher.save_state(search_logger.search_data_folderpath)

if __name__ == '__main__':
    main()
