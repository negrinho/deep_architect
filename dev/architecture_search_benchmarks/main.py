from darch.contrib.useful.datasets.loaders import load_cifar10
from darch.contrib.useful.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from darch.contrib.useful.datasets.dataset import InMemoryDataset
import darch.search_logging as sl
import search_spaces.search_space_factory as ssf
import searchers.searcher as se

def main():
    num_classes = 10
    num_samples = 200
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = load_cifar10('data/cifar10/cifar-10-batches-py/')
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)
    evaluator = SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes,
                    './temp', max_num_training_epochs=4, log_output_to_terminal=True,
                    test_dataset=test_dataset)

    search_logger = sl.SearchLogger('./logs', 'test', resume_if_exists=True)
    search_data_path = sl.join_paths([search_logger.search_data_folderpath, "searcher_state.json"])

    search_space_factory = ssf.name_to_search_space_factory_fn['zoph_sp1'](num_classes)
    searcher = se.name_to_searcher_fn['evolution_pop=20_samp=20_reg=t'](search_space_factory.get_search_space)

    if sl.file_exists(search_data_path):
        state = sl.read_jsonfile(search_data_path)
        searcher.load(state)

    for i in xrange(search_logger.current_evaluation_id, num_samples):
        evaluation_logger = search_logger.get_current_evaluation_logger()
        inputs, outputs, hs, vs, searcher_eval_token = searcher.sample()
        evaluation_logger.log_config(vs, searcher_eval_token)
        evaluation_logger.log_features(inputs, outputs, hs)
        results = evaluator.eval(inputs, outputs, hs)
        evaluation_logger.log_results(results)
        print('Sample %d: %f' % (i, results['validation_accuracy']))
        searcher.update(results['validation_accuracy'], searcher_eval_token)
        searcher.save_state(search_logger.search_data_folderpath)

if __name__ == '__main__':
    main()
