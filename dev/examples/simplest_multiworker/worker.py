from deep_architect.contrib.misc.datasets.loaders import load_mnist
from deep_architect.contrib.misc.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from deep_architect.contrib.misc.datasets.dataset import InMemoryDataset
import deep_architect.search_logging as sl
import deep_architect.visualization as vi
import deep_architect.utils as ut
from deep_architect.searchers.common import specify
import search_space as ss


def main():
    cmd = ut.CommandLineArgs()
    cmd.add('config_filepath', 'str')
    cmd.add('worker_id', 'int')
    cmd.add('num_workers', 'int')
    out = cmd.parse()
    cfg = ut.read_jsonfile(out['config_filepath'])

    # Loading the data.
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = load_mnist('data/mnist')
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)

    # Creating up the evaluator.
    evaluator = SimpleClassifierEvaluator(
        train_dataset,
        val_dataset,
        ss.num_classes,
        './temp/worker%d' % out["worker_id"],
        max_eval_time_in_minutes=cfg['max_eval_time_in_minutes'],
        log_output_to_terminal=True,
        test_dataset=test_dataset)

    for evaluation_id in range(out["worker_id"], cfg["num_samples"],
                               out["num_workers"]):
        logger = sl.EvaluationLogger(
            cfg["folderpath"],
            cfg["search_name"],
            evaluation_id,
            abort_if_notexists=True)
        if not logger.results_exist():
            eval_cfg = logger.read_config()
            inputs, outputs = ss.search_space_fn()
            specify(outputs, eval_cfg["hyperp_value_lst"])
            results = evaluator.eval(inputs, outputs)
            logger.log_results(results)


if __name__ == '__main__':
    main()