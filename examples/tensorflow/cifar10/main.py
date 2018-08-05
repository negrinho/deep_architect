import deep_architect.search_logging as sl

# Make sure that only one GPU is visible.
if __name__ == '__main__':
    cmd = sl.CommandLineArgs()
    cmd.add('config_filepath', 'str', 'examples/tensorflow/cifar10/config.json', True)
    cmd.add('key', 'str')
    d_cmd = cmd.parse()
    cfg = sl.read_jsonfile(d_cmd['config_filepath'])[d_cmd['key']]

    if cfg['use_gpu']:
        import deep_architect.contrib.useful.gpu_utils as gpu_utils
        gpu_id = gpu_utils.get_available_gpu(0.1, 5.0)
        print "Using GPU %d" % gpu_id
        assert gpu_id is not None
        gpu_utils.set_visible_gpus([gpu_id])

from deep_architect.contrib.useful.datasets.loaders import load_cifar10
from deep_architect.contrib.useful.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from deep_architect.contrib.useful.datasets.dataset import InMemoryDataset
import deep_architect.contrib.useful.search_spaces.tensorflow.cnn2d as css_cnn2d
import deep_architect.contrib.useful.search_spaces.tensorflow.dnn as css_dnn
import deep_architect.modules as mo
import deep_architect.searchers as se
import deep_architect.hyperparameters as hp
import deep_architect.visualization as vi

D = hp.Discrete

class SSF0(mo.SearchSpaceFactory):
    def __init__(self, num_classes):
        mo.SearchSpaceFactory.__init__(self)
        self.num_classes = num_classes

    def _get_search_space(self):
        h_num_spatial_reductions = D([2, 3, 4])
        h_pool_op = D(['max', 'avg'])
        inputs, outputs = css_dnn.dnn_net(self.num_classes)
        # inputs, outputs = mo.siso_sequential([
        #     css_cnn2d.conv_net(h_num_spatial_reductions),
        #     css_cnn2d.spatial_squeeze(h_pool_op, D([self.num_classes]))])
        hyperps = {
            'learning_rate_init' : D([1e-2, 1e-3, 1e-4, 1e-5]),
            'optimizer_type' : D(['adam', 'sgd_mom'])}
        return inputs, outputs, hyperps

def main(cfg):
    num_classes = 10
    num_samples = cfg['num_samples']
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = load_cifar10('data/cifar10/cifar-10-batches-py/')
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)

    search_logger = sl.SearchLogger(cfg['folderpath'], cfg['search_name'],
        create_parent_folders=True, resume_if_exists=cfg['resume_if_exists'],
        delete_if_exists=cfg['delete_if_exists'])
    evaluator = SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes,
        model_path=search_logger.get_search_data_folderpath() + '/current_model',
        max_eval_time_in_minutes=cfg['max_eval_time_in_minutes'],
        log_output_to_terminal=True, test_dataset=test_dataset)

    search_space_factory = SSF0(num_classes)
    searcher = se.RandomSearcher(search_space_factory.get_search_space)
    for _ in xrange(num_samples - search_logger.get_current_evaluation_id()):
        evaluation_logger = search_logger.get_current_evaluation_logger()
        inputs, outputs, hs, hyperp_value_lst, searcher_eval_token = searcher.sample()
        results = evaluator.eval(inputs, outputs, hs)
        evaluation_logger.log_config(hyperp_value_lst, searcher_eval_token)
        evaluation_logger.log_features(inputs, outputs, hs)
        evaluation_logger.log_results(results)
        vi.draw_graph(outputs.values(), True, True, print_to_screen=False,
            out_folderpath=evaluation_logger.get_user_data_folderpath())
        searcher.update(results['validation_accuracy'], searcher_eval_token)

if __name__ == '__main__':
    main(cfg)
