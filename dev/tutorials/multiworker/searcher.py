# Main/Searcher
import deep_architect.searchers.random as se
import deep_architect.core as co
import deep_architect.search_logging as log
import deep_architect.utils as ut
from search_space import dnn_net
import argparse

import keras.losses


def get_search_space(num_classes):

    def fn():
        co.Scope.reset_default_scope()
        inputs, outputs = dnn_net(num_classes)
        return inputs, outputs, {}

    return fn


def main(config):

    searcher = se.RandomSearcher(get_search_space(
        config.num_classes))  # random searcher
    # create a logging folder to log information (config and features)
    logger = log.SearchLogger(
        'logs',
        config.exp_name,
        resume_if_exists=True,
        create_parent_folders=True)
    # return values
    architectures = dict()

    for i in range(int(config.num_samples)):
        print("Sampling architecture %d" % i)
        inputs, outputs, h_value_hist, searcher_eval_token = searcher.sample()
        eval_logger = logger.get_current_evaluation_logger()
        eval_logger.log_config(h_value_hist, searcher_eval_token)
        eval_logger.log_features(inputs, outputs)
        architectures[i] = {
            'config_filepath': eval_logger.config_filepath,
            'evaluation_filepath': eval_logger.get_evaluation_folderpath()
        }

    # write to a json file to communicate with master
    ut.write_jsonfile(architectures, config.result_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--num_classes',
        type=int,
        default='',
        required=True,
        help='Number of classes to predict')
    parser.add_argument(
        '--num_samples',
        type=int,
        default='',
        required=True,
        help='Number of architecture to sample')
    parser.add_argument(
        '--exp_name',
        type=str,
        default='',
        required=True,
        help='Name of the experiment')
    parser.add_argument(
        '--result_fp',
        type=str,
        default='',
        required=True,
        help='Filepath of resulting architecture information')
    args = parser.parse_args()
    main(args)
