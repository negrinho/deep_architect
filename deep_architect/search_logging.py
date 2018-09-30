import json
import os
import shutil
import time
import subprocess
import deep_architect.utils as ut
from deep_architect.surrogates.common import extract_features
from six import iteritems, itervalues


def get_search_folderpath(folderpath, search_name):
    return ut.join_paths([folderpath, search_name])


def get_search_data_folderpath(folderpath, search_name):
    return ut.join_paths(
        [get_search_folderpath(folderpath, search_name), "search_data"])


def get_all_evaluations_folderpath(folderpath, search_name):
    return ut.join_paths(
        [get_search_folderpath(folderpath, search_name), 'evaluations'])


def get_evaluation_folderpath(folderpath, search_name, evaluation_id):
    return ut.join_paths([
        get_all_evaluations_folderpath(folderpath, search_name),
        'x%d' % evaluation_id
    ])


def get_evaluation_data_folderpath(folderpath, search_name, evaluation_id):
    return ut.join_paths([
        get_evaluation_folderpath(folderpath, search_name, evaluation_id),
        "eval_data"
    ])


def create_search_folderpath(folderpath,
                             search_name,
                             abort_if_exists=False,
                             delete_if_exists=False,
                             create_parent_folders=False):
    assert not (abort_if_exists and delete_if_exists)

    search_folderpath = get_search_folderpath(folderpath, search_name)
    search_data_folderpath = get_search_data_folderpath(
        folderpath, search_name)
    all_evaluations_folderpath = get_all_evaluations_folderpath(
        folderpath, search_name)

    if delete_if_exists:
        ut.delete_folder(search_folderpath, False, False)
    assert not (abort_if_exists and ut.folder_exists(search_folderpath))

    if not ut.folder_exists(search_folderpath):
        ut.create_folder(
            search_folderpath, create_parent_folders=create_parent_folders)
        ut.create_folder(search_data_folderpath)
        ut.create_folder(all_evaluations_folderpath)


class EvaluationLogger:
    """Evaluation logger for a simple evaluation.

    The logging is divided into three parts: config, features, and results.
    All three parts are represented as JSON files in disk, i.e., dictionaries.
    The config JSON encodes the architecture to be evaluated. This encoding is
    tied to the search space the evaluation was drawn from, and it can be used
    to reproduce the architecture to be evaluated given the search space.

    The features JSON contains a string representation of the architecture that
    we can use along with the information in the results to train a model that
    predicts the performance of an architecture. This is useful if the
    evaluation used to collect the results is very expensive. See also
    :func:`deep_architect.surrogates.common.extract_features`.

    The results JSON contains the results of the evaluating the particular
    architecture. In the case of deep learning, this often involves training the
    architecture for a given task on a training set and evaluating it on a
    validation set.

    Args:
        all_evaluations_folderpath (str): Path to the folder where all the
            evaluation log folders lie. This folder is managed by the search
            logger.

        evaluation_id (int): Number of the evaluation with which the logger is
            associated with. The numbering starts at zero.
    """

    def __init__(self,
                 folderpath,
                 search_name,
                 evaluation_id,
                 abort_if_exists=False,
                 abort_if_notexists=False):

        self.evaluation_folderpath = get_evaluation_folderpath(
            folderpath, search_name, evaluation_id)
        self.evaluation_data_folderpath = get_evaluation_data_folderpath(
            folderpath, search_name, evaluation_id)

        assert (not abort_if_exists) or (not ut.folder_exists(
            self.evaluation_folderpath))
        assert (not abort_if_notexists) or ut.folder_exists(
            self.evaluation_folderpath)
        ut.create_folder(
            self.evaluation_folderpath,
            abort_if_exists=abort_if_exists,
            create_parent_folders=True)
        ut.create_folder(
            self.evaluation_data_folderpath,
            abort_if_exists=abort_if_exists,
            create_parent_folders=True)

        self.config_filepath = ut.join_paths(
            [self.evaluation_folderpath, 'config.json'])
        self.features_filepath = ut.join_paths(
            [self.evaluation_folderpath, 'features.json'])
        self.results_filepath = ut.join_paths(
            [self.evaluation_folderpath, 'results.json'])

    def log_config(self, hyperp_value_lst, searcher_evaluation_token):
        """Logs a config JSON that describing the evaluation to be done.

        The config JSON has the list with the ordered sequence of hyperparameter
        values that allow to replicate the same evaluation given the same
        search space; the searcher evaluation token, that can be given back to
        the same searcher allowing it to update back its state. The searcher
        evaluation token is returned by the searcher when a new architecture
        to evaluate is sampled. See, for example,
        :meth:`deep_architect.searchers.MCTSSearcher.sample`. The format of the searcher
        evaluation token is searcher dependent, but it should be JSON serializable
        in all cases.

        Creates ``config.json`` in the evaluation log folder.

        Args:
            hyperp_value_lst (list[object]): List with the sequence of JSON
                serializable hyperparameter values that define the architecture
                to evaluate.
            searcher_evaluation_token (dict[str, object]): Dictionary that is
                JSON serializable and it is enough, when given back to the
                searcher along with the results, for the searcher to update
                its state appropriately.
        """
        assert not ut.file_exists(self.config_filepath)
        config_d = {
            'hyperp_value_lst': hyperp_value_lst,
            'searcher_evaluation_token': searcher_evaluation_token
        }
        ut.write_jsonfile(config_d, self.config_filepath)

    def read_config(self):
        assert ut.file_exists(self.config_filepath)
        return ut.read_jsonfile(self.config_filepath)

    def config_exists(self):
        return ut.file_exists(self.config_filepath)

    def log_results(self, results):
        """Logs the results of evaluating an architecture.

        The dictionary contains many metrics about the architecture..
        In machine learning, this often involves training the model on a training
        set and evaluating the model on a validation set. In domains different
        than machine learning, other forms of evaluation may make sense.

        Creates ``results.json`` in the evaluation log folder.

        Args:
            dict[object]: Dictionary of JSON serializable metrics and information
                about the evaluated architecture.
        """
        assert (not ut.file_exists(self.results_filepath))
        assert ut.file_exists(self.config_filepath) and ut.file_exists(
            self.features_filepath)
        assert isinstance(results, dict)
        ut.write_jsonfile(results, self.results_filepath)

    def read_results(self):
        assert ut.file_exists(self.results_filepath)
        return ut.read_jsonfile(self.results_filepath)

    def results_exist(self):
        return ut.file_exists(self.results_filepath)

    def get_evaluation_folderpath(self):
        """Path to the evaluation folder where all the standard evaluation
        logs (e.g., ``config.json``, ``features.json``, and ``results.json``)
        are written to.

        Only standard logging information about the evaluation should be written
        here. See
        :meth:`deep_architect.search_logging.EvaluationLogger.get_evaluation_data_folderpath`
        for a path to a folder that can
        be used to store non-standard user logging information.

        Returns:
            str:
                Path to the folder where the standard logs about the evaluation
                are written to.
        """
        return self.evaluation_folderpath

    def get_evaluation_data_folderpath(self):
        """Path to the user data folder where non-standard logging data can
        be stored.

        This is useful to store additional information about the evaluated
        model, e.g., example predictions of the model, model weights, or
        model predictions on the validation set.

        See :meth:`deep_architect.search_logging.EvaluationLogger.get_evaluation_folderpath`
        for the path for the standard JSON logs for an evaluation.

        Returns:
            str: Path to the folder where the evaluations logs are written to.
        """
        return self.evaluation_data_folderpath


def read_evaluation_folder(evaluation_folderpath):
    """Reads all the standard JSON log files associated to a single evaluation.

    See also :func:`deep_architect.search_logging.read_search_folder` for the function
    that reads all the evaluations in a search folder.

    Args:
        evaluation_folderpath (str): Path to the folder containing the standard
            JSON logs.

    Returns:
        dict[str,dict[str,object]]:
            Nested dictionaries with the logged information. The first
            dictionary has keys corresponding to the names of the standard
            log files.
    """
    assert ut.folder_exists(evaluation_folderpath)

    name_to_log = {}
    for name in ['config', 'features', 'results']:
        log_filepath = ut.join_paths([evaluation_folderpath, name + '.json'])
        name_to_log[name] = ut.read_jsonfile(log_filepath)
    return name_to_log


def read_search_folder(search_folderpath):
    """Reads all the standard JSON log files associated to a search experiment.

    See also :func:`deep_architect.search_logging.read_evaluation_folder` for the function
    that reads a single evaluation folder. The list of dictionaries is ordered
    in increasing order of evaluation id.

    Args:
        search_folderpath (str): Path to the search folder used for logging.

    Returns:
        list[dict[str,dict[str,object]]]:
            List of nested dictionaries with the logged information. Each
            dictionary in the list corresponds to an evaluation.
    """
    assert ut.folder_exists(search_folderpath)
    all_evaluations_folderpath = ut.join_paths(
        [search_folderpath, 'evaluations'])
    eval_id = 0
    log_lst = []
    while True:
        evaluation_folderpath = ut.join_paths(
            [all_evaluations_folderpath,
             'x%d' % eval_id])
        if ut.folder_exists(evaluation_folderpath):
            name_to_log = read_evaluation_folder(evaluation_folderpath)
            log_lst.append(name_to_log)
            eval_id += 1
        else:
            break
    return log_lst


# functionality below is useful to read search log folders that are nested.
# non-nested folders are preferred though.


def is_search_log_folder(folderpath):
    return ut.folder_exists(ut.join_paths([folderpath, 'evaluations', 'x0']))


def recursive_list_log_folders(folderpath):
    def _iter(p, lst):
        if is_search_log_folder(p):
            lst.append(p)
        else:
            for p_child in ut.list_folders(p):
                _iter(p_child, lst)

    log_folderpath_lst = []
    _iter(folderpath, log_folderpath_lst)
    return log_folderpath_lst


def recursive_read_search_folders(folderpath):
    all_log_lst = []
    for p in recursive_list_log_folders(folderpath):
        d = {'search_folderpath': p, 'log_lst': read_search_folder(p)}
        d['num_logs'] = len(d['log_lst'])
        all_log_lst.append(d)
    return all_log_lst
