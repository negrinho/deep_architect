import json
import os
import shutil
import time
import darch.surrogates as su

def read_jsonfile(filepath):
    with open(filepath, 'r') as f:
        d = json.load(f)
        return d

def write_jsonfile(d, filepath, sort_keys=False):
    with open(filepath, 'w') as f:
        json.dump(d, f, indent=4, sort_keys=sort_keys)

def read_textfile(filepath, strip=True):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        if strip:
            lines = [line.strip() for line in lines]
        return lines

def write_textfile(filepath, lines, append=False, with_newline=True):
    mode = 'a' if append else 'w'
    with open(filepath, mode) as f:
        for line in lines:
            f.write(line)
            if with_newline:
                f.write("\n")

def path_prefix(path):
    return os.path.split(path)[0]

def join_paths(paths):
    return os.path.join(*paths)

def path_exists(path):
    return os.path.exists(path)

def file_exists(path):
    return os.path.isfile(path)

def folder_exists(path):
    return os.path.isdir(path)

def create_folder(folderpath, abort_if_exists=True, create_parent_folders=False):
    assert not file_exists(folderpath)
    assert create_parent_folders or folder_exists(path_prefix(folderpath))
    assert not (abort_if_exists and folder_exists(folderpath))

    if not folder_exists(folderpath):
        os.makedirs(folderpath)

def delete_folder(folderpath, abort_if_nonempty=True, abort_if_notexists=True):
    assert folder_exists(folderpath) or (not abort_if_notexists)

    if folder_exists(folderpath):
        assert len(os.listdir(folderpath)) == 0 or (not abort_if_nonempty)
        shutil.rmtree(folderpath)
    else:
        assert not abort_if_notexists

def list_paths(folderpath,
        ignore_files=False, ignore_dirs=False,
        ignore_hidden_folders=True, ignore_hidden_files=True, ignore_file_exts=None,
        recursive=False, use_relative_paths=False):

    assert folder_exists(folderpath)

    path_list = []
    # enumerating all desired paths in a directory.
    for root, dirs, files in os.walk(folderpath):
        if ignore_hidden_folders:
            dirs[:] = [d for d in dirs if not d[0] == '.']
        if ignore_hidden_files:
            files = [f for f in files if not f[0] == '.']
        if ignore_file_exts != None:
            files = [f for f in files if not any([
                f.endswith(ext) for ext in ignore_file_exts])]

        # get only the path relative to this path.
        if not use_relative_paths:
            pref_root = root
        else:
            pref_root = os.path.relpath(root, folderpath)

        if not ignore_files:
            path_list.extend([join_paths([pref_root, f]) for f in files])
        if not ignore_dirs:
            path_list.extend([join_paths([pref_root, d]) for d in dirs])

        if not recursive:
            break
    return path_list

def list_files(folderpath,
        ignore_hidden_folders=True, ignore_hidden_files=True, ignore_file_exts=None,
        recursive=False, use_relative_paths=False):

    return list_paths(folderpath, ignore_dirs=True,
        ignore_hidden_folders=ignore_hidden_folders,
        ignore_hidden_files=ignore_hidden_files, ignore_file_exts=ignore_file_exts,
        recursive=recursive, use_relative_paths=use_relative_paths)

def list_folders(folderpath, ignore_hidden_folders=True,
        recursive=False, use_relative_paths=False):

    return list_paths(folderpath, ignore_files=True,
        ignore_hidden_folders=ignore_hidden_folders,
        recursive=recursive, use_relative_paths=use_relative_paths)

def convert_between_time_units(x, src_units='seconds', dst_units='hours'):
    d = {}
    d['seconds'] = 1.0
    d['minutes'] = 60.0 * d['seconds']
    d['hours'] = 60.0 * d['minutes']
    d['days'] = 24.0 * d['hours']
    d['weeks'] = 7.0 * d['days']
    d['miliseconds'] = d['seconds'] * 1e-3
    d['microseconds'] = d['seconds'] * 1e-6
    d['nanoseconds'] = d['seconds'] * 1e-9
    return (x * d[src_units]) / d[dst_units]

def convert_between_byte_units(x, src_units='bytes', dst_units='megabytes'):
    units = ['bytes', 'kilobytes', 'megabytes', 'gigabytes', 'terabytes']
    assert (src_units in units) and (dst_units in units)
    return x / float(
        2 ** (10 * (units.index(dst_units) - units.index(src_units))))

class SequenceTracker:
    """Useful to keep track of sequences for logging.

    Args:
        abort_if_different_lengths (bool, optional): If ``True``, the sequences
            being tracked must have the same length at all times.
    """
    def __init__(self, abort_if_different_lengths=False):
        self.abort_if_different_lengths = abort_if_different_lengths
        self.d = {}

    def append(self, d):
        """Append one additional data point to the sequences.

        If the dictionary contains keys that do not exist yet, a new sequence
        with that name is created.

        .. note::
            If ``abort_if_different_lengths`` is ``True``, the same set of
            keys has to be used for all calls to this method, or it will
            assert ``False``.

        Args:
            d (dict[str, object]): Dictionary where keys are the names of the
                sequences, and values are the additional data point to add to
                the sequence.
        """
        for k, v in d.iteritems():
            assert type(k) == str and len(k) > 0
            if k not in self.d:
                self.d[k] = []
            self.d[k].append(v)

        if self.abort_if_different_lengths:
            assert len(set([len(v) for v in self.d.itervalues()])) <= 1

    def get_dict(self):
        """Get a dictionary representation of the tracked sequences.

        Each sequence is represented as a list.

        Returns:
            dict[str, list[object]]:
                Dictionary with the sequences being tracked.
        """
        return dict(self.d)

class TimerManager:
    """Timer management class to create and extract information from multiple
    timers.

    Useful for getting time information for various timing events.
    """
    def __init__(self):
        self.init_time = time.time()
        self.last_registered = self.init_time
        self.name_to_timer = {}

    def create_timer(self, timer_name, abort_if_timer_exists=True):
        """Creates a named timer.

        The events ``tick`` and ``start`` are created with a new timer.

        Args:
            timer_name (str): Name of the timer to create.
            abort_if_timer_exists (bool, optional): If ``True`` and a timer with the same
                name exists, it asserts ``False``. If ``False``, it
                creates a new timer with the desired name, overwriting a
                potential existing timer with the same name.
        """
        assert not abort_if_timer_exists or timer_name not in self.name_to_timer
        start_time = time.time()
        self.name_to_timer[timer_name] = {'start' : start_time, 'tick' : start_time}

    def create_timer_event(self, timer_name, event_name, abort_if_event_exists=True):
        """Creates a named timer event associated to an existing timer.

        The named timer in which to register the event must exist. The
        event names ``tick`` and ``start`` are protected, and should not be used.
        The ``tick`` event is meant for events that happen periodically.

        Args:
            timer_name (str): Name of the timer to create the event in.
            event_name (str): Name of the event to create.
            abort_if_event_exists (bool, optional): If ``True`` and a timer event with the
                same name exists, it asserts ``False``. If ``False``, it
                creates a new event in the specified timer and name, overwriting a
                potential existing event with the same name.
        """
        assert not (event_name == 'tick' or event_name == 'start')
        timer = self.name_to_timer[timer_name]
        assert not abort_if_event_exists or event_name not in timer
        timer[event_name] = time.time()

    def tick_timer(self, timer_name):
        """Resets the time of the tick event associated to a specific named timer.

        Args:
            timer_name (str): Name of the timer to reset the tick event.
        """
        self.name_to_timer[timer_name]['tick'] = time.time()

    def get_time_since_event(self, timer_name, event_name, units='seconds'):
        """Get the time interval since the specified event occurred.

        Time can be retrieved in units ranging from ``nanoseconds`` to ``weeks``.

        Args:
            timer_name (str): Name of the timer.
            event_name (str): Name of the event.
            units (str, optional): Desired time units.

        Returns:
            float: Time in the desired time units.
        """
        delta = time.time() - self.name_to_timer[timer_name][event_name]
        return convert_between_time_units(delta, dst_units=units)

    def get_time_between_events(self, timer_name,
            earlier_event_name, later_event_name, units='seconds'):
        """Gets the time interval between two specific events in the same timer.

        Asserts ``False`` if the order of the events is not respected.
        Both the timer and the events must exist.
        Time can be retrieved in units ranging from ``nanoseconds`` to ``weeks``.

        Args:
            timer_name (str): Name of the timer where two event live.
            earlier_event_name (str): Name of earlier event.
            later_event_name (str): Name of the later event.
            units (str, optional): Desired time units.

        Returns:
            float: Time interval in the desired time units.
        """
        timer = self.name_to_timer[timer_name]
        delta = timer[earlier_event_name] - timer[later_event_name]
        assert delta >= 0.0
        return convert_between_time_units(delta, dst_units=units)

    def get_time_since_last_tick(self, timer_name, units='seconds'):
        """Get time elapsed since the last tick event of the desired timer.

        The timer must exist.
        Time can be retrieved in units ranging from ``nanoseconds`` to ``weeks``.

        Args:
            timer_name (str): Name of the timer.
            units (str, optional): Desired time units.

        Returns:
            float: Time interval in the desired units.
        """
        delta = time.time() - self.name_to_timer[timer_name]['tick']
        return convert_between_time_units(delta, dst_units=units)

# TODO: copy the code over for the experiments.
class SearchLogger:
    """Class managing the logging of a search experiment.

    Logging is based on the creation of folders. Each search experiment has
    a folder. Each search log folder contains multiple evaluation log folders,
    one for each architecture in the search that will be evaluated.

    Search logging is an **important** part of the framework as allows it
    us to collect supervised data from architecture evaluations. This
    dataset can in turn be used to train models to mimic the way deep
    learning experts accumulate expertise by training different models in
    different tasks.

    See also :class:`EvaluationLogger` for a logger for individual architecture
    evaluations. The evaluation loggers are managed by the the search logger.

    .. note::
        At most one of the possible boolean options about an existing search log
        folder can be ``True``. If all boolean options are ``False`` and a
        search log folder of the same name is found, the creation of the
        search logger asserts ``False``.

    Args:
        folderpath (str): Path to the folder where the search folder for the
            search experiment is to be placed (or found, if resuming the
            experiment).
        search_name (str): Name to give to the search experiment. The folder
            will have that name (or a related name, dependending on its
            existence and the options passed to the logger).
        resume_if_exists (bool, optional): If ``True`` and a logging folder is found
            for a search with the same name, resumes logging from the evaluation
            found.
        make_search_name_unique_by_numbering (bool, optional): If ``True`` and a logging
            folder is found for an experiment with the same name, a new name is
            created by suffixing the ``search_name`` with the next number that
            would make the suffixed name unique.
        create_parent_folders (bool, optional): If ``True`` and the folder where the
            search logs should lie does not exist, creates it along with any
            necessary parent folders. If ``False`` and the folder does not exist,
            it asserts ``False``.
    """
    def __init__(self, folderpath, search_name,
            resume_if_exists=False, delete_if_exists=False,
            make_search_name_unique_by_numbering=False, create_parent_folders=False):
        ok_if_exists = sum(x for x in [resume_if_exists, delete_if_exists,
            make_search_name_unique_by_numbering])
        assert ok_if_exists == 0 or ok_if_exists == 1

        self.folderpath = folderpath
        if not make_search_name_unique_by_numbering:
            self.search_name = search_name
        else:
            # getting a unique name.
            cnt = 0
            while folder_exists(join_paths([folderpath, search_name + '-%d' % cnt])):
                cnt += 1
            self.search_name = search_name + '-%d' % cnt

        self.search_folderpath = join_paths([folderpath, self.search_name])
        self.search_data_folderpath = join_paths([self.search_folderpath, 'search_data'])
        self.all_evaluations_folderpath = join_paths([self.search_folderpath, 'evaluations'])
        # self.code_folderpath = join_paths([self.search_folderpath, 'code'])

        assert ok_if_exists == 1 or not folder_exists(self.search_folderpath)
        if folder_exists(self.search_folderpath):
            if resume_if_exists:
                eval_id = 0
                while True:
                    if folder_exists(join_paths([
                        self.all_evaluations_folderpath, 'x%d' % eval_id])):
                        eval_id += 1
                    else:
                        break
                self.current_evaluation_id = eval_id

            if delete_if_exists:
                delete_folder(self.search_folderpath, False, True)
                self._create_search_folders(create_parent_folders)
                self.current_evaluation_id = 0
        else:
            self._create_search_folders(create_parent_folders)
            self.current_evaluation_id = 0

    def _create_search_folders(self, create_parent_folders):
        """Creates the subfolders of the search folder.

        Args:
            create_parent_folders (bool): Whether to create parent folders
                leading to the search folder if they do not exist.
        """
        create_folder(self.search_folderpath, create_parent_folders=create_parent_folders)
        create_folder(self.search_data_folderpath)
        create_folder(self.all_evaluations_folderpath)
        # create_folder(self.code_folderpath)

    def get_current_evaluation_logger(self):
        """Gets the evaluation logger for the next evaluation.

        Each evaluation logger is associated to a single subfolder of the
        evaluations subfolder. The returned evaluation logger is used to
        log the information about this particular evaluation.

        .. note::
            This changes the state of the search logger. The next call to this
            function will return a new evaluation logger and increment the
            number of evaluations counter for the current search.

        Returns:
            darch.search_logging.EvaluationLogger:
                Evaluation logger for the next evaluation.
        """
        logger = EvaluationLogger(self.all_evaluations_folderpath, self.current_evaluation_id)
        self.current_evaluation_id += 1
        return logger

    def get_search_data_folderpath(self):
        """Get the search data folder where data that is common to all evaluations
        can be stored.

        The user can use this folder to store whatever appropriate search level data.
        An example use-case is to store a file for the state of the
        searcher after some number of evaluations to allow us to return the
        searcher to the same state without having to repeat all evaluations.

        Returns:
            str: Path to the folder reserved for search level user data.
        """
        return self.search_data_folderpath

class EvaluationLogger:
    """Evaluation logger for a simple evaluation.

    The logging zis divided into three parts: config, features, and results.
    All three parts are represented as JSON files in disk, i.e., dictionaries.
    The config JSON encodes the architecture to be evaluated. This encoding is
    tied to the search space the evaluation was drawn from, and it can be used
    to reproduce the architecture to be evaluated given the search space.

    The features JSON contains a string representation of the architecture that
    we can use along with the information in the results to train a model that
    predicts the performance of an architecture. This is useful if the
    evaluation used to collect the results is very expensive. See also
    :func:`darch.surrogates.extract_features`.

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
    def __init__(self, all_evaluations_folderpath, evaluation_id):
        self.evaluation_folderpath = join_paths([
            all_evaluations_folderpath, 'x%d' % evaluation_id])
        self.user_data_folderpath = join_paths([self.evaluation_folderpath, 'user_data'])
        assert not folder_exists(self.evaluation_folderpath)
        create_folder(self.evaluation_folderpath)
        create_folder(self.user_data_folderpath)

        self.config_filepath = join_paths([self.evaluation_folderpath, 'config.json'])
        self.features_filepath = join_paths([self.evaluation_folderpath, 'features.json'])
        self.results_filepath = join_paths([self.evaluation_folderpath, 'results.json'])

    def log_config(self, hyperp_value_lst, searcher_evaluation_token):
        """Logs a config JSON that describing the evaluation to be done.

        The config JSON has the list with the ordered sequence of hyperparameter
        values that allow to replicate the same evaluation given the same
        search space; the searcher evaluation token, that can be given back to
        the same searcher allowing it to update back its state. The searcher
        evaluation token is returned by the searcher when a new architecture
        to evaluate is sampled. See, for example,
        :meth:`darch.searchers.MCTSearcher.sample`. The format of the searcher
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
        assert not file_exists(self.config_filepath)
        config_d = {
            'hyperp_value_lst' : hyperp_value_lst,
            'searcher_evaluation_token' : searcher_evaluation_token}
        write_jsonfile(config_d, self.config_filepath)

    def log_features(self, inputs, outputs, hyperps):
        """Logs a feature representation of the architecture to be evaluated.

        The feature representation is extracted directly from the dictionaries
        of inputs, outputs, and hyperparameters from which some number of
        (often all) modules in the network are reachable. The dictionaries
        of inputs, outputs, and hyperparameters are often a result of sampling
        an architecture from the search space with a searcher.

        Creates ``features.json`` in the evaluation log folder. See
        :func:`darch.surrogates.extract_features` for the function that extracts
        features from the dictionary representation of an architecture.

        Args:
            inputs (dict[str, darch.core.Input]): Dictionary with the
                inputs of the architecture to evaluate.
            outputs (dict[str, darch.core.Output]): Dictionary with the
                outputs of the architecture to evaluate.
            hyperps (dict[str, darch.core.Hyperparameter]): Dictionary with the
                hyperparameters of the architecture to evaluate.
        """
        assert not file_exists(self.features_filepath)
        feats = su.extract_features(inputs, outputs, hyperps)
        write_jsonfile(feats, self.features_filepath)

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
        assert (not file_exists(self.results_filepath))
        assert file_exists(self.config_filepath) and file_exists(self.features_filepath)
        assert isinstance(results, dict)
        write_jsonfile(results, self.results_filepath)

    def get_evaluation_folderpath(self):
        """Path to the evaluation folder where all the standard evaluation
        logs (e.g., ``config.json``, ``features.json``, and ``results.json``)
        are written to.

        Only standard logging information about the evaluation should be written
        here. See
        :meth:`darch.search_logging.EvaluationLogger.get_user_data_folderpath`
        for a path to a folder that can
        be used to store non-standard user logging information.

        Returns:
            str:
                Path to the folder where the standard logs about the evaluation
                are written to.
        """
        return self.evaluation_folderpath

    def get_user_data_folderpath(self):
        """Path to the user data folder where non-standard logging data can
        be stored.

        This is useful to store additional information about the evaluated
        model, e.g., example predictions of the model, model weights, or
        model predictions on the validation set.

        See :meth:`darch.search_logging.EvaluationLogger.get_evaluation_folderpath`
        for the path for the standard JSON logs for an evaluation.

        Returns:
            str: Path to the folder where the evaluations logs are written to.
        """

        return self.user_data_folderpath

def read_evaluation_folder(evaluation_folderpath):
    """Reads all the standard JSON log files associated to a single evaluation.

    See also :func:`darch.search_logging.read_search_folder` for the function
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
    assert folder_exists(evaluation_folderpath)

    name_to_log = {}
    for name in ['config', 'features', 'results']:
        log_filepath = join_paths([evaluation_folderpath, name + '.json'])
        name_to_log[name] = read_jsonfile(log_filepath)
    return name_to_log

def read_search_folder(search_folderpath):
    """Reads all the standard JSON log files associated to a search experiment.

    See also :func:`darch.search_logging.read_evaluation_folder` for the function
    that reads a single evaluation folder. The list of dictionaries is ordered
    in increasing order of evaluation id.

    Args:
        search_folderpath (str): Path to the search folder used for logging.

    Returns:
        list[dict[str,dict[str,object]]]:
            List of nested dictionaries with the logged information. Each
            dictionary in the list corresponds to an evaluation.
    """
    assert folder_exists(search_folderpath)
    all_evaluations_folderpath = join_paths([search_folderpath, 'evaluations'])
    eval_id = 0
    log_lst = []
    while True:
        evaluation_folderpath = join_paths([all_evaluations_folderpath, 'x%d' % eval_id])
        if folder_exists(evaluation_folderpath):
            name_to_log = read_evaluation_folder(evaluation_folderpath)
            log_lst.append(name_to_log)
            eval_id += 1
        else:
            break
    return log_lst

# functionality below is useful to read search log folders that are nested.
# non-nested folders are preferred though.
def is_search_log_folder(folderpath):
    return folder_exists(join_paths([folderpath, 'evaluations', 'x0']))

def recursive_list_log_folders(folderpath):
    def _iter(p, lst):
        if is_search_log_folder(p):
            lst.append(p)
        else:
            for p_child in list_folders(p):
                _iter(p_child, lst)

    log_folderpath_lst = []
    _iter(folderpath, log_folderpath_lst)
    return log_folderpath_lst

def recursive_read_search_folders(folderpath):
    all_log_lst = []
    for p in recursive_list_log_folders(folderpath):
        d = {'search_folderpath' : p, 'log_lst' : read_search_folder(p)}
        d['num_logs'] = len(d['log_lst'])
        all_log_lst.append(d)
    return all_log_lst
