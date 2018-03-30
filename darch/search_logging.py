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
    def __init__(self, abort_if_different_lengths=False):
        self.abort_if_different_lengths = abort_if_different_lengths
        self.d = {}

    def append(self, d):
        for k, v in d.iteritems():
            assert type(k) == str and len(k) > 0
            if k not in self.d:
                self.d[k] = []
            self.d[k].append(v)

        if self.abort_if_different_lengths:
            assert len(set([len(v) for v in self.d.itervalues()])) <= 1

    def get_dict(self):
        return dict(self.d)

class TimeTracker:
    def __init__(self):
        self.init_time = time.time()
        self.last_registered = self.init_time

    def time_since_creation(self, units='seconds'):
        now = time.time()
        elapsed = now - self.init_time
        return convert_between_time_units(elapsed, dst_units=units)

    def time_since_last_call(self, units='seconds'):
        now = time.time()
        elapsed = now - self.last_registered
        self.last_registered = now
        return convert_between_time_units(elapsed, dst_units=units)

class TimerManager:
    def __init__(self):
        self.init_time = time.time()
        self.last_registered = self.init_time
        self.name_to_timer = {}

    def create_timer(self, timer_name, abort_if_timer_exists=True):
        assert not abort_if_timer_exists or timer_name not in self.name_to_timer
        start_time = time.time()
        self.name_to_timer[timer_name] = {'start' : start_time, 'tick' : start_time}

    def create_timer_event(self, timer_name, event_name, abort_if_event_exists=True):
        assert not timer_name == 'tick'
        timer = self.name_to_timer[timer_name]
        assert not abort_if_event_exists or event_name not in timer
        timer[event_name] = time.time()

    def tick_timer(self, timer_name):
        self.name_to_timer[timer_name]['tick'] = time.time()

    def get_time_since_event(self, timer_name, event_name, units='seconds'):
        delta = time.time() - self.name_to_timer[timer_name][event_name]
        return convert_between_time_units(delta, dst_units=units)

    def get_time_between_events(self, timer_name,
            earlier_event_name, later_event_name, units='seconds'):
        timer = self.name_to_timer[timer_name]
        delta = timer[earlier_event_name] - timer[later_event_name]
        assert delta >= 0.0
        return convert_between_time_units(delta, dst_units=units)

    def get_time_since_last_tick(self, timer_name, units='seconds'):
        delta = time.time() - self.name_to_timer[timer_name]['tick']
        return convert_between_time_units(delta, dst_units=units)

class SearchLogger:
    def __init__(self, folderpath, search_name,
            resume_if_exists=False, delete_if_exists=False,
            make_search_name_unique_by_numbering=False, create_parent_folders=False):
        assert sum(x for x in [resume_if_exists, delete_if_exists,
            make_search_name_unique_by_numbering]) <= 1

        self.folderpath = folderpath
        if not make_search_name_unique_by_numbering:
            self.search_name = search_name
        else:
            cnt = 0
            while folder_exists(join_paths([folderpath, search_name + '-%d' % cnt])):
                cnt += 1
            self.search_name = search_name + '-%d' % cnt

        self.search_folderpath = join_paths([folderpath, self.search_name])
        self.search_data_folderpath = join_paths([self.search_folderpath, 'search_data'])
        self.all_evaluations_folderpath = join_paths([self.search_folderpath, 'evaluations'])
        # self.code_folderpath = join_paths([self.search_folderpath, 'code'])

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
        create_folder(self.search_folderpath, create_parent_folders=create_parent_folders)
        create_folder(self.search_data_folderpath)
        create_folder(self.all_evaluations_folderpath)
        # create_folder(self.code_folderpath)

    def get_current_evaluation_logger(self):
        logger = EvaluationLogger(self.all_evaluations_folderpath, self.current_evaluation_id)
        self.current_evaluation_id += 1
        return logger

    def get_search_data_folderpath(self):
        return self.search_data_folderpath

class EvaluationLogger:
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
        assert not file_exists(self.config_filepath)
        config_d = {
            'hyperp_value_lst' : hyperp_value_lst,
            'searcher_evaluation_token' : searcher_evaluation_token}
        write_jsonfile(config_d, self.config_filepath)

    def log_features(self, inputs, outputs, hs):
        assert not file_exists(self.features_filepath)
        feats = su.extract_features(inputs, outputs, hs)
        write_jsonfile(feats, self.features_filepath)

    def log_results(self, results):
        assert (not file_exists(self.results_filepath))
        assert file_exists(self.config_filepath) and file_exists(self.features_filepath)
        assert isinstance(results, dict)
        write_jsonfile(results, self.results_filepath)

    def get_evaluation_folderpath(self):
        return self.evaluation_folderpath

    def get_user_data_folderpath(self):
        return self.user_data_folderpath

def read_evaluation_folder(evaluation_folderpath):
    assert folder_exists(evaluation_folderpath)

    name_to_log = {}
    for name in ['config', 'features', 'results']:
        log_filepath = join_paths([evaluation_folderpath, name + '.json'])
        name_to_log[name] = read_jsonfile(log_filepath)
    return name_to_log

def read_search_folder(search_folderpath):
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

# TODO: write error messages for the loggers, e.g., asserts.
# TODO: add some error checking or options to the read_log
# TODO: the traversal of the logging folders can be done better, e.g., some
# additional features.
# TODO: maybe move some of this file system manipulation to their own folder.
# TODO: integrate better the use of list files and list folders.
# TODO: check how to better integrate with the other models.
# TODO: add more user_data and functionality to load then.
# TODO: add the ability to have a function that is applied to each file type.
# TODO: read_evaluation_folder can be done in a more complicated manner
# that allows for some configs to not be present.