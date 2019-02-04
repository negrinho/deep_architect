import json
import os
import shutil
import time
import subprocess
import argparse
import pickle
from six import iteritems, itervalues


def json_object_to_json_string(d):
    return json.dumps(d, sort_keys=True)


def json_string_to_json_object(s):
    return json.loads(s)


def extract_simple_name(s):
    start = s.index('.') + 1
    end = len(s) - s[::-1].index('-') - 1
    return s[start:end]


def sleep(time_in_seconds):
    time.sleep(time_in_seconds)


def run_bash_command(cmd):
    str_output = subprocess.check_output(cmd, shell=True)
    return str_output


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


def read_picklefile(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def write_picklefile(x, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(x, f)


def path_prefix(path):
    return os.path.split(path)[0]


def join_paths(paths):
    return os.path.join(*paths)


def path_exists(path):
    return os.path.exists(path)


def file_exists(path):
    return os.path.isfile(path)


def delete_file(filepath, abort_if_notexists=True):
    assert file_exists(filepath) or (not abort_if_notexists)
    if file_exists(filepath):
        os.remove(filepath)


def folder_exists(path):
    return os.path.isdir(path)


def create_folder(folderpath, abort_if_exists=True,
                  create_parent_folders=False):
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
               ignore_files=False,
               ignore_dirs=False,
               ignore_hidden_folders=True,
               ignore_hidden_files=True,
               ignore_file_exts=None,
               recursive=False,
               use_relative_paths=False):

    assert folder_exists(folderpath)

    path_list = []
    # enumerating all desired paths in a directory.
    for root, dirs, files in os.walk(folderpath):
        if ignore_hidden_folders:
            dirs[:] = [d for d in dirs if not d[0] == '.']
        if ignore_hidden_files:
            files = [f for f in files if not f[0] == '.']
        if ignore_file_exts != None:
            files = [
                f for f in files
                if not any([f.endswith(ext) for ext in ignore_file_exts])
            ]

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
               ignore_hidden_folders=True,
               ignore_hidden_files=True,
               ignore_file_exts=None,
               recursive=False,
               use_relative_paths=False):

    return list_paths(
        folderpath,
        ignore_dirs=True,
        ignore_hidden_folders=ignore_hidden_folders,
        ignore_hidden_files=ignore_hidden_files,
        ignore_file_exts=ignore_file_exts,
        recursive=recursive,
        use_relative_paths=use_relative_paths)


def list_folders(folderpath,
                 ignore_hidden_folders=True,
                 recursive=False,
                 use_relative_paths=False):

    return list_paths(
        folderpath,
        ignore_files=True,
        ignore_hidden_folders=ignore_hidden_folders,
        recursive=recursive,
        use_relative_paths=use_relative_paths)


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
    return x / float(2**
                     (10 * (units.index(dst_units) - units.index(src_units))))


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
        for k, v in iteritems(d):
            assert type(k) == str and len(k) > 0
            if k not in self.d:
                self.d[k] = []
            self.d[k].append(v)

        if self.abort_if_different_lengths:
            assert len(set([len(v) for v in itervalues(self.d)])) <= 1

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
        self.name_to_timer[timer_name] = {
            'start': start_time,
            'tick': start_time
        }

    def create_timer_event(self,
                           timer_name,
                           event_name,
                           abort_if_event_exists=True):
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

    def get_time_between_events(self,
                                timer_name,
                                earlier_event_name,
                                later_event_name,
                                units='seconds'):
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


class CommandLineArgs:

    def __init__(self, argname_prefix=''):
        self.parser = argparse.ArgumentParser()
        self.argname_prefix = argname_prefix

    def add(self,
            argname,
            argtype,
            default_value=None,
            optional=False,
            help=None,
            valid_value_lst=None,
            list_valued=False):
        valid_types = {'int': int, 'str': str, 'float': float}
        assert argtype in valid_types

        nargs = None if not list_valued else '*'
        argtype = valid_types[argtype]

        self.parser.add_argument(
            '--' + self.argname_prefix + argname,
            required=not optional,
            default=default_value,
            nargs=nargs,
            type=argtype,
            choices=valid_value_lst,
            help=help)

    def parse(self):
        return vars(self.parser.parse_args())

    def get_parser(self):
        return self.parser


def get_config():
    cmd = CommandLineArgs()
    cmd.add('config_filepath', 'str')
    out = cmd.parse()
    cfg = read_jsonfile(out['config_filepath'])
    return cfg
