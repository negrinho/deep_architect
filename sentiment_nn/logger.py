import json
import os
from distutils.dir_util import copy_tree
import datetime

# directory structure
# main_dir
# --exp#####
# ---- user_data/ TODO
# ------?
# ---- code
# ---- user_metrics.json
# ---- config.json [hyperparameters, values, config (dictionary)]
# ---- features.json [module_connections]
# ---- results.json [results]
# ---- searcher_tokens.json [config]
# ---- metrics.json [accuracy, area under the curve/integral, FLEXIBILITY]

class SearchLogger(object):
    def __init__(self, name="", file_type = "json", inc=[], dir=""):
        self.name = name
        self.file_type = file_type.lower()
        self.file_name = name + "." + self.file_type
        # List of parameters/value names to be saved
        self.include = inc
        self.logger_dir_name = dir

    def json_log(self, arg_dict, dir_path=""):
        try:
            assert self.file_type == "json"
            out_file_path = os.path.join(dir_path, self.logger_dir_name)
            if not os.path.exists(out_file_path):
                os.makedirs(out_file_path)
            out_file = os.path.join(out_file_path, self.file_name)
            with open(out_file, "w") as fi:
                out_dict = {}
                for val in self.include:
                    out_dict[val] = str(arg_dict.get(val, "NA"))
                json.dump(out_dict, fi, indent=4)
        except AssertionError:
            print("Failed, logger type is not json.")

    def txt_log(self, arg_dict, dir_path=""):
        try:
            assert self.file_type == "txt"
            if dir_path:
               out_file = os.path.join(dir_path, self.logger_dir_name, self.file_name)
            else:
                out_file = os.path.join(self.logger_dir_name, self.file_name)
            with open(out_file, "w") as fi:
                out_dict = {}
                for val in self.include:
                    out_dict[val] = str(arg_dict.get(val, "NA"))
                fi.write(json.dumps(out_dict))
        except AssertionError:
            print("Failed, logger type is not txt.")

    # How to add support for saving matplotlib jpegs? Would need to integrate with visualization methods?

    def __str__(self):
        ret = ""
        if self.name:
            ret += self.name + "\n"
        if self.include:
            ret += "included values: "
            ret += str(self.include) + "\n"
        if self.logger_dir_name:
            ret += "logger directory name: "
            ret += self.logger_dir_name + "\n"
        return ret

    def __eq__(self, other):
        if not isinstance(other, SearchLogger):
            return False
        if self.name == other.name and self.include == other.include:
            return True
        else:
            return False


class LoggerManager(object):
    class PreexistingError(Exception):
        pass

    def __init__(self,
                 exp_label="experiment",
                 exp_dir_path="experiments",
                 code_dir_path="code",
                 code_arch_dir_path="code_archive"
                 ):
        self.experiments_dir_path = exp_dir_path
        self.code_dir_path = code_dir_path
        self.code_archive_dir_path = code_arch_dir_path

        try:
            if not os.path.exists(self.experiments_dir_path):
                os.makedirs(self.experiments_dir_path)
            if not os.path.exists(self.code_archive_dir_path):
                os.makedirs(self.code_archive_dir_path)
            assert os.path.exists(self.code_dir_path)
        except AssertionError:
            print("Could not find code directory from given path.")

        self.experiments_label = exp_label
        self.curr_experiment = 0

        config_logger = SearchLogger(name = "config",
                                     inc = ["hyperparameters", "values", "config"],
                                     dir = "config")
        features_logger = SearchLogger(name="features",
                                     inc=["features"],
                                     dir="features")
        results_logger = SearchLogger(name="results",
                                     inc=["results"],
                                     dir="results")
        searcher_logger = SearchLogger(name="searcher",
                                     inc=["config"],
                                     dir="searcher")

        self.default_loggers = {config_logger, features_logger, results_logger, searcher_logger}
        self.custom_loggers = set()

    def __str__(self):
        ret = "Environment settings:\n"
        ret += "experiments directory path: {}\n" \
              "code directory path: {}\n" \
              "code archive directory path: {}\n".format(self.experiments_dir_path,
                                                       self.code_dir_path,
                                                       self.code_archive_dir_path)
        ret += "LoggerManager state:\n"
        ret += "{\n"
        ret += "Default loggers:\n"
        for logger in self.default_loggers:
            ret += "\t\t" + str(logger) + "\n"
        ret += "}\n{\n"
        ret += "\tCustom loggers {}:\n".format(len(self.custom_loggers))
        for logger in self.custom_loggers:
            ret += "\t\t" + str(logger) + "\n"
        ret += "}"
        return ret

    def add_custom_logger(self, logger_def):
        # Create new SearchLogger within LoggerManager
        try:
            new_logger = SearchLogger(**logger_def)
            # assert isinstance(logger, SearchLogger)
            if new_logger in self.custom_loggers:
                raise LoggerManager.PreexistingError()
            else:
                self.custom_loggers.add(new_logger)
                print("Added logger.")
        # except AssertionError:
        #     print("Failed to add logger, not a SearchLogger instance.")
        except LoggerManager.PreexistingError:
            print("Failed to add logger, already exists.")

    def remove_custom_logger(self, to_del_logger_name):
        prior_count = len(self.custom_loggers)
        self.custom_loggers = set(filter(lambda l : l.name != to_del_logger_name, self.custom_loggers))
        post_count = len(self.custom_loggers)

        if prior_count == post_count:
            print("Failed to remove, does not exist.")
        else:
            print("Removed logger.")

    def archive_code(self):
        try:
            assert os.path.exists(self.code_dir_path)
            copy_tree(self.code_dir_path, self.code_archive_dir_path)
            print("Copied code from " +  self.code_dir_path + " to " + self.code_archive_dir_path + ".")
        except AssertionError:
            print("Code directory path does not exist")

    def next_exp_instance_dir(self):
        instance_dir_name = self.experiments_label + str(self.curr_experiment)
        return os.path.join(self.experiments_dir_path, instance_dir_name)

    def log_all(self, **kwargs):
        try:
            exp_instance_dir = self.next_exp_instance_dir()
            self.curr_experiment += 1
            assert not os.path.exists(exp_instance_dir)

            user_data_path = os.path.join(exp_instance_dir, "user_data")
            os.makedirs(user_data_path)

            for logger in self.default_loggers:
                if logger.file_type == "json":
                    # Don't technically need to do this since these are default loggers
                    logger.json_log(kwargs, dir_path=exp_instance_dir)


            for logger in self.custom_loggers:
                # TODO add support for more file types
                if logger.file_type == "json":
                    logger.json_log(kwargs, dir_path=user_data_path)
                elif logger.file_type == "txt":
                    logger.txt_log(kwargs, dir_path=user_data_path)

        except AssertionError:
            print("Next experiment instance directory already exists (somehow). "
                  "Aborting logger to avoid overwriting files and incrementing experiment counter.")
