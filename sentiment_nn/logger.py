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


class Logger(object):
    def __init__(self, name="", inc=[], path=""):
        self.name = name
        # self.file_name = name + "." + self.file_type
        # List of parameters/value names to be saved
        self.include = inc
        self.logger_dir_path = path
        if not os.path.exists(self.logger_dir_path):
            os.makedirs(self.logger_dir_path)

    # def json_log(self, arg_dict, dir_path=""):
    #     try:
    #         assert self.file_type == "json"
    #         out_file_path = os.path.join(dir_path, self.logger_dir_name)
    #         if not os.path.exists(out_file_path):
    #             os.makedirs(out_file_path)
    #         out_file = os.path.join(out_file_path, self.file_name)
    #         with open(out_file, "w") as fi:
    #             out_dict = {}
    #             for val in self.include:
    #                 out_dict[val] = str(arg_dict.get(val, "NA"))
    #             json.dump(out_dict, fi, indent=4)
    #     except AssertionError:
    #         print("Failed, logger type is not json.")

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


class JSONLogger(Logger):
    def __init__(self, name="", inc=[], path=""):
        Logger.__init__(self,name,inc,path)
        # self.name = name
        self.file_name = name + ".json"
        # List of parameters/value names to be saved
        # self.include = inc
        # self.logger_dir_path = path

    def log(self, arg_dict):
        out_file = os.path.join(self.logger_dir_path, self.file_name)
        with open(out_file, "w") as fi:
            out_dict = {}
            for val in self.include:
                out_dict[val] = str(arg_dict.get(val, "NA"))
            json.dump(out_dict, fi, indent=4)

# set up experiments and code archive directory
# check to make sure code directory exists
def setup_directories(
                 experiments_dir_path="experiments",
                 code_dir_path="code",
                 code_archive_dir_path="code_archive"):
    try:
        if not os.path.exists(experiments_dir_path):
            os.makedirs(experiments_dir_path)
        if not os.path.exists(code_archive_dir_path):
            os.makedirs(code_archive_dir_path)
        if code_dir_path:
            assert os.path.exists(code_dir_path)
        else: # code_dir_path is the empty string
            assert len([content for content in os.listdir(os.getcwd()) if ".py" in content]) > 0
            code_dir_path = os.getcwd()
    except AssertionError:
        print("Invalid or non-existent code directory from given path.")

    return experiments_dir_path,code_dir_path,code_archive_dir_path

# iterator that makes next experiment directory
# and nested user_data directory
# returns next experiment directory path
# if resume is True then it finds the most recent experiment directory
# if the most recent experiment directory has a results.json then creates
# a next directory, else redoes the most recent one
def experiment_directory_iterator(experiments_dir_path,
                                  experiment_label,
                                  start=0,
                                  resume=False):
    if resume:
        # find most recent experiment path with a results.json
        all_experiments = [exp_dir[len(experiment_label):] for exp_dir in
            os.listdir(experiments_dir_path) if experiment_label in exp_dir]
        if all_experiments:
            int_all_experiments = [int(x) for x in all_experiments]
            most_recent = max(int_all_experiments)
            most_recent_dir = experiment_label + str(most_recent)
            most_recent_contents = os.listdir(os.path.join(experiments_dir_path,most_recent_dir))
            if "results.json" in most_recent_contents:
                i = most_recent + 1
            else:
                i = most_recent
        else:
            i = start
    else:
        i = start
    while True:
        inst_name = experiment_label + str(i)
        inst_path = os.path.join(experiments_dir_path,inst_name)
        os.makedirs(os.path.join(inst_path,"user_data"))
        yield inst_path
        i += 1

def archive_code(code_dir_path,code_archive_dir_path):
    try:
        assert os.path.exists(code_dir_path)
        assert os.path.exists(code_archive_dir_path)
        copy_tree(code_dir_path, code_archive_dir_path)
        print("Copied code from " +  code_dir_path + " to " + code_archive_dir_path + ".")
    except AssertionError:
        print("Code directory path does not exist")
