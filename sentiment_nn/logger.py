import json
import os

label = "exp"

# directory structure
# main_dir
# --exp#####
# ---- user_data/ TODO
# ------
# ---- user_metrics.json TODO
# ---- config.json [hyperparameters, values, config (dictionary)]
# ---- features.json [module_connections]
# ---- results.json [results]
# ---- searcher_tokens.json [config]
# ---- metrics.json [accuracy, area under the curve/integral, FLEXIBILITY]

default_loggers = {"config.json": ["hyperparameters", "values", "config"],
                   "features.json": ["features"],
                   "results.json": ["results"],
                   "searcher.json": ["config"]}

class Logger:
    def __init__(self, nm = "", inc =[], pth=""):
        self.name = nm
        self.include = inc
        self.path = pth

    def log(self, d):
        with open(self.path, "w") as fi:
            tbd = {}
            for val in self.include:
                tbd[val] = str(d.get(val, "NA"))
            json.dump(tbd, fi, indent=4)

def log_instance(**kwargs):
    # wrapper
    # run for each model
    # first get latest model number
    all = os.listdir(os.getcwd())
    exp_nums = [int(dir[len(label):]) for dir in all if label == dir[:len(label)]]
    # will probably have to make this more unique
    if exp_nums:
        exp_num = max(exp_nums)+1
    else:
        exp_num = 1
    exp_dir = label + str(exp_num)

    assert not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

    assert not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

    instance_loggers = []
    for logger_name, logger_inc in default_loggers.items():
        instance_loggers.append(Logger(nm = logger_name, inc = logger_inc, pth = os.path.join(exp_dir, logger_name)))

    for logger in instance_loggers:
        logger.log(kwargs)
