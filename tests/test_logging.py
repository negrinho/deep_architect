
# TODO: this is done to check that there are good ways of 
# storing information about the models trained.

# TODO: there can exist multiple options to work with it.

# NOTE: some of the logic is independent of the framework,
# other logic is dependent of the framework, like 
# how to store or save a model

# keep the models or only keep the results.
# type of file to keep about. what is the most important.

# can save:
# validation performance.
# stuff that is returned by the evaluator.

# each model gets a folder.
# basically a 

# cfg_d, vs, val, ...
# with this information, and potentially the information about the search_space
# fn
# save the searcher too maybe, it is possible to retrain an existing model.

class Saver:
    def __init__(self):
        pass