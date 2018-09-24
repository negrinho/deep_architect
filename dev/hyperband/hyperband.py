from common import History

class HyperBandExtended(object):
    """An Extended Version of HyperBand Algorithm (Li et. al 2017)
    The main difference is that the user is not restricted to using the validation 
    loss as the metric. The user will need to define the following functions:  
        get_hyp_config_fn(n): returns a list of n sampled configurations
        run_fn(T, r): takes list of configurations T and resource r for each 
            config, and returns the results (often a validation metric) from 
            evaluating that configuration with resource r 
        top_k_fn(T, P, k): takes in list of config T, list of validation performance 
            P, integer k and returns the (sorted) top k list of configuration
        get_best_fn (T, P, cur_config, cur_perf): takes in list of
            validation performance T, list of corresponding configs P, best config 
            and best validation performance so far. Returns the updated best 
            config and best validation performance.  
    NOTE: T, P, and r need to be consistent across functions. We recommend the 
    following representation: 
    - resource r should be a float or an int, and serializable. 
    - T should take the form list of (config_id, config_representation). Each 
    element should be serializable (for storage in History object)
    - For validation performance P, there are 2 possible representations: 
        (1) the most basic form is a list of float indicating the validation 
            result at each resource r. 
        (2) Another common representation is P = list[(val_perf, history)], 
        where val_perf is the validation performance of a particular configuration, 
        and history is a dictionary of results (progression of val_loss and val_perf 
        vs intervals of resources) from evaluating configuration. This is particularly 
        useful when graphing so that we have a more complete picture of the evaluation. 
        For an example see ArchitectureSearchHyperBand class.

    This class also stores all history in a History object, for reference and 
    graphing purposes. 
    """
    def __init__(self, metric, get_hyp_config_fn, run_fn, top_k_fn, get_best_fn): 
        self.get_hyperparameter_configuration = get_hyp_config_fn
        self.run_then_return_val_performance =  run_fn 
        self.top_k = top_k_fn
        self.get_best_performance = get_best_fn
        self.history = History()

    def evaluate(self, R, eta = 3):
        """HyperBand algorithm for hyperparameter optimization 
        Args: 
            R (int): maximum amount of resource that can be allocated to a single 
                configuration
            eta (int): the proportion of configurations discarded in each round of 
                SucessiveHalving
        Return: configuration with the smallest loss 
        """
        s_max = int(math.floor(math.log(R, eta))) # number of grid search, or "bracket"
        B = (s_max + 1) * R # total resources
        best_config, best_perf = None, None 
        for s in xrange(s_max, -1, -1):
            n = int(math.ceil(float(B) * (eta**s) / (float(R) * (s+1)))) # num configs
            r = R * (eta**(-s)) # resource for each config 
            # begin SuccessiveHalving 
            T = self.get_hyperparameter_configuration(n)
            self.history.total_configs += n
            for i in xrange(s+1):
                if len(T) == 0: # no more configurations
                    return max_perf
                n_i = int(math.floor(n * (eta ** (-i))))
                r_i = int(math.floor(r * (eta ** i)))
                P = self.run_then_return_val_performance(T, r_i)
                self.history.record_succesiveHalving(s, T, P, r_i) 
                k = int(math.floor(float(n_i) / eta))
                if (k == 0): break 
                T = self.top_k(T, P, k)
                self.history.total_resources += len(T) * r_i
            self.history.record_best_of_bracket(s, T, P)

            (best_config, best_perf) = self.get_best_performance(T, P, best_config, best_perf)

        self.history.record_best_config(best_config, best_perf)
        return best_config

    def graph_hyperband(self, save_dir='./tmp/hyperband.jpg'):
        """Extract from history and show Hyperband graph (best performance for 
        each successive halving bracket)
        """
        pass 

    def graph_sucessiveHalving(self, bracket_id, save_dir='./tmp/sh.jpg'): 
        pass 

class SimpleArchitectureSearchHyperBand(HyperBandExtended):
    """
    Hyperband algorithm for simple architecture search experiments. It is designed 
    to maximize the hyperband algorithm, thus some arguments need to conform certain 
    specifications. For more flexibility, one can use HyperBandExtended and 
    define the required functions. 
    Args: 
        searcher: DeepArchitect Searcher (RandomSearch is most common)
        evaluator: DeepArchitect Evaluator. The evaluator is similar to the normal
            evaluator defined in DeepArchitect (see mnist tutorial) with one 
            difference: The evaluate function takes in (DeepArchitect inputs, 
            DeepArchitect outputs, max resource) and returns a dictionary: 
                    {"val_accuracy": ...,
                     "history": {"resource": [], "val_acc": [], etc.}}
                where "history" is the history of evaluating that architecture 
        metric: must be either "val_loss" or "val_accuracy"
        resource_type must be either "epoch" or "training_time"

    The Configurations T and Performance P will have the following representations
        Configurations T = (arch_id, (inputs, outputs, hs)). Note that inputs,
            outputs, and hs sampled from the searcher are enough to define 
            an architecture 
        Performance P = (final_val_perf, dict("resource":[], "val_loss":[]...))

    TODO: perhaps have an enabled mode of "multiworker" or not? 
    """
    def __init__(self, searcher, evaluator, metric='val_loss', resource_type='epoch'): 
        assert(metric == 'val_loss' or metric == 'val_accuracy')
        assert(resource_type == 'epoch' or resource_type == 'training_time')
        self.searcher = searcher
        self.evaluator = evaluator
        self.metric = metric 
        self.resource_type = resource_type
        self.history = History() 

    def get_hyperparameter_configuration(self, n): 
        configs = [] 
        for i in range(n): 
            (inputs, outputs, hs, h_value_his, searcher_token) = self.searcher.sample() 
            configs.append((i, (inputs, outputs, hs)))
        return configs 

    def run_then_return_val_performance(self, T, r): 
        P = []
        for (arch_id, (inputs, outputs, hs)) in T: 
            results = self.evaluator.evaluate(inputs, outputs, hs, r) 
            P.append(results['val_accuracy'], results['history'])
        return P 

    def top_k(self, T, P, k):
        reverse = False if self.metric == 'val_loss' else True
        # combine T, P and sort according to validation performance 
        sorted_tp = sorted(list(zip(T, P)), key=lambda tp: tp[1][0], reverse=reverse)
        # map back to only T and get the top k
        return list(map(lambda tp: tp[0], sorted_tp[:k]))

    def get_best_performance(self, T, P, best_config, best_perf): 
        assert(len(T) == len(P) == 0)
        get_max = False if self.metric == 'val_loss' else True 
        cur_config, cur_perf = T[0], P[0]
        if (self.metric == 'val_accuracy'): 
            return cur_config if best_config == None or (cur_perf[0] > best_perf[0]) else best_config
        elif (self.metric == 'val_loss'): 
            return cur_config if best_config == None or (cur_perf[0] < best_perf[0]) else best_config
        
    def graph_hyperband(self, save_dir='./tmp/hyperband.jpg'):
        """More customized graphing functionality
        """
        import pandas as pd 
        from plotnine import *  
        pass 

    def graph_sucessiveHalving(self, bracket_id, save_dir='./tmp/sh.jpg'): 
        import pandas as pd 
        import plotnine as plot 
        pass 



     