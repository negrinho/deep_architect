
class HyperBand(object):
    """Abstract Class Encapsulation of HyperBand algorithm for hyperparameter optimization 
    Directly implemented from the HyperBand paper (Li et. al 2017)
    Mainly for reference and not for use 
    """
    def __init__(self): 
        if type(self) == HyperBand: 
            raise Exception("Can't instantiate Abstract Class; must define a subclass")

    def get_hyperparameter_configuration(self, n): 
        raise NotImplementedError("Need to be defined by users") 

    def run_then_return_val_loss(self, t, r): 
        raise NotImplementedError("Need to be defined by users") 

    def top_k(self, T, P, k):
        raise NotImplementedError("Need to be defined by users") 

    def get_best_performance(self, L, T, best_so_far): 
        raise NotImplementedError("Need to be defined by users") 

    def evaluate(self, R, eta = 3):
        """HyperBand algorithm for hyperparameter optimization 
        Args: 
            R: maximum amount of resource that can be allocated to a single 
                configuration. Type int  
            eta: the proportion of configurations discarded in each round of 
                SucessiveHalving. Type int 
        Return: configuration with the smallest loss 
        """
        s_max = int(math.floor(math.log(R, eta))) # number of grid search
        B = (s_max + 1) * R # total resources
        max_perf = None
        self.total_resources = 0 # number of resources allocated in total
        self.total_configs = 0 # total configurations evaluated 
        for s in xrange(s_max, -1, -1):
            n = int(math.ceil(float(B) * (eta**s) / (float(R) * (s+1)))) # num of configs
            r = R * (eta**(-s)) # resource for each config 
            # begin SuccessiveHalving 
            T = self.get_hyperparameter_configuration(n)
            self.total_configs += n
            for i in xrange(s+1):
                if len(T) == 0: # no more configurations
                    return max_perf
                n_i = int(math.floor(n * (eta ** (-i))))
                r_i = int(math.floor(r * (eta ** i)))
                L = [self.run_then_return_val_loss(t, r_i) for t in T] 
                k = int(math.floor(float(n_i) / eta))
                if (k == 0): break 
                T = self.top_k(T, L, k)
                self.total_resources += len(T) * r_i

            max_perf = self.get_best_performance(L, T, max_perf)

        return max_perf


class History(object):
    """To store history of the training such as time, val_loss, val_acc, 
    architecture, etc. 

    General structure: 
        Hyperband = s number of successive_halving bracket 
        each successive_halving bracket starts with n configurations and r resources each 
    
    So to keep track:
        Each successive halving bracket: 
            Each configuration: at each point we want to record history, we 
                record the following information 
                    (1) resource usage
                    (2) val_loss
                    (3) val_acc/user-defined metric 
        Hyperband: a dictionary of 
            keys = successive halving bracket IDs
            values = the bracket and most successful configurations (of that bracket)
    """ 
    def __init__(self): 
        self.brackets = dict() # successive halving brackets 
        self.total_configurations = 0 # total configs evaluated 
        self.total_resources = 0 # number of resources allocated in total
        self.best_configuration = None 
        self.best_performance = None 

    def record_successiveHalving(self, bracket_id, configs, performances, resource): 
        """
        Args: 
            bracket_id (int): ID of the successiveHalving bracket 
            configs (list): List of configurations
            performances (list): List of validation performances. Note that
                this should take 2 forms: 
                    (1) list of float, indicating the validation performance only. 
                        This case is simple but the graph would be less expressive 
                    (2) list of (float, dictionary), where first element is final 
                        validation performance, and dictionary is the history of 
                        the evaluation, typically something like
                        {"resource": [r0,r1,...], 
                        "val_loss": [l0,l1,...], 
                        "val_acc": [a0,a1,...]}
            resource (float/int): max resource allocated for each config

        Log a snapshot of the configurations' performances at each resource point
        """
        assert(len(configs) == len(performances))
        if bracket_id not in self.brackets: 
            self.brackets[bracket_id] = dict() 
        for (config, performance) in zip(configs, performances): 
            self.brackets[config][resource] = performance

    def record_best_of_bracket(self, bracket_id, configs, performances): 
        assert(bracket_id in self.brackets)
        assert(len(configs) == len(performances) == 1)
        self.brackets[bracket_id]["best_config"] = (configs[0], performances[0])

    def record_best_config(self, best_config, best_perf): 
        self.best_configuration = best_config
        self.best_performance = best_perf

    def convert_to_graph_hyperband_ready(self, metric, resource):
        """Convert to hyperband graph ready 
        - For each bracket, extract the best config 
        NOTE: This part makes certain assumption about the representations 
        of configurations T and performance P 

        Should have information about what type of resource and performance, 
        otherwise can default in 'resource' and 'performance'

        Right now only use for SimpleArchitectureSearchHyperband
        """ 
        data = {'configuration': [], metric: [], resource:[]}
        for bracket in self.brackets: 
            config, performance = self.brackets[bracket]
            history = performance[1] 
            config_id = config
            n = len(history[metric])
            data['configuration'] += [config_id for _ in range(n)]
            data[metric] += history[metric]
            data[resource] += history[resource]
        return data 

    def convert_a_bracket_to_graph_ready(self, bracket_id, metric, resource): 
        """Convert history to produce a successiveHalving graph 
        """
        data = {'configuration': [], metric: [], resource:[]}
        config, performance = self.brackets[bracket_id]
        history = performance[1] 
        config_id = config
        n = len(history[metric])
        data['configuration'] += [config_id for _ in range(n)]
        data[metric] += history[metric]
        data[resource] += history[resource]
        return data  

    def get_all_bracket_IDs(self): 
        """Return a sorted IDs of all brackets"""
        pass 

    def save(): 
        """Save to json file"""
        pass  

    def load(): 
        """Load from json file"""
        pass 