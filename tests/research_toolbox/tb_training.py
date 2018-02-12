### sequences, schedules, and counters (useful for training)
import numpy as np

# TODO: this sequence is not very useful.
class Sequence:
    def __init__(self):
        self.data = []
    
    def append(self, x):
        self.data.append(x)

# TODO: this can be done with the patience counter.
class PatienceRateSchedule:
    def __init__(self, rate_init, rate_mult, rate_patience,  
            rate_max=np.inf, rate_min=-np.inf, minimizing=True):
 
        assert (rate_patience > 0 and (rate_mult > 0.0 and rate_mult <= 1.0) and
            rate_min > 0.0 and rate_max >= rate_min) 
        
        self.rate_init = rate_init
        self.rate_mult = rate_mult
        self.rate_patience = rate_patience
        self.rate_max = rate_max
        self.rate_min = rate_min
        self.minimizing = minimizing
        
        # for keeping track of the learning rate updates
        self.counter = rate_patience
        self.prev_value = np.inf if minimizing else -np.inf
        self.cur_rate = rate_init

    def update(self, v):
        # if it improved, reset counter.
        if (self.minimizing and v < self.prev_value) or (
                (not self.minimizing) and v > self.prev_value) :
            self.counter = self.rate_patience
        else:
            self.counter -= 1
            if self.counter == 0:
                self.cur_rate *= self.rate_mult
                # rate truncation
                self.cur_rate = min(max(self.rate_min, self.cur_rate), self.rate_max)
                self.counter = self.rate_patience

        self.prev_value = min(v, self.prev_value) if self.minimizing else max(v, self.prev_value)

    def get_rate(self):
        return self.cur_rate

# TODO: write AdditiveRateSchedule; MultiplicativeRateSchedule.
class AddRateSchedule:
    def __init__(self, rate_init, rate_end, duration):
        
        assert rate_init > 0 and rate_end > 0 and duration > 0

        self.rate_init = rate_init
        self.rate_delta = (rate_init - rate_end) / float(duration)
        self.duration = duration
        
        self.num_steps = 0
        self.cur_rate = rate_init

    def update(self, v):
        assert self.num_steps < self.duration
        self.num_steps += 1

        self.cur_rate += self.rate_delta 

    def get_rate(self):
        return self.cur_rate

class MultRateSchedule:
    def __init__(self, rate_init, rate_mult):
        
        assert rate_init > 0 and rate_mult > 0

        self.rate_init = rate_init
        self.rate_mult = rate_mult

    def update(self, v):
        self.cur_rate *= self.rate_mult

    def get_rate(self):
        return self.cur_rate

class ConstantRateSchedule:
    def __init__(self, rate):
        self.rate = rate
    
    def update(self, v):
        pass
    
    def get_rate(self):
        return self.rate

class StepwiseRateSchedule:
    def __init__(self, rates, durations):
        assert len( rates ) == len( durations )

        self.schedule = PiecewiseSchedule( 
            [ConstantRateSchedule(r) for r in rates],
            durations)

    def update(self, v):
        pass
    
    def get_rate(self):
        return self.schedule.get_rate()

class PiecewiseSchedule:
    def __init__(self, schedules, durations):
       
        assert len(schedules) > 0 and len(schedules) == len(durations) and (
            all([d > 0 for d in durations]))

        self.schedules = schedules
        self.durations = durations

        self.num_steps = 0
        self.idx = 0

    def update(self, v):
        n = self.num_steps
        self.idx = 0
        for d in durations:
            n -= d
            if n < 0:
                break
            self.idx += 1
        
        self.schedules[self.idx]

    def get_rate(self):
        return self.schedules[self.idx].get_rate()

class PatienceCounter:
    def __init__(self, patience, init_val=None, minimizing=True, improv_thres=0.0):
        assert patience > 0 
        
        self.minimizing = minimizing
        self.improv_thres = improv_thres
        self.patience = patience
        self.counter = patience

        if init_val is not None:
            self.best = init_val
        else:
            if minimizing:
                self.best = np.inf
            else:
                self.best = -np.inf

    def update(self, v):
        assert self.counter > 0

        # if it improved, reset counter.
        if (self.minimizing and self.best - v > self.improv_thres) or (
            (not self.minimizing) and v - self.best > self.improv_thres) :

            self.counter = self.patience
        
        else:

            self.counter -= 1

        # update with the best seen so far.
        if self.minimizing:
            self.best = min(v, self.best)  
        
        else: 
            self.best = max(v, self.best)

    def has_stopped(self):
        return self.counter == 0

# NOTE: actually, state_dict needs a copy.
# example
# cond_fn = lambda old_x, x: old_x['acc'] < x['acc']
# save_fn = lambda x: tb.copy_update_dict(x, { 'model' : x['model'].save_dict() })
# load_fn = lambda x: x,
# for example, but it allows more complex functionality.
class Checkpoint:
    def __init__(self, initial_state, cond_fn, save_fn, load_fn):
        self.state = initial_state
        self.cond_fn = cond_fn
        self.save_fn = save_fn
        self.load_fn = load_fn
    
    def update(self, x):
        if self.cond_fn(self.state, x):
            self.state = self.save_fn(x)
    
    def get(self):
        return self.load_fn( self.state )

def get_best(eval_fns, minimize):
    best_i = None
    if minimize:
        best_v = np.inf
    else:
        best_v = -np.inf

    for i, fn in enumerate(eval_fns):
        v = fn()
        if minimize:
            if v < best_v:
                best_v = v
                best_i = i
        else:
            if v > best_v:
                best_v = v
                best_i = i
    
    return (best_i, best_v)

### for storing the data
class InMemoryDataset:
    """Wrapper around a dataset for iteration that allows cycling over the 
    dataset. 

    This functionality is especially useful for training. One can specify if 
    the data is to be shuffled at the end of each epoch. It is also possible
    to specify a transformation function to applied to the batch before
    being returned by next_batch.

    """
    
    def __init__(self, X, y, shuffle_at_epoch_begin, batch_transform_fn=None):
        if X.shape[0] != y.shape[0]:
            assert ValueError("X and y the same number of examples.")

        self.X = X
        self.y = y
        self.shuffle_at_epoch_begin = shuffle_at_epoch_begin
        self.batch_transform_fn = batch_transform_fn
        self.iter_i = 0

    def get_num_examples(self):
        return self.X.shape[0]

    def next_batch(self, batch_size):
        """Returns the next batch in the dataset. 

        If there are fewer that batch_size examples until the end
        of the epoch, next_batch returns only as many examples as there are 
        remaining in the epoch.

        """

        n = self.X.shape[0]
        i = self.iter_i

        # shuffling step.
        if i == 0 and self.shuffle_at_epoch_begin:
            inds = np.random.permutation(n)
            self.X = self.X[inds]
            self.y = self.y[inds]

        # getting the batch.
        eff_batch_size = min(batch_size, n - i)
        X_batch = self.X[i:i + eff_batch_size]
        y_batch = self.y[i:i + eff_batch_size]
        self.iter_i = (self.iter_i + eff_batch_size) % n

        # transform if a transform function was defined.
        if self.batch_transform_fn != None:
            X_batch_out, y_batch_out = self.batch_transform_fn(X_batch, y_batch)
        else:
            X_batch_out, y_batch_out = X_batch, y_batch

        return (X_batch_out, y_batch_out)
