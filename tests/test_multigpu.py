
# NOTE: this use case is going to be very unstable for now.
import darch.searchers as se
import multiprocessing
import psutil
import research_toolbox.tb_resources as tb_re
import research_toolbox.tb_logging as tb_lg

class SimpleCoordinator:
    def __init__(self, searcher, worker_lst):
        self.searcher = searcher
        self.worker_lst = worker_lst
    
    def update(self, val, cfg_d):
        self.searcher.update(val, cfg_d)
        
    def sample(self):
        _, _, _, v_hist, cfg_d = self.searcher.sample()

    def run(num_samples):
        cnt = 0
        while cnt < num_samples:
            for wk in self.worker_lst:
                if wk.is_result_available():
                    r = wk.get_result()
                    self.update(val, cfg_d)
                    cnt += 1

                if wk.is_available():
                    cfg = self.sample()
                    wk.run(cfg)

# NOTE: the runner is the one that can do something about it.
# NOTE: this is pretty much done.
def run(searcher, worker_lst, eval_fn, num_samples):
    cnt = 0
    cfg_d 
    while cnt < num_samples:
        for i, wk in enumerate(worker_lst):
            if wk.is_result_available():
                r = wk.get_result()
                searcher.update(val, cfg_d)
                cnt += 1

            if wk.is_available():
                _, _, _, v_hist, cfg_d = searcher.sample()
                wk.run(v_hist)    

# the instantiation of the searchers can be done here.

# TODO: do the simple cpu first.

# NOTE: some functionality that is not needed.

class CPUWorker:
    def __init__(self, search_space_fn, eval_fn):
        self.p = None
        self.p_start_time = None
        self.search_space_fn = search_space_fn
        self.eval_fn = eval_fn
    
    def is_busy(self):
        return self.p is None or p.is_alive()

    def current_run_memory(self, units='mb'):
        raise NotImplementedError
        # mbs_process(p.pid)
    
    def current_run_time(self, units='s'):
        now = time.time()
        return tb_lg.convert_between_time_units(now - self.p_start_time, dst_units=unit)
    
    # NOTE: may also return the 
    def get_result(self):
        return self.r
    
    # the logic to check this goes here.
    def run(self, v_hist):
        assert not self.is_busy()

        self.search_space_fn()

        self.p_start_time = time.time()
        p = multiprocessing.Process(target=experiment_fn, kwargs=kwargs)
        
    def abort(self):
        self.p.terminate()




class Coordinator:
    def __init__(self, worker_lst):
        self.worker_lst = worker_lst
    
    def per_loop_update(self):
        raise NotImplementedError

    def result_available_update(self, val, cfg_d):
        raise NotImplementedError
        
    def sample(self):
        raise NotImplementedError

    def run(num_samples):
        cnt = 0
        while cnt < num_samples:
            for wk in self.worker_lst:
                if wk.is_result_available():
                    r = wk.get_result()
                    self.result_available_update(r)
                    cnt += 1

                if wk.is_available():
                    cfg = self.sample()
                    wk.run(cfg)
                
            self.per_loop_update()

class Worker:
    def __init__(self):
        pass
    
    def is_busy(self):
        pass
    
    def is_available(self):
        pass

    def current_run_memory(self, units='mb'):
        pass
    
    def current_run_time(self, units='s'):
        pass
    
    # NOTE: may also return the 
    def get_result(self):
        return self.r
    
    # the logic to check this goes here.
    def run(self, v_hist):
        pass
    
    def abort(self):
        pass

class CPUWorker:
    def __init__(self, search_space_fn, eval_fn):
        self.p = None
        self.p_start_time = None
        self.search_space_fn = search_space_fn
        self.eval_fn = eval_fn
    
    def is_busy(self):
        return self.p is None or p.is_alive()

    def current_run_memory(self, units='mb'):
        raise NotImplementedError
        # mbs_process(p.pid)
    
    def current_run_time(self, units='s'):
        now = time.time()
        return tb_lg.convert_between_time_units(now - self.p_start_time, dst_units=unit)
    
    # NOTE: may also return the 
    def get_result(self):
        return self.r
    
    # the logic to check this goes here.
    def run(self, v_hist):
        assert not self.is_busy()

        self.search_space_fn()

        self.p_start_time = time.time()
        p = multiprocessing.Process(target=experiment_fn, kwargs=kwargs)
        
    def abort(self):
        self.p.terminate()

# NOTE: in what way can this be different.
# it may be different because 


class Coordinator:
    def __init__(self, searcher, worker_lst):
        self.searcher = searcher
        self.worker_lst = worker_lst
        self.cfg_d_lst = [None] * len(worker_lst)
    
    def per_loop_update(self):
        pass

    def result_available_update(self, val, cfg_d):
        self.searcher.update(val, cfg_d)
        
    def sample(self):
        pass

    def run(num_samples):
        cnt = 0
        while cnt < num_samples:
            for wk in self.worker_lst:
                if wk.is_result_available():
                    r = wk.get_result()
                    self.result_available_update(r)
                    cnt += 1

                if wk.is_available():
                    cfg = self.sample()
                    wk.run(cfg)
                
            self.per_loop_update()




class CNoState(Coordinator):
    def __init__(self, searcher, worker_lst):
        Coordinator.__init__(self, worker_lst)
        self.seacher = searcher

    def per_loop_update(self):
        pass
    
    def result_available_update(self, r):
        pass

    def sample(self):
        _, _, _, v_hist, cfg_d = self.searcher.sample()
        return v_hist

class MCTSCoordinator(Coordinator):
    def __init__(self, searcher, worker_lst):
        Coordinator.__init__(self, worker_lst)
        self.seacher = searcher

    def per_loop_update(self):
        pass
    
    def result_available_update(self, r):
        self.searcher.update()

    def sample(self):
        _, _, _, v_hist, cfg_d = self.searcher.sample()
        return v_hist

# NOTE: for now, I'm just solving this independently for each of the models.

class CMCTS(Coordinator):
    def __init__(self, searcher, worker_lst):
        Coordinator.__init__(self, worker_lst)
        self.worker_lst = worker_lst
        self.seacher = searcher
        self.cfg_d_lst = [None] * len(worker_lst)

    def per_loop_update(self):
        pass
    
    def result_available_update(self, r):
        self.searcher.update()

    def sample(self):
        _, _, _, v_hist = self.searcher.sample()
        return v_hist

def RandomCoordinator(Coordinator):
    def __init__(self, searcher, worker_lst, per_cycle_sleep)
        self.per_cycle_sleep = per_cycle_sleep

    def per_cycle_update(self):
        sleep(per_cycle_sleep)
    
    def new_result_update(self, val, v_hist, cfg_d):
        pass


# some only have new_result_updates. 
# others do not have that.

def MCTSCoordinator(Coordinator):
    pass


# all these things are supported on evaluation, but the correct 
# use of the hardware depends on a lot of things.

# this depends on the stuff that needs to run.
# information about the job that is being ran 


# NOTE: I don't know if this is going to be in MBs.
def mbs_process(pid):
    psutil_p = psutil.Process(pid)
    mem_p = psutil_p.memory_info()[0]
    
    return mem_p

# TODO: I will need to convert some of these.
# TODO: how to make sure that this is going to work in terms of exposing 
# the right GPUs.
class CPUWorker:
    def __init__(self, search_space_fn, eval_fn):
        self.p = None
        self.p_start_time = None
    
    def is_busy(self):
        return self.p is None or p.is_alive()

    def current_run_memory(self, units='mb'):
        mbs_process(p.pid)
        
    
    def current_run_time(self, units='s'):
        now = time.time()
        return tb_lg.convert_between_time_units(now - self.p_start_time, dst_units=unit)
    
    # NOTE: may also return the 
    def get_result(self):
        return self.r
    
    # the logic to check this goes here.
    def run(self, v_hist):
        assert not self.is_busy()

        self.p_start_time = time.time()
        p = multiprocessing.Process(target=experiment_fn, kwargs=kwargs)
        
    def abort(self):
        self.p.terminate()

# NOTE: if I keep that information, it means that I can 
# get back the state from the model.

# NOTE: the worker ID may know its value.
# it does not matter.

# NOTE: returning things hsa to be done via something.

# the only thing I need is the value really, at least for now.

# NOTE: this is needs to communicate a serialized version of this.


def run_guarded_experiment(maxmemory_mbs, maxtime_secs, experiment_fn, **kwargs):
        start = time.time()
        
        p = multiprocessing.Process(target=experiment_fn, kwargs=kwargs)
        p.start()
        while p.is_alive():
            p.join(1.0)     
            try:
                mbs_p = mbs_process(p.pid)
                if mbs_p > maxmemory_mbs:
                    print "Limit of %0.2f MB exceeded. Terminating." % maxmemory_mbs
                    p.terminate()

                secs_p = time.time() - start
                if secs_p > maxtime_secs:
                    print "Limit of %0.2f secs exceeded. Terminating." % maxtime_secs
                    p.terminate()
            except psutil.NoSuchProcess:
                pass

# 

# NOTE: it does not even has to be a cpu worker. this is important.
# NOTE: this only works if it is called before Tensorflow gets the chance
# to look at the GPUs.
# ignore GPUs for  


if __name__ == '__main__':
    worker_lst = None
    searcher = None
    num_samples = 16
    crd = Coordinator(searcher, worker_lst)

    crd.run(num_samples)

# where would the result be kept. 
# 

# I guess it is the 




# NOTE: certain of these things can be blocking.

class CPUWorker(Worker):
    def __init__(self):
        pass
    
    


import psutil
import multiprocessing
import time

def mbs_process(pid):
    psutil_p = psutil.Process(pid)
    mem_p = psutil_p.memory_info()[0]
    
    return mem_p

# NOTE: this needs to be turned into something else.
def run_guarded_experiment(maxmemory_mbs, maxtime_secs, experiment_fn, **kwargs):
        start = time.time()
        
        p = multiprocessing.Process(target=experiment_fn, kwargs=kwargs)
        p.start()
        while p.is_alive():
            p.join(1.0)     
            try:
                mbs_p = mbs_process(p.pid)
                if mbs_p > maxmemory_mbs:
                    print "Limit of %0.2f MB exceeded. Terminating." % maxmemory_mbs
                    p.terminate()

                secs_p = time.time() - start
                if secs_p > maxtime_secs:
                    print "Limit of %0.2f secs exceeded. Terminating." % maxtime_secs
                    p.terminate()
            except psutil.NoSuchProcess:
                pass


### TODO: think about how to do this in the case where we want to think
# about different things.
# NOTE: how to stop a process and stuff like that. periodic communication.
# this would be needed for the multi process in the case of 

# TODO: try different modes of computation. 

# should work on a machine on a machine
# should do some ammount of process management.
# runs somewhere, does not matter where. this is important. the most 
# important thing is guarantee that things are comparable. focus on the 
# matrix cluster for now.
# or just locally for now.

# monitoring 
# it is possible to look at this only with CPUs. 
# this needs to be general.

# gets a standby_fn; each time it waits, it is doing the standby functionality 
# this needs to have access to the searcher. fine.
# all the assynchronous logic is going to be done there.

# can fail. this is important.

# I don't think that it necessary needs to know about the searcher, but this 
# is going to be interesting.

# coordinator is always the same.
# 
# 
#  

# this mostly makes sense for the case where there are many things working.
# NOTE: the fact that this is not stable means that MCTS can not work very wel 
# in some cases.



# should have samples prepared for the workers.
# everytime it fills, it should have different ones.
# make sure that I can have different ones all the time.
# this requires more information.






# basically, if there are some free, do something with it.

# worker can be running on multiple places.
# answers to the worker are typically fast to reply.


# requires the implementation of the waiting one.
# NOTE: this can be higher level than this one. 
# this is going to be interesting.

# can delay the application of the new results after they are available.



def fn(search_space_fn, v_lst, eval_fn):
    (inputs, outputs, hs) = search_space_fn()
    se.specify(outputs.values(), hs.values())

    v = eval_fn(inputs, outputs, hs)
    return v




    # the coordinator has to be adjusted to the workers
    # at least in the type of information that it is expecting.
    # the other ones may also be running.
    # some of the other ones may only be done if 
    # if it does not matter.
    # NOTE: only updated when 

    # NOTE: the coordinators and the searchers have to be matched.
    # I can ask, what is the index of something.
    # this can be better done by the searcher.
    # after how many updates to the cost function do we do 
    # something about it.

    # NOTE: perhaps do things to avoid repetitions.

# because of the information, it does require more stuff.
# it requires the use of

# example coordinator for the case where there are multiple models.

# NOTE: this is going to change a lot the updates for the MCTS and others.
# this is because I can no longer assume that they are synchronous.

# depends on the searcher. 
# if there I want to stuff at the end of each cycle or not.


# will need to keep all the information.

# should be able to do a test gpu, test all searchers.
# generate plots easily.

# TODO: also, perhaps guarantee that there are no repeats.
# this may or may not be desirable.

# TODO: it should be possible to use it to run the model, but 
# also to do the update.

# wraps a bunch of stuff.

# NOTE: this is going to be rethought for the incremental case.

# NOTE: for now, I'm just ignoring the incremental case. this is going to
# be interesting to do later, but it should be super similar.


# NOTE: the result available update needs to have the information about 
# the configuration that was ran. it is essentially a new update for the 
# searcher.

# NOTE: this is based on a searcher.

        
# TODO: it can terminate the processes in the case where there are many 
# things going on.

# this would be the most straighforward coordinator based on a searcher 
# and not asking for many interesting things between how to interleave 
# computation and update.

# NOTE: to make this simpler, I will have to change the model to make sure 
# that it allows for this waiting functionality that can be done offline,
# or at least put it on the side.

# NOTE: this 

# NOTE: because this is remote, it makes things more difficult to 
# how to get things to work with multiple machines.
# the data has to loaded multiple times.

# TODO: it would be convenient to do the optimization by using some of the
# cycles of the model.

# NOTE: the most important part is making sure that I make good use of 
# the hardware.