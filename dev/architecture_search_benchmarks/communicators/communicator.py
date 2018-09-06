"""
This is an abstract class that wraps communication for architecture search in a
master-worker model.
"""
class Communicator(object):
    def __init__(self, num_workers, rank):
        self.num_workers = num_workers
        self.rank = rank
        self.finished = 0
    
    def is_master(self):
        return self.rank == 0
    
    def is_worker(self):
        return self.rank > 0
    
    def get_rank(self):
        return self.rank
    
    def publish_results_to_master(self, results, evaluation_id, searcher_eval_token):
        if not self.is_worker():
            raise ValueError("Master cannot publish results")
        self._publish_results_to_master(self, results, evaluation_id, searcher_eval_token)
    
    def _publish_results_to_master(self, results, evaluation_id, searcher_eval_token):
        raise NotImplementedError
    
    def receive_architecture_in_worker(self):
        if not self.is_worker():
            raise ValueError("Master cannot receive architecture")
        return self._receive_architecture_in_worker()

    def _receive_architecture_in_worker(self):
        raise NotImplementedError
    
    def is_ready_to_publish_architecture(self):
        if not self.is_master():
            raise ValueError("Worker cannot publish architecture")
        self._is_ready_to_publish_architecture()
    
    def _is_ready_to_publish_architecture(self):
        raise NotImplementedError

    def publish_architecture_to_worker(self, vs, current_evaluation_id, 
                                       searcher_eval_token):
        if not self.is_master():
            raise ValueError("Worker cannot publish architecture")
        self._publish_architecture_to_worker(self, vs, current_evaluation_id, 
                                             searcher_eval_token)
    
    def _publish_architecture_to_worker(self, vs, current_evaluation_id, 
                                        searcher_eval_token):
        raise NotImplementedError
    
    def receive_results_in_master(self, src):
        if not self.is_master():
            raise ValueError("Worker cannot receive results")
        return self._receive_results_in_master(src)

    def _receive_results_in_master(self, src):
        raise NotImplementedError
    
    def kill_worker(self):
        if not self.is_master():
            raise ValueError("Worker cannot kill another worker")
        return self._kill_worker()

    def _kill_worker(self):
        raise NotImplementedError

def get_communicator(name, num_procs=2):
    from file_communicator import FileCommunicator
    from mpi_communicator import MPICommunicator
    if name == 'mpi':
        return MPICommunicator()
    elif name == 'file':
        return FileCommunicator(num_procs)