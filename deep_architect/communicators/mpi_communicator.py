from mpi4py import MPI
from deep_architect.communicators.communicator import Communicator
"""
Tags for the requests used by the communicator
"""
READY_REQ = 0
MODEL_REQ = 1
RESULTS_REQ = 2
"""
Contains implementation for MPI based Communicator. All requests used are
non-blocking unless mentioned otherwise in comments.
"""


class MPICommunicator(Communicator):

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        super(MPICommunicator, self).__init__(self.comm.Get_size() - 1,
                                              self.comm.Get_rank())
        self.done = False
        if self.is_master():
            self.ready_requests = [
                self.comm.irecv(source=i + 1, tag=READY_REQ)
                for i in range(self.num_workers)
            ]
            self.eval_requests = [
                self.comm.irecv(source=i + 1, tag=RESULTS_REQ)
                for i in range(self.num_workers)
            ]
            self.next_worker = -1

    """
    Called in worker process only.
    Synchronously sends the results from worker to master. Returns nothing.
    """

    def _publish_results_to_master(self, results, evaluation_id,
                                   searcher_eval_token):
        self.comm.ssend((results, evaluation_id, searcher_eval_token),
                        dest=0,
                        tag=RESULTS_REQ)

    """
    Called in worker process only.
    Receives architecture from master. Synchronously sends a ready request to
    master signalling that worker is ready to receive new architecture. Then
    does a blocking receive of the architecture sent by master and returns the
    architecure. If master instead sends a kill signal, returns None for that
    and any future invocations of _receive_architecture_in_worker.
    """

    def _receive_architecture_in_worker(self):
        if self.done:
            return None

        self.comm.ssend([self.rank], dest=0, tag=READY_REQ)

        (vs, evaluation_id, searcher_eval_token, kill) = self.comm.recv(
            source=0, tag=MODEL_REQ)

        if kill:
            self.done = True
            return None

        return vs, evaluation_id, searcher_eval_token

    """
    Called in master process only.
    Iterates through ready requests and checks if any of them have been
    returned. If so, set the next worker corresponding to the ready request that
    returned, and return True. If none of the workers have sent back a ready
    request, return False.
    """

    def _is_ready_to_publish_architecture(self):
        for idx, req in enumerate(self.ready_requests):
            if req:
                test, msg = req.test()
                if test:
                    self.next_worker = idx + 1
                    return True
        return False

    """
    Called in master process only.
    Sends architecture to the worker that was designated as ready in
    _is_ready_to_publish_architecture. Then resets the ready request for that
    worker. Returns nothing.
    """

    def _publish_architecture_to_worker(self, vs, current_evaluation_id,
                                        searcher_eval_token):
        self.comm.isend((vs, current_evaluation_id, searcher_eval_token, False),
                        dest=self.next_worker,
                        tag=MODEL_REQ)
        self.ready_requests[self.next_worker - 1] = (self.comm.irecv(
            source=self.next_worker, tag=READY_REQ))

    """
    Called in master process only.
    Checks if the src worker has sent back results. If so, returns the result
    and resets the request to get results in the future. Else, returns None.
    """

    def _receive_results_in_master(self, src):
        test, msg = self.eval_requests[src].test()
        if test:
            self.eval_requests[src] = self.comm.irecv(
                source=src + 1, tag=RESULTS_REQ)
        return msg if test else None

    """
    Called in master process only.
    Sends a kill signal to given worker. Doesn't return anything.
    """

    def _kill_worker(self):
        self.comm.isend((0, 0, 0, True), dest=self.next_worker, tag=MODEL_REQ)
        self.ready_requests[self.next_worker - 1] = None
