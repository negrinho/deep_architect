import os
import portalocker

from deep_architect.contrib.communicators.communicator import Communicator
from deep_architect.contrib.communicators.file_utils import (consume_file,
                                                             read_file,
                                                             write_file)


class FileCommunicator(Communicator):

    def __init__(self,
                 num_procs,
                 dirname='file_comm',
                 worker_queue_file='worker_queue',
                 worker_results_prefix='worker_results_'):
        # make directory where communication files are created
        try:
            os.makedirs(dirname)
        except OSError:
            pass

        # claim a rank for the process
        lock = portalocker.Lock(os.path.join(dirname, 'init'),
                                mode='a+',
                                flags=portalocker.LOCK_EX)
        lock.acquire()
        fh = lock.fh
        fh.seek(0)
        curnum = fh.read()
        if len(curnum) is 0:
            rank = 0
        else:
            rank = int(curnum)

        if rank >= num_procs:
            raise ValueError('Number of processes > the number of workers')
        fh.seek(0)
        fh.truncate(0)
        fh.write(str(rank + 1))
        lock.release()

        super(FileCommunicator, self).__init__(num_procs - 1, rank)
        self.worker_queue_file = os.path.join(dirname, worker_queue_file)
        self.worker_results_prefix = os.path.join(dirname,
                                                  worker_results_prefix)
        self.done = False

    def _publish_results_to_master(self, results, evaluation_id,
                                   searcher_eval_token):
        write_file(self.worker_results_prefix + str(self.rank),
                   (results, evaluation_id, searcher_eval_token))

    def _receive_architecture_in_worker(self):
        while not self.done:
            file_data = consume_file(self.worker_queue_file)
            # continue looping until there is something in the queue file
            if file_data is None:
                continue

            # if kill signal is given, return None, otherwise return contents of file
            vs, evaluation_id, searcher_eval_token, kill = file_data
            if kill:
                write_file(self.worker_results_prefix + str(self.rank), 'done')
                self.done = True
                return None

            return vs, evaluation_id, searcher_eval_token
        return None

    def _is_ready_to_publish_architecture(self):
        file_data = read_file(self.worker_queue_file)
        return file_data is None

    def _publish_architecture_to_worker(self, vs, current_evaluation_id,
                                        searcher_eval_token):
        write_file(self.worker_queue_file,
                   (vs, current_evaluation_id, searcher_eval_token, False))

    def _receive_results_in_master(self, src):
        result = consume_file(self.worker_results_prefix + str(src + 1))
        if result == 'done':
            self.finished += 1
            return None
        return result

    def _kill_worker(self):
        write_file(self.worker_queue_file, (0, 0, 0, True))
