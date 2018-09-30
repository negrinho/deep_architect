import portalocker
import pickle


def clear_file(fh):
    fh.seek(0)
    fh.truncate(0)


def consume_file(filename):
    lock = portalocker.Lock(filename, mode='a+b', flags=portalocker.LOCK_EX)
    lock.acquire()
    fh = lock.fh
    fh.seek(0)
    if len(fh.read()) is 0:
        file_data = None
    else:
        fh.seek(0)
        file_data = pickle.load(fh)
    if file_data is not None:
        clear_file(fh)
        pickle.dump(None, fh)
    lock.release()
    return file_data


def read_file(filename):
    lock = portalocker.Lock(filename, mode='a+b', flags=portalocker.LOCK_EX)
    lock.acquire()
    fh = lock.fh
    fh.seek(0)
    if len(fh.read()) is 0:
        file_data = None
    else:
        fh.seek(0)
        file_data = pickle.load(fh)
    lock.release()
    return file_data


def write_file(filename, obj):
    file_data = 0
    while file_data is not None:
        lock = portalocker.Lock(
            filename, mode='a+b', flags=portalocker.LOCK_EX)
        lock.acquire()
        fh = lock.fh
        fh.seek(0)
        if len(fh.read()) is 0:
            file_data = None
        else:
            fh.seek(0)
            file_data = pickle.load(fh)
        if file_data is None:
            clear_file(fh)
            pickle.dump(obj, fh)
        lock.release()
