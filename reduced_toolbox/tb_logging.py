import os
import time
import psutil
import reduced_toolbox.tb_resources as tb_re


def memory_process(pid, units='mb'):
    psutil_p = psutil.Process(pid)
    mem_p = psutil_p.memory_info()[0]

    return tb_re.convert_between_byte_units(mem_p, dst_units=units)


def convert_between_time_units(x, src_units='s', dst_units='h'):
    units = ['s', 'm', 'h', 'd', 'w']
    assert src_units in units and dst_units in units
    d = {}
    d['s'] = 1.0
    d['m'] = 60.0 * d['s']
    d['h'] = 60.0 * d['m']
    d['d'] = 24.0 * d['h']
    d['w'] = 7.0 * d['d']
    return (x * d[src_units]) / d[dst_units]


class MemoryTracker:
    def __init__(self):
        self.last_registered = 0.0
        self.max_registered = 0.0

    def memory_total(self, units='mb'):
        mem_now = memory_process(os.getpid(), units)
        if self.max_registered < mem_now:
            self.max_registered = mem_now

        return tb_re.convert_between_byte_units(mem_now, dst_units=units)

    def memory_since_last(self, units='mb'):
        mem_now = self.memory_total('b')

        mem_dif = mem_now - self.last_registered
        self.last_registered = mem_now

        return tb_re.convert_between_byte_units(mem_dif, dst_units=units)

    def memory_max(self, units='mb'):
        return tb_re.convert_between_byte_units(self.max_registered, dst_units=units)


class TimeTracker:
    def __init__(self):
        self.init_time = time.time()
        self.last_registered = self.init_time

    def time_since_start(self, units='s'):
        now = time.time()
        elapsed = now - self.init_time

        return convert_between_time_units(elapsed, dst_units=units)

    def time_since_last(self, units='s'):
        now = time.time()
        elapsed = now - self.last_registered
        self.last_registered = now

        return convert_between_time_units(elapsed, dst_units=units)