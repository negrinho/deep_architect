### date and time
import datetime
import sys

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

# TODO: do not just return a string representation, return the numbers.
def now(omit_date=False, omit_time=False, time_before_date=False):
    assert (not omit_time) or (not omit_date)

    d = datetime.datetime.now()

    date_s = ''
    if not omit_date:
        date_s = "%d-%.2d-%.2d" % (d.year, d.month, d.day)
    time_s = ''
    if not omit_time:
        time_s = "%.2d:%.2d:%.2d" % (d.hour, d.minute, d.second)
    
    vs = []
    if not omit_date:
        vs.append(date_s)
    if not omit_time:
        vs.append(time_s)

    # creating the string
    if len(vs) == 2 and time_before_date:
        vs = vs[::-1]
    s = '|'.join(vs)

    return s

def now_dict():
    x = datetime.datetime.now()

    return  create_dict(
        ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond']
        [x.year, x.month, x.day, x.hour, x.minute, x.second, x.microsecond])

### logging
import time
import os

### TODO: add calendar. or clock. or both. check formatting for both.

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

class MemoryTracker:
    def __init__(self):
        self.last_registered = 0.0
        self.max_registered = 0.0

    def memory_total(self, units='mb'):
        mem_now = mbs_process(os.getpid())
        if self.max_registered < mem_now:
            self.max_registered = mem_now

        return convert_between_byte_units(mem_now, dst_units=units)
        
    def memory_since_last(self, units='mb'):
        mem_now = self.memory_total()        
        
        mem_dif = mem_now - self.last_registered
        self.last_registered = mem_now

        return convert_between_byte_units(mem_dif, dst_units=units)

    def memory_max(self, units='mb'):
        return convert_between_byte_units(self.max_registered, dst_units=units)

def print_time(timer, pref_str='', units='s'):
    print('%s%0.2f %s since start.' % (pref_str, 
        timer.time_since_start(units=units), units) )
    print("%s%0.2f %s seconds since last call." % (pref_str, 
        timer.time_since_last(units=units), units) )

def print_memory(memer, pref_str='', units='mb'):
    print('%s%0.2f %s total.' % (pref_str, 
        memer.memory_total(units=units), units.upper()) )
    print("%s%0.2f %s since last call." % (pref_str, 
        memer.memory_since_last(units=units), units.upper()) )
    
def print_memorytime(memer, timer, pref_str='', mem_units='mb', time_units='s'):
    print_memory(memer, pref_str, units=mem_units)
    print_time(timer, pref_str, units=time_units)    

def print_oneliner_memorytime(memer, timer, pref_str='', 
        mem_units='mb', time_units='s'):

    print('%s (%0.2f %s last; %0.2f %s total; %0.2f %s last; %0.2f %s total)'
                 % (pref_str,
                    timer.time_since_last(units=time_units),
                    time_units, 
                    timer.time_since_start(units=time_units),
                    time_units,                     
                    memer.memory_since_last(units=mem_units),
                    mem_units.upper(), 
                    memer.memory_total(units=mem_units),
                    mem_units.upper()) )

class Logger:
    def __init__(self, fpath, 
        append_to_file=False, capture_all_output=False):
        
        if append_to_file:
            self.f = open(fpath, 'a')
        else:
            self.f = open(fpath, 'w')

        if capture_all_output:
            capture_output(self.f)
    
    def log(self, s, desc=None, preappend_datetime=False):
        
        if preappend_datetime:
            self.f.write( now() + '\n' )
        
        if desc is not None:
            self.f.write( desc + '\n' )
        
        if not isinstance(s, str):
            s = pprint.pformat(s)

        self.f.write( s + '\n' )

# check if files are flushed automatically upon termination, or there is 
# something else that needs to be done. or maybe have a flush flag or something.
def capture_output(f, capture_stdout=True, capture_stderr=True):
    """Takes a file as argument. The file is managed externally."""
    if capture_stdout:
        sys.stdout = f
    if capture_stderr:
        sys.stderr = f
