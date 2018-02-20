from collections import OrderedDict
from six import iterkeys, itervalues, iteritems

class OrderedSet:
    def __init__(self):
        self.d = OrderedDict()
    
    def add(self, x):
        if x not in self.d:
            self.d[x] = None
    
    def update(self, xs):
        for x in xs:
            if x not in self.d:
                self.d[x] = None
    
    def __len__(self):
        return len(self.d)

    def __iter__(self):
        for x in iterkeys(self.d):
            yield x
    
    def __contains__(self, item):
        return item in self.d

# TODO: it is possible to structure the scope in more nicely to keep 
# hyperparameters, inputs, outputs, and modules separated.
class Scope:
    def __init__(self):
        self.name_to_elem = OrderedDict()
        self.elem_to_name = OrderedDict()

    def register(self, x, name):
        assert name not in self.name_to_elem

        self.name_to_elem[name] = x
        self.elem_to_name[x] = name

    def get_unused_name(self, prefix):
        i = 0
        while True:
            name = prefix + str(i)
            if name not in self.name_to_elem:
                break
            i += 1
        return name

    def get_name(self, x):
        return self.elem_to_name[x]

    @staticmethod
    def reset_default_scope():
        Scope.default_scope = Scope()

Scope.default_scope = Scope()

class Addressable:
    def __init__(self, scope, name):
        scope.register(self, name)
        self.scope = scope
    
    def __repr__(self):
        return self.get_name()   

    def get_name(self):
        return self.scope.get_name(self)

    def _get_base_name(self):
        return self.__class__.__name__

class Hyperparameter(Addressable):
    def __init__(self, scope=None, name=None):
        if scope is None:
            scope = Scope.default_scope

        if name is None:
            prefix = '.'.join(['H', self._get_base_name() + '-'])
        else:
            prefix = '.'.join(['H', name + '-'])
            
        name = scope.get_unused_name(prefix)
        Addressable.__init__(self, scope, name)

        self.set_done = False
        self.modules = OrderedSet()

    def is_set(self):
        return self.set_done
    
    def set_val(self, v):
        assert not self.set_done
        self._check_val(v)
        
        self.set_done = True
        self.val = v

        for x in self.modules:
            x._update()

    def get_val(self):
        assert self.set_done
        return self.val

    def _register_module(self, x):
        self.modules.add(x)

    def _check_val(self, v):
        pass
    
class Input(Addressable):
    def __init__(self, module, scope, name):
        name = '.'.join([module.get_name(), 'I', name])
        Addressable.__init__(self, scope, name)

        self.module = module
        self.from_out = None

    def is_connected(self):
        return self.from_out is not None

    def get_connected_output(self):
        return self.from_out

    def get_module(self):
        return self.module

    def connect(self, from_out):
        assert isinstance(from_out, Output) 
        assert self.from_out is None

        self.from_out = from_out
        from_out.to_ins.append(self)
    
    def disconnect(self):
        assert self.from_out is not None
        self.from_out.to_ins.remove(self)
        self.from_out = None

    def reroute_connected_output(self, to_in):
        assert isinstance(to_in, Input)

        old_ox = self.from_out
        self.disconnect()
        old_ox.connect(to_in)

class Output(Addressable):
    def __init__(self, module, scope, name):
        name = '.'.join([module.get_name(), 'O', name])
        Addressable.__init__(self, scope, name)

        self.module = module
        self.to_ins = []

    def is_connected(self):
        return len(self.to_ins) > 0

    def get_connected_inputs(self):
        return self.to_ins
    
    def get_module(self):
        return self.module

    def connect(self, to_in):
        to_in.connect(self)

    def disconnect_all(self):
        for ix in self.to_ins:
            ix.disconnect()

    # NOTE: the inputs that self connects to are rerouted to a different output.
    def reroute_all_connected_inputs(self, from_out):
        for ix in self.to_ins:
            ix.disconnect()
            ix.connect(from_out)

class Module(Addressable):
    def __init__(self, scope=None, name=None):
        if scope is None:
            scope = Scope.default_scope

        if name is None:
            prefix = '.'.join(['M', self._get_base_name() + '-'])
        else:
            prefix = '.'.join(['M', name + '-'])
        
        name = scope.get_unused_name(prefix)
        Addressable.__init__(self, scope, name)
        
        self.inputs = OrderedDict()
        self.outputs = OrderedDict()
        self.hs = OrderedDict()
        self._is_compiled = False
    
    def _register_input(self, name):
        assert name not in self.inputs
        self.inputs[name] = Input(self, self.scope, name)

    def _register_output(self, name):
        assert name not in self.inputs
        self.outputs[name] = Output(self, self.scope, name)

    def _register_hyperparameter(self, h, name):
        assert isinstance(h, Hyperparameter) and name not in self.hs 
        self.hs[name] = h
        h._register_module(self)

    def _update(self):
        pass

    def _compile(self):
        pass
    
    def _forward(self):
        raise NotImplementedError

    def forward(self):
        if not self._is_compiled:
            self._compile()
            self._is_compiled = True
        self._forward()

def extract_unique_modules(in_or_out_or_mod_lst):
    ms = OrderedSet()
    for x in in_or_out_or_mod_lst:
        if isinstance(x, Input):
            ms.add( x.get_module() )
        elif isinstance(x, Output):
            ms.add( x.get_module() )
        elif isinstance(x, Module):
            ms.add(x)
        else:
            raise ValueError
    return list(ms)

def extract_unique_inputs(input_or_module_lst):
    inputs = OrderedSet()
    for x in input_or_module_lst:
        if isinstance(x, Input):
            inputs.add(x)
        elif isinstance(x, Module):
            inputs.update( x.inputs.values() )
        else:
            raise ValueError
    return list(inputs)

def extract_unique_outputs(output_or_module_lst):
    outputs = OrderedSet()
    for x in output_or_module_lst:
        if isinstance(x, Output):
            outputs.add(x)
        elif isinstance(x, Module):
            outputs.update( x.outputs.values() )
        else:
            raise ValueError
    return list(outputs)

# assumes that the inputs provided are sufficient to evaluate all the network.
def determine_module_eval_seq(input_lst):
    ms = extract_unique_modules(input_lst)
    
    module_seq = []
    module_memo = set()
    input_memo = set(input_lst)
    for m in ms:
        if m not in module_memo and all(ix in input_memo for ix in itervalues(m.inputs)):
            module_seq.append(m)
            module_memo.add(m)

            for ox in itervalues(m.outputs):
                ix_lst = ox.get_connected_inputs()
                input_memo.update(ix_lst)
                m_lst = [ix.get_module() for ix in ix_lst]
                ms.extend(m_lst)

    return module_seq

def backward_traverse(module_lst, fn, memo=None):
    if memo == None:
        memo = set()

    ms = [m for m in module_lst if m not in memo]
    for m in ms:
        is_over = fn(m)
        if is_over:
            break
        else:
            for ix in itervalues(m.inputs):
                if ix.is_connected():
                    m_prev = ix.get_connected_output().get_module()
                    
                    if m_prev not in memo:
                        memo.add(m_prev)
                        ms.append(m_prev)

def forward_traverse(module_lst, fn, memo=None):
    if memo == None:
        memo = set()
    
    ms = [m for m in module_lst if m not in memo]
    for m in ms:
        is_over = fn(m)
        if is_over:
            break
        else:
            for ox in itervalues(m.outputs):
                if ox.is_connected():
                    for ix in ox.get_connected_inputs():
                        m_next = ix.get_module()
                    
                        if m_next not in memo:
                            memo.add(m_next)
                            ms.append(m_next)

def is_specified(module_lst):
    is_spec = [True]
    def fn(x):
        for h in itervalues(x.hs):
            if not h.is_set():
                is_spec[0] = False
                return True
        return False
    backward_traverse(module_lst, fn)
    return is_spec[0]

def get_unset_hyperparameters(module_lst):
    assert not is_specified(module_lst)

    hs = OrderedSet() 
    def fn(x):
        for h in itervalues(x.hs):
            if not h.is_set():
                hs.add(h)
        return False
    backward_traverse(module_lst, fn)
    return hs

# NOTE: forward needs to be efficient in the dynamic case.
# precompute the evaluation sequence and apply in that case.
def forward(input_to_val, _module_seq=None):
    if _module_seq is None:
        inputs = input_to_val.keys()
        _module_seq = determine_module_eval_seq(inputs)

    for ix, v in iteritems(input_to_val):
        ix.val = v

    for x in _module_seq:
        x.forward()
        for ox in itervalues(x.outputs):
            for ix in ox.get_connected_inputs():
                ix.val = ox.val