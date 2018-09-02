# TODO: 
# (1) Some tricks to hide ParameterCollection in the background, like with scope 
# or dynet computational graph? 
# (2) RNN module 

import dynet as dy 
import deep_architect.modules as mo 
import deep_architect.core as co
import deep_architect.hyperparameters as hp

class DyParameterCollection(dy.ParameterCollection): 
    """Wrapper around dynet ParameterCollection object

    DyNet ParameterCollection is an object that holds the parameters of the model. 
    DyNet Module needs this to be able to add parameters into the model (using 
    dynet.ParameterCollection.add_parameters() or .add_lookup_parameters()). This 
    wrapper object needs only to be created once, import to search space modules 
    to add parameters, and renew every time a new architecture is sampled. 
    
    """
    def __init__(self):
        self.params = dy.ParameterCollection() 

    def renew_collection(self):
        """Renew every time new architecture is sampled to clear out old parameters"""  
        self.params = dy.ParameterCollection()  

    def get_collection(self):
        """Call inside a search space module, and when needs to add to trainer"""  
        return self.params 

M = DyParameterCollection() 

D = hp.Discrete

class DyModule(co.Module): 

    def __init__(self, name, compile_fn, name_to_hyperp, input_names, 
                output_names, scope=None): 
        co.Module.__init__(self, scope, name)
        self._register(input_names, output_names, name_to_hyperp)
        self._compile_fn = compile_fn 

    def _compile(self): 
        input_name_to_val = self._get_input_values() 
        hyperp_name_to_val = self._get_hyperp_values()

        self._fn = self._compile_fn(input_name_to_val, hyperp_name_to_val)
        
    def _forward(self): 
        input_name_to_val = self._get_input_values()
        output_name_to_val = self._fn(input_name_to_val)
        self._set_output_values(output_name_to_val) 

    def _update(self): pass  


def siso_dym(name, compile_fn, name_to_hyperp, scope=None): 
    "Dynet module for single-input, single-output"
    return DyModule(name, compile_fn, name_to_hyperp, ['In'], ['Out'], scope).get_io()

# just put here to streamline everything 
def nonlinearity(h_nonlin_name): 
    def cfn(di, dh): 
        def fn(di): 
            nonlin_name = dh['nonlin_name']
            if nonlin_name == 'relu': 
                Out = dy.rectify(di['In'])
            elif nonlin_name == 'elu': 
                Out = dy.elu(di['In'])
            elif nonlin_name == 'sigmoid': 
                Out = dy.logistic(di['In'])
            elif nonlin_name == 'tanh':
                Out = dy.tanh(di['In'])
            else: 
                raise ValueError
            return {'Out': Out}
        return fn 
    return siso_dym('Nonlinearity', cfn, {'nonlin_name' : h_nonlin_name})

def dropout(h_keep_prob): 
    def cfn(di, dh): 
        p = dh['keep_prop']
        def fn(di): 
            return {'Out': dy.dropout(di['In'], p)}
        return fn 
    return siso_dym('Dropout', cfn, {'keep_prop': h_keep_prob})


def affine(h_u):
    def cfn(di, dh):
        shape = di['In'].dim() # ((r, c), batch_dim)
        m, n = dh['units'], shape[0][0]
        pW = M.get_collection().add_parameters((m, n))
        pb = M.get_collection().add_parameters((m, 1))
        def fn(di): 
            In = di['In']
            W, b = pW.expr(), pb.expr()
            return {'Out': W*In + b}
        return fn 
    return siso_dym('Affine', cfn, {'units': h_u})

