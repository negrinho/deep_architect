# TODO: 
# (2) RNN module 
from dynet import DyParameterCollection

M = DyParameterCollection() 

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

