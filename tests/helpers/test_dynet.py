import dynet as dy
import deep_architect.core as co
import deep_architect.searchers.random as se
import deep_architect.modules as mo

# TODO:
# (2) RNN module
# Create a general search space in "deep_learning_backend"
from deep_architect.helpers.dynet import DyParameterCollection, siso_dym

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


def test_search_space(num_classes):
    return mo.siso_sequential([
        affine(D([8])),
        nonlinearity(D(['tanh'])),
        affine(D([num_classes])),
        nonlinearity(D(['sigmoid']))
    ])

def ss_fn():
    co.Scope.reset_default_scope()
    num_classes = 1
    inputs, outputs = test_search_space(num_classes)
    return inputs, outputs, {}

# create training instances
def create_xor_instances(num_rounds=2000):
    x = []
    y = []
    for i in range(num_rounds):
        for x1 in 0,1:
            for x2 in 0,1:
                answer = 0 if x1==x2 else 1
                x.append((x1,x2))
                y.append(answer)
    return x, y

def almost_equal(x, y):
    return (abs(x-y) < 10**(-8))

def evaluate(x_train, y_train, inputs, outputs, hs):
    trainer = dy.SimpleSGDTrainer(M.get_collection())
    total_loss = 0
    seen_instances = 0
    correct = 0
    for (x_ins, y_ins) in zip(x_train, y_train):
        dy.renew_cg() # new computation graph b/c dynamic network
        x = dy.vecInput(len(x_ins)) # like tf.placeholder
        x.set(x_ins)
        y = dy.scalarInput(y_ins)
        co.forward({inputs['In'] : x})
        logits = dy.reshape(outputs['Out'].val, (1,))
        if (logits.value() >= 0.5 and almost_equal(y_ins, 1)) or (logits.value() < 0.5 and almost_equal(y_ins, 0)):
            correct += 1
        loss = dy.binary_log_loss(logits, y)
        seen_instances += 1
        total_loss += loss.value()
        loss.backward()
        trainer.update()
        if (seen_instances > 1 and seen_instances % 10 == 0):
            print "average loss is:", total_loss/seen_instances
    return {'val_accuracy': correct/(1.0*seen_instances)}

def main():
    num_sample = 1
    (x, y) = create_xor_instances()
    searcher = se.RandomSearcher(ss_fn)
    for _ in xrange(num_sample):
        M.renew_collection()
        inputs, outputs, hs, _, searcher_eval_token = searcher.sample()
        val_acc = evaluate(x, y, inputs, outputs, hs)['val_accuracy']
        print "val_acc=", val_acc
        searcher.update(val_acc, searcher_eval_token)

if __name__ == '__main__':
    main()