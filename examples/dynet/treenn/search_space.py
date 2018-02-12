import dynet as dy

import darch.core as co
import darch.hyperparameters as hps
import darch.utils as ut

# NOTE: may need a dynet helper.
import darch.tensorflow_helpers as tf_helpers

import darch.searchers as se
import darch.modules as mo

import numpy as np

TFM = tf_helpers.TFModule
D = hps.Discrete

def SISODyM(name, compile_fn, name_to_h={}, scope=None):
    ### NOTE: this is going to be similar to it.
    # return tf_helpers.TFModule(name, name_to_h, compile_fn, 
    #         ['In'], ['Out'], scope)


def ReLU():
    def cfn():
        def fn(In):
            return {'Out' : dy.
        tf.nn.relu(In)}
        return fn
    return SISODyM('ReLU', cfn)

def Affine(h_m, h_W_init_fn, h_b_init_fn):
    def cfn(In, m, W_init_fn, b_init_fn):
        shape = In.get_shape().as_list()
        n = np.product(shape[1:])
        W = tf.Variable( W_init_fn( [n, m] ) )
        b = tf.Variable( b_init_fn( [m] ) )
        def fn(In):
            if len(shape) > 2:
                In = tf.reshape(In, [-1, n])
            Out = tf.add(tf.matmul(In, W), b)
            # print In.get_shape().as_list()
            return {'Out' : Out}
        return fn
    return SISODyM('Affine', cfn, 
        {'m' : h_m, 'W_init_fn' : h_W_init_fn, 'b_init_fn' : h_b_init_fn})

def Dropout(h_keep_prob):
    def cfn(keep_prob):
        p = tf.placeholder(tf.float32)
        def fn(In):
            return {'Out' : tf.nn.dropout(In, p)} 
        return fn, {p : keep_prob}, {p : 1.0} 
    return SISODyM('Dropout', cfn, {'keep_prob' : h_keep_prob})
    
# TODO: perhaps add hyperparameters.
def BatchNormalization():
    def cfn():
        p_var = tf.placeholder(tf.bool)
        def fn(In):
            return {'Out' : tf.layers.batch_normalization(In, training=p_var) }
        return fn, {p_var : 1}, {p_var : 0}     
    return SISODyM('BatchNormalization', cfn)

def Conv2D(h_num_filters, h_filter_size, h_stride, h_W_init_fn, h_b_init_fn):
    def cfn(In, num_filters, filter_size, stride, W_init_fn, b_init_fn):
        (_, height, width, channels) = In.get_shape().as_list()

        W = tf.Variable( W_init_fn( [filter_size, filter_size, channels, num_filters] ) )
        b = tf.Variable( b_init_fn( [num_filters] ) )
        def fn(In):
            return {'Out' : tf.nn.bias_add(
                tf.nn.conv2d(In, W, [1, stride, stride, 1], 'SAME'), b)}
        return fn

    return SISODyM('Conv2D', cfn, {
        'num_filters' : h_num_filters, 
        'filter_size' : h_filter_size, 
        'stride' : h_stride,
        'W_init_fn' : h_W_init_fn, 
        'b_init_fn' : h_b_init_fn,
        })

def MaxPool(h_kernel_size, h_stride):
    def cfn(kernel_size, stride):
        def fn(In):
            return {'Out' : tf.nn.max_pool(In, 
                [1, kernel_size, kernel_size, 1], [1, stride, stride, 1], 'SAME')}
        return fn
    return SISODyM('MaxPool', cfn, {
        'kernel_size' : h_kernel_size, 
        'stride' : h_stride,
        })

def Add():
    return tf_helpers.TFModule('Add', {}, 
        lambda: lambda In0, In1: tf.add(In0, In1), 
        ['In0', 'In1'], ['Out'])

def AffineSimplified(h_m):
    def cfn(In, m):
        shape = In.get_shape().as_list()
        n = np.product(shape[1:])

        def fn(In):
            if len(shape) > 2:
                In = tf.reshape(In, [-1, n])
            return {'Out' : tf.layers.dense(In, m)}
        return fn
    return SISODyM('AffineSimplified', cfn, {'m' : h_m})

def Nonlinearity(h_or):
    def cfn(idx):
        def fn(In):
            if idx == 0:
                Out = tf.nn.relu(In)
            elif idx == 1:
                Out = tf.nn.relu6(In)
            elif idx == 2:
                Out = tf.nn.crelu(In)
            elif idx == 3:
                Out = tf.nn.elu(In)
            elif idx == 4:
                Out = tf.nn.softplus(In)
            else:
                raise ValueError
            return {"Out" : Out}
        return fn
    return SISODyM('Nonlinearity', cfn, {'idx' : h_or})

def DNNCell(h_num_hidden, h_nonlin, h_swap, 
        h_opt_drop, h_opt_bn, h_drop_keep_prob):
    ms = [
        AffineSimplified(h_num_hidden),
        Nonlinearity(h_nonlin),
        mo.SISOPermutation([
            lambda: io_fn( 
                mo.SISOOptional(
                    lambda: io_fn( Dropout(h_drop_keep_prob) ), 
                    h_opt_drop ) ),
            lambda: io_fn( 
                mo.SISOOptional(
                    lambda: io_fn( BatchNormalization() ), 
                    h_opt_bn ) ),
        ], h_swap),
        mo.Empty()
    ]
    ut.connect_sequentially(ms)
    return io_lst_fn( ms )

io_fn = lambda m: (m.inputs, m.outputs)
io_lst_fn = lambda m_lst: (m_lst[0].inputs, m_lst[-1].outputs)





class TreeLSTMBuilder(object):
    def __init__(self, model, word_vocab, wdim, hdim):
        self.WS = [model.add_parameters((hdim, wdim)) for _ in "iou"]
        self.US = [model.add_parameters((hdim, 2*hdim)) for _ in "iou"]
        self.UFS =[model.add_parameters((hdim, hdim)) for _ in "ff"]
        self.BS = [model.add_parameters(hdim) for _ in "iouf"]
        self.E = model.add_lookup_parameters((len(word_vocab),wdim))
        self.w2i = word_vocab

    def expr_for_tree(self, tree, decorate=False):
        assert(not tree.isleaf())
        if len(tree.children) == 1:
            assert(tree.children[0].isleaf())
            emb = self.E[self.w2i.get(tree.children[0].label,0)] 
            Wi,Wo,Wu   = [dy.parameter(w) for w in self.WS]
            bi,bo,bu,_ = [dy.parameter(b) for b in self.BS]
            i = dy.logistic(dy.affine_transform([bi, Wi, emb]))
            o = dy.logistic(dy.affine_transform([bo, Wo, emb]))
            u = dy.tanh(    dy.affine_transform([bu, Wu, emb]))
            c = dy.cmult(i,u)
            h = dy.cmult(o,dy.tanh(c))
            if decorate: tree._e = h
            return h, c
        assert(len(tree.children) == 2),tree.children[0]
        e1, c1 = self.expr_for_tree(tree.children[0], decorate)
        e2, c2 = self.expr_for_tree(tree.children[1], decorate)
        Ui,Uo,Uu = [dy.parameter(u) for u in self.US]
        Uf1,Uf2 = [dy.parameter(u) for u in self.UFS]
        bi,bo,bu,bf = [dy.parameter(b) for b in self.BS]
        e = dy.concatenate([e1,e2])
        i = dy.logistic(dy.affine_transform([bi, Ui, e]))
        o = dy.logistic(dy.affine_transform([bo, Uo, e]))
        f1 = dy.logistic(dy.affine_transform([bf, Uf1, e1]))
        f2 = dy.logistic(dy.affine_transform([bf, Uf2, e2]))
        u = dy.tanh(     dy.affine_transform([bu, Uu, e]))
        c = dy.cmult(i,u) + dy.cmult(f1,c1) + dy.cmult(f2,c2)
        h = dy.cmult(o,dy.tanh(c))
        if decorate: tree._e = h
        return h, c

train = read_dataset("examples/dynet/treenn/data/trees/train.txt")
dev = read_dataset("examples/dynet/treenn/data/trees/dev.txt")

l2i, w2i, i2l, i2w = get_vocabs(train)

model = dy.Model()
builder = TreeLSTMBuilder(model, w2i, args.WEMBED_SIZE, args.HIDDEN_SIZE)
W_ = model.add_parameters((len(l2i), args.HIDDEN_SIZE))
trainer = dy.AdamTrainer(model)
trainer.set_clip_threshold(-1.0)
trainer.set_sparse_updates(True if args.SPARSE == 1 else False)