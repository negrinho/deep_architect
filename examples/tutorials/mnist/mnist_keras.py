# Using TensorFlow Backend 

# Search Space 
import keras 
import numpy as np

import deep_architect.modules as mo  
from deep_architect.contrib.useful.search_spaces.tensorflow.common import siso_tfm, D

def Flatten(): 
    def cfn(di, dh): 
        def fn(di): 
            return {'Out': keras.layers.Flatten() (di['In'])} 
        return fn
    return siso_tfm('Flatten', cfn, {}) # use siso_tfm for now 

def Dense(h_units): 
    def cfn(di, dh):
        def fn(di):
            return {'Out' : keras.layers.Dense(dh['units']) (di['In'])}
        return fn
    return siso_tfm('Dense', cfn, {'units' : h_units})

def Nonlinearity(h_nonlin_name):
    def cfn(di, dh):
        def fn(di):
            nonlin_name = dh['nonlin_name']
            if nonlin_name == 'relu':
                Out = keras.layers.Activation('relu')(di['In'])
            elif nonlin_name == 'tanh':
                Out = keras.layers.Activation('tanh')(di['In'])
            elif nonlin_name == 'elu':
                Out = keras.layers.Activation('elu')(di['In'])
            elif nonlin_name == 'softplus':
                Out = keras.layers.Activation('softplus')(di['In'])
            else:
                raise ValueError
            return {"Out" : Out}
        return fn
    return siso_tfm('Nonlinearity', cfn, {'nonlin_name' : h_nonlin_name})

def Dropout(h_keep_prob):
    def cfn(di, dh):
        def fn(di):
            return {'Out' : keras.layers.Dropout(dh['keep_prob'])(di['In'])}
        return fn
    return siso_tfm('Dropout', cfn, {'keep_prob' : h_keep_prob})

def BatchNormalization():
    def cfn(di, dh):
        def fn(di):
            return {'Out' : keras.layers.BatchNormalization()(di['In'])}
        return fn
    return siso_tfm('BatchNormalization', cfn, {})

def dnn_net_simple(num_classes): 

        # declaring hyperparameter
        h_nonlin_name = D(['relu', 'tanh', 'elu', 'softplus']) # nonlinearity function names to choose from
        h_opt_drop = D([0, 1]) # dropout optional hyperparameter; 0 is exclude, 1 is include 
        h_drop_keep_prob = D([0.25, 0.5, 0.75]) # dropout probability to choose from 
        h_opt_bn = D([0, 1]) 
        h_num_hidden = D([64, 128, 256, 512, 1024]) # number of hidden units for affine transform module 
        h_swap = D([0, 1]) # order of swapping for permutation 
        h_num_repeats = D([1, 2]) # 1 is appearing once, 2 is appearing twice
        
        # defining search space topology 
        model = mo.siso_sequential([
            Flatten(),
            mo.siso_repeat(lambda: mo.siso_sequential([
                Dense(h_num_hidden),
                Nonlinearity(h_nonlin_name),
                mo.siso_permutation([
                    lambda: mo.siso_optional(lambda: Dropout(h_drop_keep_prob), h_opt_drop),
                    lambda: mo.siso_optional(BatchNormalization, h_opt_bn),
                ], h_swap)
            ]), h_num_repeats),
            Dense(D([num_classes]))
        ])
        
        return model 

def dnn_cell(h_num_hidden, h_nonlin_name, h_swap, h_opt_drop, h_opt_bn, h_drop_keep_prob):
    return mo.siso_sequential([
        Dense(h_num_hidden),
        Nonlinearity(h_nonlin_name),
        mo.siso_permutation([
            lambda: mo.siso_optional(lambda: Dropout(h_drop_keep_prob), h_opt_drop),
            lambda: mo.siso_optional(BatchNormalization, h_opt_bn),
        ], h_swap)])

def dnn_net(num_classes):
    h_nonlin_name = D(['relu', 'tanh', 'elu', 'softplus'])
    h_swap = D([0, 1])
    h_opt_drop = D([0, 1])
    h_opt_bn = D([0, 1])
    return mo.siso_sequential([
        Flatten(), 
        mo.siso_repeat(lambda: dnn_cell(
            D([64, 128, 256, 512, 1024]),
            h_nonlin_name, h_swap, h_opt_drop, h_opt_bn,
            D([0.25, 0.5, 0.75])), D([1, 2])),
        Dense(D([num_classes]))])

# Main/Searcher 
import deep_architect.searchers.random as se
import deep_architect.core as co 
from keras.datasets import mnist

def get_search_space(num_classes):
    def fn(): 
        co.Scope.reset_default_scope()
        inputs, outputs = dnn_net(num_classes)
        return inputs, outputs, {}
    return fn

def main():
    num_classes = 10
    num_samples = 3
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    evaluator = SimpleClassifierEvaluator((x_train, y_train), (x_test, y_test), num_classes,
                './temp', max_num_training_epochs=5, log_output_to_terminal=True) # defining evaluator 

    searcher = se.RandomSearcher(get_search_space(num_classes))
    best_val_acc, best_architecture = 0., -1
    for i in xrange(num_samples):
        print("Sampling architecture %d" % i)
        inputs, outputs, hs, _, searcher_eval_token = searcher.sample()
        val_acc = evaluator.evaluate(inputs, outputs, hs)['validation_accuracy'] # evaluate and return validation accuracy
        print("Finished evaluating architecture %d, validation accuracy is %f" % (i, val_acc))
        if val_acc > best_val_acc: 
            best_val_acc = val_acc
            best_architecture = i
        searcher.update(val_acc, searcher_eval_token)
    print("Best validation accuracy is %f with architecture %d" % (best_val_acc, best_architecture)) 

# Evaluator 

class SimpleClassifierEvaluator:

    def __init__(self, train_dataset, val_dataset, num_classes, model_path,
            max_num_training_epochs=10, optimizer_type='adam', batch_size=256,
            learning_rate=1e-3, display_step=1, log_output_to_terminal=True, 
            test_dataset=None):

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_classes = num_classes

        self.max_num_training_epochs = max_num_training_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer_type = optimizer_type
        self.log_output_to_terminal = log_output_to_terminal
        self.display_step = display_step
        self.model_path = model_path
        self.test_dataset = test_dataset

    def get_optimizer(self): 
        optimizer = None 
        if self.optimizer_type == 'adam':
            optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        elif self.optimizer_type == 'sgd':
            optimizer = keras.optimizers.SGD(lr=self.learning_rate)
        elif self.optimizer_type == 'sgd_mom':
            optimizer = keras.optimizers.SGD(lr=self.learning_rate, momentum=0.99)
        else:
            raise ValueError("Unknown optimizer.")
        return optimizer 

    def evaluate(self, inputs, outputs, hs):
        keras.backend.clear_session() 

        lr = self.learning_rate 
        (x_train, y_train) = self.train_dataset
        (x_val, y_val) = self.val_dataset

        X = keras.layers.Input(x_train[0].shape)
        co.forward({inputs['In'] : X})
        logits = outputs['Out'].val
        probs = keras.layers.Softmax()(logits)

        model = keras.models.Model(inputs=[inputs['In'].val], outputs=[probs])
        optimizer = self.get_optimizer()
        model.compile(optimizer=optimizer, 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])
        model.summary() 
        model.fit(x_train, y_train, 
                batch_size=self.batch_size, 
                epochs=self.max_num_training_epochs, 
                validation_data=(x_val, y_val))

        [val_loss, val_metric] = model.evaluate(x_val, y_val)
        results = {'validation_loss': val_loss, 
                    'validation_accuracy': val_metric}
        return results 
        

if __name__ == "__main__": 
    main() 