"""
Library of computational modules for search space. This library is customized 
to work with any backend, C++ DyNet being one of them. Unlike other libraries for 
TensorFlow and PyTorch, these modules don't require compile and forward functions, 
as these functions should be implemented by the backend. See Modules.cpp within 
the DyNet folder for more details. 
NOTE: for multiple inputs modules (eg Concat), both inputs need to be connected, 
otherwise search space can break down
"""

import deep_architect.core as co 
import deep_architect.modules as mo 
import deep_architect.hyperparameters as hp 
import numpy as np 

D = hp.Discrete # Discrete hyperparameter class

def module(name, name_to_hyperp, input_list, output_list, scope=None):
    """This general module works for multiple input/output. Simple wrapper
    around DeepArchitect Module 
    Args: 
        name (str): name of module 
        name_to_hyperp (Dict[str, darch.hyperparameter]: 
            Dictionary mapping hyperparameter names to hyperparameter 
        input_list (list[str]): names of inputs to current module. eg: ["In"], ["In1"], ["Pad"], etc. 
        output_list (list[str]): names of outputs of current module 
        scope (darch.Scope): scope that module is going to be registered in 
    """  
    m = co.Module(scope, name) 
    m._register(input_list, output_list, name_to_hyperp)
    return m.get_io()

def siso_module(name, name_to_hyperp, input_list=None, output_list=None, scope=None):
    """Simple wrapper around single iutput single output (siso) modules
    """
    if (input_list == None): input_list = ["In"]
    if (output_list == None): output_list = ["Out"]
    m = co.Module(scope, name) 
    m._register(input_list, output_list, name_to_hyperp)
    return m.get_io() 

def concat(h_dim):
    """Concatenate (exactly) 2 modules together 
    Args: 
        h_dim (darch.hyperparameter): Hyperparameter of dimension to concatenate along. Per convention, 
            0 is row, 1 is column, etc. 

    todo: (1) Extend to have variable inputs. Basically a hyperparam defined by user, 
            then create list inside function (like in multi_embedding)
          (2) Add padding option along other dimensions?
    """ 
    return module('Concatenate', {'dim': h_dim}, ['In0', 'In1'], ['Out'])

def nonlinearity(h_nonlin_name): 
    """Args: 
        h_nonlin_name (darch.hyperparameter): Hyperparameter of nonlinearity functions to search over 
            Right now support "tanh", "sigmoid", "elu", "relu"

    todo: (1) Support more nonlinearity functions
          (2) Make it case insensitive? 
    """
    return siso_module("Nonlinearity", {'nonlin_name': h_nonlin_name})

def affine(h_u):
    """Affine Transformation Module (Wx + b)
    Args: 
        h_u (darch.hyperparameter): Hyperparameter of hidden units to search over
    """ 
    return siso_module('Affine', {'units': h_u})

def dropout(h_keep_prob):
    """Args: 
        h_keep_prob (darch.hyperparameter): Hyperparameter of dropout probabilities
    """  
    return siso_module('Dropout', {'keep_prob': h_keep_prob})

def batch_normalization():
    return siso_module('BatchNormalization', {})

# h_level: 0 for word level, 1 for char level 
def embedding_lookup(name, h_level, h_use_pretrained, h_num_feature):
    """Embedding Lookup Module for 1 feature. Takes an input and output the 
    embeddings of such input and the embedding of padding 
    Args: 
        name (string): Name of feature to lookup (word, character, capitalization, predicate, etc) 
        h_level (darch.hyperparameter): Level of embedding, 0 for word embedding, 
            1 for character embedding 
        h_use_pretrained (darch.hyperparameter): 1 if want to use pretrained embeddings, 
            0 otherwise. If want to use pretrained, need to specify the details 
            of the pretrained embeddings (hash, pretrained, etc.) when compiling 
        h_num_feature (darch.hyperparameter): Number of features in the embeddings 
    
    """ 
    return module('Embedding', {'level': h_level, 
                                'name': name, 
                                'use_pretrained': h_use_pretrained, 
                                'num_feature': h_num_feature},
                    ['In'], ['Out', 'Pad'])

def multi_embedding_lookup(features, h_feature_select):
    """Generalized version of Embedding Lookup module. Support embedding lookup 
    for multiple features (words, capitalization, pos, etc.)
    Args: 
        features (list[string, darch.hyperparameter, darch.hyperparameter]): list of 
            (name, use_pretrained, num_feature), where name is name of feature, 
            use_pretrained is hyperparameter to specify whether we want to use 
            pretrained embeddings, and num_feature is the number of embedding features
        h_feature_select (darch.hyperparameter): list of selected features to 
            search over. For example, [['word', 'capitalization'], ['word']] 
            means 2 options to search over: use word feature and capitalization
            feature, or just use word feature 0 
    """ 
    hyps = dict()
    for i in xrange(len(features)): 
        (name, use_pretrained, num_feature) = features[i] 
        hyps['use_pretrained_%s' % name] = use_pretrained
        hyps['num_feature_%s' % name] = num_feature 
    hyps['features_selected'] = h_feature_select
    return module('MultiEmbeddings', hyps, ['In'], ['Out', 'Pad']) 

def window(supply_pad, h_num_win):
    """Window Module. Similar to convolution operation, but on sequence data. 
    Args: 
        supply_pad (bool): specify whether the user will supply padding. Else 
            implicit padding with 0
        h_num_win (darch.hyperparameter): hyperparameters of size of window  
    
    todo: (1) is it useful to support non-padding (like a valid option in convolution?)
    """
    if (supply_pad):
        return module('Window', {'num_win': h_num_win}, ['In', 'Pad'], ['Out'])
    else:
        return siso_module('Window', {'num_win': h_num_win})

def reverse():
    """Reverse a matrix or a vector
    todo: support which dimension to reverse 
    """
    return siso_module("Reverse", {}) 

def lstm(h_layer, h_units, h_direction):
    """Long Short-Term Memory Cell 
    Args: 
        h_layer (darch.hyperparameter): number of hidden layers 
        h_units (darch.hyperparameter): number of hidden units 
        h_direction (darch.hyperparameter): 0 is forward, 1 is backward 
    """
    return siso_module("LSTM", {'layer': h_layer, 'hidden_units': h_units, 'direction': h_direction})

def gru(h_layer, h_units, h_direction):
    """Gated Recurrent Unit Cell 
    Args: 
        h_layer (darch.hyperparameter): number of hidden layers 
        h_units (darch.hyperparameter): number of hidden units 
        h_direction (darch.hyperparameter): 0 is forward, 1 is backward 
    """
    return siso_module("GRU", {'layer': h_layer, 'hidden_units': h_units, 'direction': h_direction})

def pick(h_indexes, h_dim):
    """Pick the element with index(es)
    Args:
        h_indexes (darch.hyperparameters): list of indexes to pick.
            0 is first index. -1 is last index
        h_dim (darch.hyperparameters): dimension to pick  
    """

    return siso_module("Pick", {'indexes': h_indexes, 'dim': h_dim})

def remove(h_indexes, h_dim):
    """Args:
        h_indexes (darch.hyperparameters): list of indexes to remove.
            0 is first index. -1 is last index
        h_dim (darch.hyperparameters): dimension to remove  
    """
    return siso_dym("Remove", {'indexes': h_indexes, 'dim': h_dim}) 

def word2char():
    """Convert word to a list of character. NOTE: this currently supports
    converting embeddings only. 
    
    todo: (1) add support for converting raw words as well? 
    """
    return siso_module("Word2Char", {}) 

def reshape(h_dim):
    """Args: 
        h_dim (darch.hyperparameters): dimension to get rid of. -1 is batch dimension. 
                0, 1, 2, etc. are normal dimensions
    """ 
    return siso_module("Reshape", {'dim': h_dim})

def crf(h_class):
    """Conditional Random Field Module 
    """
    return siso_module("CRF", {'num_class': h_class})

def highway_connection(h_units):
    """Residual connection for RNNs. Mostly based on paper by He et al. 
    https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf
    Args: 
        h_units (darch.hyperparameters): must be equal to the hidden units of 
            the RNNs that employ this highway connection 
    """ 
    return module("HighwayConn", {'hidden_units': h_units}, ["h_prime", "x"], ["Out"]) 

def neg_log_softmax(): 
    """Negative Log Softmax Loss""" 
    return siso_module("NegLogSoftmax", {})
