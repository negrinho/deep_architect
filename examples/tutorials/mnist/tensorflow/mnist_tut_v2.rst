Simple Architecture Search for MNIST Tutorial
+++++++++++++++++++++++++++++++++++++++++++++

Goals
=====

-  Introduce the basic building blocks of Neural Architecture Search and
   DeepArchitect, using the canonical MNIST handwritten-digit
   classification task.
-  Demonstrate how to construct a simple neural architecture system with
   DeepArchitect.

Prerequesite
============

-  Have DeepArchitect installed (link to Installation page)
-  Have at least one of these framework installed: TensorFlow, Keras,
   PyTorch, DyNet
-  Experience with using one of the above framework to develop and train
   deep learning models

Overview
========

What is Architecture Search?
----------------------------

Developing and training Deep Learning architectures is a trial-and-error
process that requires expert experience. Architecture Search is the
process designed to alleviate ML practitioners from this painstaking
process, by automatically searching for the best Deep Learning
architectures for a given task.

Architecture Search is commonly decomposed into three main subproblems:

-  Designing the search space
-  Designing the searcher
-  Evaluating the sampled architecture

| We will go into what each of this means in this tutorial, using MNIST
  as a concrete example.
| We will often refer to the TensorFlow code below from the TensorFlow
  tutorial (link)

.. code:: python

   import tensorflow as tf
   mnist = tf.keras.datasets.mnist
       
   (x_train, y_train),(x_test, y_test) = mnist.load_data()
   x_train, x_test = x_train / 255.0, x_test / 255.0
       
   model = tf.keras.models.Sequential([
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(512, activation=tf.nn.relu),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
   ])
   model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
       
   model.fit(x_train, y_train, epochs=5)
   model.evaluate(x_test, y_test)

Running The Program
-------------------

-  Link to respective examples
-  Point PYTHONPATH to deep_architect
-  do python2 …
-  show output

Search Space
============

Brief Overview of Search Space
------------------------------

A search space consists of computational modules (link to Module
section) that we want to search over and sample architectures. Each
module can be any operation that we want, be it affine transform, 2d
convolution, pooling, or as basic as addition and subtract. There are
also pre-defined modules such as siso_sequential
(single-input-single-output) to connect modules in a sequential manner,
and siso_or to choose a module out of 2 options. For each user-defined
module we need to specify:

-  Compile function: This function is called once to define all
   parameters associated with a module (eg weights and bias for an
   affine operation), before evaluating any architecture.
-  Forward function: This function computes the value at a specific
   evaluation, thus is called at every architecture evaluation. This is
   similar to the dynamic computation concepts in deep learning
   frameworks PyTorch and DyNet.

The topology of the architecture is entirely up to the users. Thus, we
can incorporate any inductive bias about the tasks at hand, making this
a useful tool for architecture exploration. More in depth details and
examples of search spaces can be found in Search Space section.

A Very Simple MNIST Search Space
--------------------------------

For this tutorial, based on the above MLP-based architecture (link), we
will design a search space (dubbed A Very Simple MNIST Search Space)
with the following specifications:

-  The general topology is flatten -> dnn cell -> dense -> softmax,
   where dnn cell is a densely-connected neural network cell consisting
   of an affine layer, nonlinearity activation, and optional layers
   dropout and batch normalization.
-  The dnn cell can be repeated once or twice
-  Hyperparameters specifications

   -  Number of hidden units for dense layer is selected among [64, 128,
      256, 512, 1024] except for the last dense layer, in which the
      units is the number of classes
   -  The activation function is selected among [‘relu’, ‘tanh’, ‘elu’]
   -  The dropout rate is selected among [0.25, 0.5, 0.75].

Fun fact: Our search space consists of x architectures.

Computational Modules
---------------------

Let’s start with the following code:

.. code:: python

   import tensorflow as tf 
   import numpy as np

   import deep_architect.modules as mo 
   import deep_architect.hyperparameters as hp  
   from deep_architect.contrib.useful.search_spaces.tensorflow.common import siso_tfm

   D = hp.Discrete # Discrete Hyperparameter

Two important aspect of DeepArchitect Computational Modules:

-  D is a discrete hyperparameter instance. Using this instance we can
   easily define hyperparameters for the search space. For example, the
   list of hidden units to choose from would be D([64, 128, 256, 512,
   1024]). More hyperparameter instances here (link to Hyperparameter)
-  siso_tfm: base module wrapper around single-input-single-output
   TensorFlow modules. This function takes in name of module (eg
   ‘Dense’), implemented compile function, a dictionary mapping
   hyperparameter name to hyperparameter values, and it returns the
   inputs and outputs connections of a module. More in siso_tfm link.

We now define a densely-connected computational modules:

.. code:: python


   def dense(h_units): 
       def cfn(di, dh): # compile function 
           Dense = tf.keras.layers.Dense(dh['units'])
           def fn(di): # forward function 
               return {'Out' : Dense(di['In'])}
           return fn
       return siso_tfm('Dense', cfn, {'units' : h_units})

-  The computational module above takes in hyperparameter of hidden
   units (eg D([64, 128, 256, 512, 1024])) and return inputs and outputs
   connections (more in search_space_construct).
-  The compile function takes in dictionary mapping input name to input
   value (di) and hyperparameter dictionary (dh). The parameters and
   computational function are defined here. It returns the forward
   function.
-  The forward function takes in the input dictionary and propagates the
   computation forward, returning the dictionary mapping output name to
   the value.

Based on the above principles, we can define the rest of the building
blocks for our A Very Simple MNIST Search Space (flatten, nonlinearity,
dropout, batch norm) as follows:

.. code:: python

   def flatten(): 
       def cfn(di, dh): 
           Flatten = tf.keras.layers.Flatten() 
           def fn(di): 
               return {'Out': Flatten(di['In'])} 
           return fn
       return siso_tfm('Flatten', cfn, {})

   def nonlinearity(h_nonlin_name):
       def cfn(di, dh):
           def fn(di):
               nonlin_name = dh['nonlin_name']
               if nonlin_name == 'relu':
                   Out = tf.keras.layers.Activation('relu')(di['In'])
               elif nonlin_name == 'tanh':
                   Out = tf.keras.layers.Activation('tanh')(di['In'])
               elif nonlin_name == 'elu':
                   Out = tf.keras.layers.Activation('elu')(di['In'])
               else: 
                   raise ValueError
               return {"Out" : Out}
           return fn
       return siso_tfm('Nonlinearity', cfn, {'nonlin_name' : h_nonlin_name})

   def dropout(h_keep_prob):
       def cfn(di, dh):
           Dropout = tf.keras.layers.Dropout(dh['keep_prob'])
           def fn(di):
               return {'Out' : Dropout(di['In'])}
           return fn
       return siso_tfm('Dropout', cfn, {'keep_prob' : h_keep_prob})

   def batch_normalization():
       def cfn(di, dh):
           bn = tf.keras.layers.BatchNormalization()
           def fn(di):
               return {'Out' : bn(di['In'])}
           return fn
       return siso_tfm('BatchNormalization', cfn, {})

Having defined these computational modules, we will now construct our
search space.

DeepArchitect Implementation of A Very Simple MNIST Search Space
----------------------------------------------------------------

.. code:: python

   def dnn_net_simple(num_classes): 

           # defining hyperparameter
           h_num_hidden = D([64, 128, 256, 512, 1024]) # number of hidden units for dense module 
           h_nonlin_name = D(['relu', 'tanh', 'elu']) # nonlinearity function names to choose from
           h_opt_drop = D([0, 1]) # dropout optional hyperparameter; 0 is exclude, 1 is include 
           h_drop_keep_prob = D([0.25, 0.5, 0.75]) # dropout probability to choose from 
           h_opt_bn = D([0, 1]) # batch_norm optional hyperparameter
           h_perm = D([0, 1]) # order of swapping for permutation 
           h_num_repeats = D([1, 2]) # 1 is appearing once, 2 is appearing twice
           
           # defining search space topology 
           model = mo.siso_sequential([
               flatten(),
               mo.siso_repeat(lambda: mo.siso_sequential([
                   dense(h_num_hidden),
                   nonlinearity(h_nonlin_name),
                   mo.siso_permutation([
                       lambda: mo.siso_optional(lambda: dropout(h_drop_keep_prob), h_opt_drop),
                       lambda: mo.siso_optional(batch_normalization, h_opt_bn),
                   ], h_perm)
               ]), h_num_repeats),
               dense(D([num_classes]))
           ])
           
           return model 

Our Discrete Hyperparameter instance can encapsulate many functionality.
h_num_hidden, h_nonlin_name, h_drop_keep_prob have been explained above.
h_opt_drop and h_opt_bn are hyperparameters for whether to include (1)
or exclude (0) dropout and batch norm modules. h_perm is the permutation
hyperparameter, with permutation 0 meaning dropout first and batch norm
second (in the order that we define), and permutation 1 in reverse
order.

The topology of our search space is very similar to the baseline
architecture. Both have similar modules and both connect each module
sequentially. We deconstruct some of the pre-defined functions below:

-  siso_sequential (link) connects the given modules sequentiallly.
   Input is the list of modules to connect.
-  siso_repeat (link) repeats a given module a number of time. Inputs
   are function returning the module and repeat hyperparameter
   (h_num_repeats)
-  siso_optional (link) decides whether to include or omit the module.
   Inputs are similar to siso_repeat, with optional hyperparameter
   instead of repeat hyperparameter
-  siso_permutation (link) decides the ordering of the given modules.
   Inputs are a list of function returning the modules and permutation
   hyperparameter (h_swap)

These pre-defined modules are very useful in defining the topology of
our search space. More in modules (link)

To make things a bit more modular, we refactor the search space above
into a dnn_net and dnn_cell, such that it is more inline with our search
space description (link):

.. code:: python


   def dnn_cell(h_num_hidden, h_nonlin_name, h_swap, h_opt_drop, h_opt_bn, h_drop_keep_prob):
       return mo.siso_sequential([
           dense(h_num_hidden),
           nonlinearity(h_nonlin_name),
           mo.siso_permutation([
               lambda: mo.siso_optional(lambda: dropout(h_drop_keep_prob), h_opt_drop),
               lambda: mo.siso_optional(batch_normalization, h_opt_bn),
           ], h_swap)])

   def dnn_net(num_classes):
       h_nonlin_name = D(['relu', 'tanh', 'elu'])
       h_swap = D([0, 1])
       h_opt_drop = D([0, 1])
       h_opt_bn = D([0, 1])
       return mo.siso_sequential([
           flatten(), 
           mo.siso_repeat(lambda: dnn_cell(
               D([64, 128, 256, 512, 1024]),
               h_nonlin_name, h_swap, h_opt_drop, h_opt_bn,
               D([0.25, 0.5, 0.75])), D([1, 2])),
           dense(D([num_classes]))])

After designing our search space, we then need a way to sample the
architecture. This can be done with a searcher

Searcher
========

Searcher is a searching algorithm that determines the way we sample an
architecture in the search space. More details are in Searcher section.
In this tutorial, we are using Random search for simplicity. Other
choices included Monte Carlo Tree Search, Sequential-Model Based
Optimization, Efficient Neural Architecture Search, and Evolutionary.

Searcher class takes in a function that when called returns a new search
space, i.e., a search space where all the hyperparameters have not being
specified. This function is implemented below

.. code:: python

   import deep_architect.searchers.random as se
   import deep_architect.core as co 

   def get_search_space(num_classes):
       def fn(): 
           co.Scope.reset_default_scope()
           inputs, outputs = dnn_net(num_classes)
           return inputs, outputs, {}
       return fn

The function that get_search_space returns is input into the Searcher.
Inside this function we reset the scope (link) and call the search space
that we constructed above.

Random Searcher can be used as follows:

.. code:: python

   num_classes = 10 
   searcher = se.RandomSearcher(get_search_space(num_classes))
   inputs, outputs, hs, _, searcher_eval_token = searcher.sample() # sampling an architecture
   # evaluating the architecture here, returning val_acc 
   searcher.update(val_acc, searcher_eval_token) # update the searcher with val_acc, not needed for Random Search but important for others

Let’s move on to see how we can evaluate the sampled architecture.

.. code:: python

   import deep_architect.searchers.random as se
   import deep_architect.core as co 

   def get_search_space(num_classes):
       def fn(): 
           co.Scope.reset_default_scope()
           inputs, outputs = dnn_net(num_classes)
           return inputs, outputs, {}
       return fn

Evaluating the Architecture
===========================

Once we sample an architecture, we need to evaluate how good this
architecture is. This typically involves the normal training and
validating procedure. The evaluator typically returns the best
validation metric (accuracy, F1, etc.) of each architecture, and we
select the best architecture based on that. More in Evaluator section.

An evaluator for this tutorial is implemented below. This simple
evaluator only trains the model and return result dictionary (which only
contains the validation accuracy, but can contain more like number of
parameters, training time, etc). More complex evaluator for mnist
utilizing different training tricks such as early stopping and reduced
learning rate can be found here (link)

.. code:: python

   class SimpleClassifierEvaluator:

       def __init__(self, train_dataset, num_classes, max_num_training_epochs=20, 
                   batch_size=256, learning_rate=1e-3):

           self.train_dataset = train_dataset
           self.num_classes = num_classes
           self.max_num_training_epochs = max_num_training_epochs
           self.learning_rate = learning_rate
           self.batch_size = batch_size
           self.val_split = 0.1 # 10% of dataset for validation

       def evaluate(self, inputs, outputs, hs):
           tf.keras.backend.clear_session() 
           tf.reset_default_graph()

           (x_train, y_train) = self.train_dataset

           X = tf.keras.layers.Input(x_train[0].shape)
           co.forward({inputs['In'] : X})  
           logits = outputs['Out'].val
           probs = tf.keras.layers.Softmax()(logits)
           model = tf.keras.models.Model(inputs=[inputs['In'].val], outputs=[probs])
           model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), 
                       loss='sparse_categorical_crossentropy', 
                       metrics=['accuracy'])
           model.summary() 
           history = model.fit(x_train, y_train, 
                               batch_size=self.batch_size, 
                               epochs=self.max_num_training_epochs, 
                               validation_split=self.val_split)

           results = {'val_acc': history.history['val_acc'][-1]}
           return results 

The important part to note is the following lines

.. code:: python

   X = tf.keras.layers.Input(x_train[0].shape)
   co.forward({inputs['In'] : X})  
   logits = outputs['Out'].val
   probs = tf.keras.layers.Softmax()(logits)
   model = tf.keras.models.Model(inputs=[inputs['In'].val], outputs=[probs])

We call ``co.forward({inputs['In'] : X})`` to forward the input through
the graph. inputs[‘In’] is an Input object (link). We can then retrieve
the output through value of Output object (link) ``outputs['Out'].val``

With the search space, searcher, and evaluator implemented, we put
everything together in the main function below

.. code:: python


   def main():

       num_classes = 10
       num_samples = 3 # number of architecture to sample 
       best_val_acc, best_architecture = 0., -1

       # load and normalize data 
       mnist = tf.keras.datasets.mnist 
       (x_train, y_train), (x_test, y_test) = mnist.load_data()
       x_train, x_test = x_train / 255.0, x_test / 255.0
       
       # defining evaluator and searcher 
       evaluator = SimpleClassifierEvaluator((x_train, y_train), num_classes,
                                               max_num_training_epochs=5) 
       searcher = se.RandomSearcher(get_search_space(num_classes))

       for i in xrange(num_samples):
           print("Sampling architecture %d" % i)
           inputs, outputs, hs, _, searcher_eval_token = searcher.sample()
           val_acc = evaluator.evaluate(inputs, outputs, hs)['val_acc'] # evaluate and return validation accuracy
           print("Finished evaluating architecture %d, validation accuracy is %f" % (i, val_acc))
           if val_acc > best_val_acc: 
               best_val_acc = val_acc
               best_architecture = i
           searcher.update(val_acc, searcher_eval_token)
       print("Best validation accuracy is %f with architecture %d" % (best_val_acc, best_architecture)) 

   if __name__ == "__main__": 
       main() 

We have finished constructing our architecture search system for MNIST!

(Note that we omitted testing, which would require logging knowledge to
log the best parameters from the best architecture (among other things
to log). See logging section below)

Optional - Logging and Visualization
====================================

Link to logging mnist example Link to visualization mnist example
