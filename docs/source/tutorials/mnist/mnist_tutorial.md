MNIST with TensorFlow Tutorial
===
Prerequisite 
---
• (Required) This tutorial assumes basic ML knowledge. ML crash course by Google provides a good overview  https://developers.google.com/machine-learning/crash-course/ml-intro  
• (Required) DeepArchitect installed (Installation page)  
• (Required) TensorFlow installed (https://www.tensorflow.org/install/). We recommend the virtualenv method.  
• (Required) Experience using tensorflow or Keras.   
• (Optional) Familiarity with MNIST handwritten digit task, its tensorflow implementation, and neural networks is recommended. 

Goals
--- 
• Introduce the basic building blocks of DeepArchitect via image classification with MNIST a well-known example in ML.  
• Demonstrate how to construct a simple neural architecture system with DeepArchitect. Specifically, we will construct a search space for the task, used a simple random searcher to sample a number of candidate architecture, and write a simple evaluator for the architecture.  

Task introduction 
---
• The task is recognizing handwritten digits numbers in the MNIST dataset. (figure 1: Picture handwritten digit) This is perhaps the most widely-used example in ML today, dubbed "Hello World of Machine Learning". The MNIST dataset consists of 60,000 examples in the training set, and 10,000 examples in the test set. Each example is a 28x28 monochrome image of a handwritten digits from 0 - 9.  Since this tutorial is based on MNIST Tensorflow implementation, we highly recommend understanding the Tensorflow implementation first: https://www.tensorflow.org/tutorials/

```python
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
```

More in depth about basic classification here: https://www.tensorflow.org/tutorials/keras/basic_classification (they replaced mnist with a 'hipper" fashion-mnist, meaning replacing 10 handwritten digits with 10 clothing items, but the fundamental ideas are the same)

Introduction to Neural Networks
---
• Densely-connected Neural Networks is the basic neural network architecture. It's also called Multi-Layer Perceptron (MLP).  
• The architecture used in the above code snippet can be visualized in figure 2. Based on this baseline architecture, we can construct a search space for our search.   
• Figure 2: graph of architecture in the tutorial.   

Running the program (perhaps better to add a script) 
---
• The main file for this tutorial is under darch/examples/tensorflow/mnist/main.py.  
• To run the program:
	○ Set PYTHONPATH to point to darch location
	○ Activate tensorflow ifusing virtualenv. 
	○ In darch directory, run python2 examples/tensorflow/mnist/main.py 
	○ Show picture of running program - figure 3. 

Getting start
---
• Let's set up our DeepArchitect program. Create a file call main.py (perhaps better name?) and add the following code (note: to have line in code tidbit): 

```python

from darch.contrib.datasets.loaders import load_mnist
from darch.contrib.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from darch.contrib.datasets.dataset import InMemoryDataset
import darch.contrib.search_spaces.tensorflow.dnn as css_dnn
import darch.modules as mo
import darch.searchers as se
	
class SSF0(mo.SearchSpaceFactory): pass 
	
def main():
    num_classes = 10 # number of handwritten digits 
    num_samples = 16 # number of architecture to sample 
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = load_mnist('data/mnist')
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)
        
if __name__ == '__main__':
    main()

```

• Code description: line 1-6 import useful functions such as load_mnist, SimpleClassifierEvaluator, and InMemoryDataset. Line 4 import a search space function that we will implement in a different file. Line 7 the Search Space Factory that we will implement as part of generating our search space. In this tutorial we are sampling and comparing 16 architectures. The last 4 lines load, split, and store dataset into conventional train, validation, and test data. (Skip talking about InMemoryDataset)   
• We will continue add and construct the system as we go. 

Search Space 
---
* Brief overview of search space:  
	* a search space consists of computational module (link to Module section) what we want to search over. Each module can be any operation that we want, be it affine transform, 2d convolution, pooling, or as basic as addition and subtract. There are also pre-defined modules such as siso_sequential (single-input-single-output) to connect modules in a sequential manner, siso_or to choose a module out of 2 options. For each user-defined module we need to specify (1) compile function (2) forward function. Compile function is called once to define all parameters associated with a module (think weights W and bias b for an affine operation), before evaluating any architecture. Forward function computes the value at a specific evaluation, thus is called at every architecture evaluation. This is similar to the dynamic computation concepts in deep learning frameworks  PyTorch and DyNet (add a bit about why that is useful?)   
	* The topology of the architecture is entirely up to us. Thus, we can incorporate any inductive bias about the tasks at hand, making this a useful tool for architecture exploration. More in depth details and examples of search spaces can be found in Search Space section. 

* Table of our proposed search space (graph, or how to best represent search space?)
	* Decompose into computational cell 
	* This search space has x architectures in total 
* Code snippet of the search space
	* Create a file called dnn.py and import the following 

```python

import darch.modules as mo  
import tensorflow as tf 
import numpy as np
from darch.contrib.search_spaces.tensorflow.common import siso_tfm, D

```

* Siso_tfm: base module wrapper around single-input-single-output modules. Input is "module_name", compile function, and dictionary mapping hyperparemter name to value 
* D is discrete hyperparameters (more in Hyperparameter section) 
* Computational modules
  * Affine Transform operation:  this module defines a compile function. 
    * Input is the hyperparameter of hidden units. Inputs to compile function are (1) di - dictionary of input values and (2) dh - dictionary mapping hyperparameter to values. 
    * Compile function get the shape and a single product of the shape (to flatten in forward function)
    * Forward function gets the actual input, flatten it by reshapping, and call tensorflow dense function (affine transform) on input and number of hidden units (dh['m']). Returns dictionary mapping output name to output values.

```python

def affine_simplified(h_m): 
    def compile_fn(di, dh):
        shape = di['In'].get_shape().as_list()
        n = np.product(shape[1:])
            def fn(di):
                In = di['In']
                if len(shape) > 2:
                    In = tf.reshape(In, [-1, n])
                return {'Out' : tf.layers.dense(In, dh['m'])}
        return fn
    return siso_tfm('AffineSimplified', compile_fn, {'m' : h_m})

```

* Nonlinearity: with similar structure (compile + forward functions) like affine transform, we have nonlinearity module 
  * H_nonlin_name is hyperparameter for nonlinearity, consists of different nonlinearity function names. 

```python

def nonlinearity(h_nonlin_name):
    def compile_fn(di, dh):
        def fn(di):
            nonlin_name = dh['nonlin_name']
            if nonlin_name == 'relu':
                Out = tf.nn.relu(di['In'])
            elif nonlin_name == 'relu6':
                Out = tf.nn.relu6(di['In'])
            elif nonlin_name == 'crelu':
                Out = tf.nn.crelu(di['In'])
            elif nonlin_name == 'elu':
                Out = tf.nn.elu(di['In'])
            elif nonlin_name == 'softplus':
                Out = tf.nn.softplus(di['In'])
            else:
                raise ValueError
            return {"Out" : Out}
    return fn
return siso_tfm('Nonlinearity', compile_fn, {'nonlin_name' : h_nonlin_name})

```

* Dropout (need to explain a bit?)


```python

def dropout(h_keep_prob):
    def compile_fn(di, dh):
        p = tf.placeholder(tf.float32)
        def fn(di):
            return {'Out' : tf.nn.dropout(di['In'], p)}
    return fn, {p : dh['keep_prob']}, {p : 1.0}
return siso_tfm('Dropout', compile_fn, {'keep_prob' : h_keep_prob})

```

* Similarly for Batchnorm

```python

def batch_normalization():
    def compile_fn(di, dh):
        p_var = tf.placeholder(tf.bool)
        def fn(di):
            return {'Out' : tf.layers.batch_normalization(di['In'], training=p_var)}
    return fn, {p_var : 1}, {p_var : 0}
return siso_tfm('BatchNormalization', compile_fn, {})

```

* Optional dropout/batchnorm:  this pre-defined module determines whether dropout is included in a particular architecture or not. 
* H_drop_keep_prob: hyperparmeters of dropout probability to choose from.  
* H_opt_drop: dropout optional hyperparameter; if 0 is select, then dropout is exclude. Vice versa, 1 is include.
	

```python

	mo.siso_optional(lambda: dropout(h_drop_keep_prob), h_opt_drop)

```

* Permutation dropout/batchnorm: pre-defined module that determines the ordering of modules. In this case, whether batchnorm is before dropout or vice versa. 
* H_swap: D([0, 1]) # order of swapping for permutation 


```python

mo.siso_permutation([
    lambda: mo.siso_optional(lambda: dropout(h_drop_keep_prob), h_opt_drop),
    lambda: mo.siso_optional(batch_normalization, h_opt_bn),
], h_swap)]
				
```

* Search space: putting everything together (use dnn_net_simple for pedagogical purpose). You can see that this is very similar to the baseline architecture above. 


```python

def dnn_net(num_classes): 

        # declaring hyperparameter
        h_nonlin_name = D(['relu', 'relu6', 'crelu', 'elu', 'softplus']) # nonlinearity function names to choose from
        h_opt_drop = D([0, 1]) # dropout optional hyperparameter; 0 is exclude, 1 is include 
        h_drop_keep_prob = D([0.25, 0.5, 0.75]) # dropout probability to choose from 
        h_opt_bn = D([0, 1]) 
        h_num_hidden = D([64, 128, 256, 512, 1024]) # number of hidden units for affine transform module 
        h_swap = D([0, 1]) # order of swapping for permutation 
        h_num_repeats = D([1, 2]) # 1 is appearing once, 2 is appearing twice
        
        # defining search space topology 
        model = mo.siso_sequential([
                mo.siso_repeat(lambda: mo.siso_sequential([
                        affine_simplified(h_num_hidden),
                        nonlinearity(h_nonlin_name),
                        mo.siso_permutation([
                                lambda: mo.siso_optional(lambda: dropout(h_drop_keep_prob), h_opt_drop),
                                lambda: mo.siso_optional(batch_normalization, h_opt_bn),
                                ], h_swap)]
                        ]), h_num_repeats)
                affine_simplified(D([num_classes]))])
        ])
        
        return model 

```

* Can refactor into computation cell dnn_cell and end up with following 

```python

def dnn_cell(h_num_hidden, h_nonlin_name, h_swap, h_opt_drop, h_opt_bn, h_drop_keep_prob):
    return mo.siso_sequential([
        affine_simplified(h_num_hidden),
        nonlinearity(h_nonlin_name),
        mo.siso_permutation([
            lambda: mo.siso_optional(lambda: dropout(h_drop_keep_prob), h_opt_drop),
            lambda: mo.siso_optional(batch_normalization, h_opt_bn),
        ], h_swap)])

def dnn_net(num_classes):
    h_nonlin_name = D(['relu', 'relu6', 'crelu', 'elu', 'softplus'])
    h_swap = D([0, 1])
    h_opt_drop = D([0, 1])
    h_opt_bn = D([0, 1])
    return mo.siso_sequential([
        mo.siso_repeat(lambda: dnn_cell(
            D([64, 128, 256, 512, 1024]),
            h_nonlin_name, h_swap, h_opt_drop, h_opt_bn,
            D([0.25, 0.5, 0.75])), D([1, 2])),
        affine_simplified(D([num_classes]))])

```

* Our search space is now living under the dnn_net function! 
* Search space factory: having implemented our search space file, we then add the following code to the main.py 
* We get our search space in line 1 of get_search_space function. The search space simply returns inputs and outputs and abstracts away all the computational modules in between. We then return that inputs, ouputs, and an empty hyperparameter dictionary for later use in the main function. 

```python

class SSF0(mo.SearchSpaceFactory):
        def __init__(self, num_classes):
                mo.SearchSpaceFactory.__init__(self)
                self.num_classes = num_classes
                
        def _get_search_space(self):
                inputs, outputs = css_dnn.dnn_net(self.num_classes)
                return inputs, outputs, {}

```

Searcher 
--- 
* Brief overview of searcher: Searcher is searching algorithm that determines the way we sample an architecture in the search space. More details are in Searcher section. 
* In this tutorial, we are using Random search for simplicity. Other choices are Monte Carlo Tree Search, Sequential-Model Based Optimization, Efficient Neural Architecture Search, and Evolutionary.   
* Defining a searcher. 
	* Input to searcher constructor is the search space function (fn: () -> inputs, outputs, hyper_dict) defined above, in the search space factory. 

```python

search_space_factory = SSF0(num_classes)
searcher = se.RandomSearcher(search_space_factory.get_search_space)

```

• You can then sample an architecture with searcher.sample() and update the information with searcher.update(). Putting the search space and searcher together, we add the following code to main function: 

```python

search_space_factory = SSF0(num_classes)
searcher = se.RandomSearcher(search_space_factory.get_search_space)
        for _ in xrange(num_samples):
        inputs, outputs, hs, _, searcher_eval_token = searcher.sample()
        val_acc = 0 # dummy holder for evaluation metric
        searcher.update(val_acc, searcher_eval_token)

```

At this point, you can already sample an architecture! We will discuss how to evaluate an architecture next. 

Evaluator 
---
* Brief overview of evaluator: once we sample an architecture, we need to evaluate how good this architecture is. This typically involves the normal training and validating procedure. The evaluator typically returns the best validation metric (accuracy, F1, etc.) of each architecture, and we select the best architecture based on that. More in Evaluator section. 

* In this tutorial we will use a simple classifier evaluator. The full code is under darch/contrib/evaluators/tensorflow/classfication.py. 

* Highlights: 
	* _compute_accuracy function: this function specifies how to compute the accuracy on validation set. We simply use correct predictions divided by total predictions. 
	* In the eval function (which will be call in main function for each architecture) 
		* Forward function: this function propagates the computation forward, now that the input is specified. All the computational modules in the graph will be compiled. 


```python

co.forward({inputs['In'] : X_pl})

```

	* The rest are the same as when training model in tensorflow. We also used advance tricks like early stopping, patience, reduce step size, and GPU support. 
	* Eval returns a dictionary contains training information, including validation and testing accuracy. 
				
Finally we define the evaluator and call eval at each sampling. 

```python

def main():
    num_classes = 10
    num_samples = 16
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = load_mnist('data/mnist')
    train_dataset = InMemoryDataset(Xtrain, ytrain, True)
    val_dataset = InMemoryDataset(Xval, yval, False)
    test_dataset = InMemoryDataset(Xtest, ytest, False)
    evaluator = SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes,
        './temp', max_eval_time_in_minutes=1.0, log_output_to_terminal=True) # defining evaluator 
    search_space_factory = SSF0(num_classes)

    searcher = se.RandomSearcher(search_space_factory.get_search_space)
    for _ in xrange(num_samples):
        inputs, outputs, hs, _, searcher_eval_token = searcher.sample()
        val_acc = evaluator.eval(inputs, outputs, hs)['validation_accuracy'] # evaluate and return validation accuracy
        searcher.update(val_acc, searcher_eval_token)

```

(Optional-- Recommended) Logging 
---
* Logging is an important aspect of architecture search. This is because there are many information that needs to be stored for evaluation and future iterations. For logging please refer to darch/examples/tensorflow/mnist_with_logging/main.py 
