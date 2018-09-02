Search Space Tutorial 
**********************
What is search space 
====================

Modules 
========

Designing Search Space 
=======================

.. code-block:: python
	
	def dnn(): 
	    return mo.siso_sequential([
	        affine(D([num_classes])), 
	        nonlinearity(D(['relu', 'tanh']))
	    ])

