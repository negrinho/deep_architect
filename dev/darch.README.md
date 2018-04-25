

todo: this file will go over the design decisions and the motivations to work
over some of these problems.

core.py
Core functionality to construct search spaces. Contains the definitions of ...
For high level usage of the library, only a very small level of understanding of
how the functionality is implemented is necessary.

Helps define what is a simple search space.

modules.py:
Modules contain mostly substitution modules, that are useful to construct new
search spaces through composition.
CamelCase functions signify that the functions return modules while lower case
functions in the typical format means that the functions return the inputs and
outputs of the module directly.
These functions are useful to implement complex search spaces through composition.
siso means that there is a single input and a single output.
mimo means that there are potentially multiple inputs and multiple outputs.

The standard followed to name hyperparameters is to prefix them with h_.

The substitution modules are implemented through lazy evaluation.

TODO: perhaps add a few examples of how this can be implemented here.

Substitution modules implement a different form of delayed evaluation for the
modules. One canonical

TODO: pointers to these documentation. this is going to be interesting.

Working directly with dictionaries of inputs and outputs is a substantial idea
that allows us to write expressions more concisely.

Relies heavily on delayed evaluation.

Sharing through passing the same hyperparameter.

The most important Python modules to understand are.

In core.py you will find the necessary functionality.


TODO: say something about if you find comments missing, looking for an example
of usage of the model is the best way to go.



