

## Search spaces
* Spatial pooling that

## Writing tests
* ...


Add more guidelines on what to do and what needs to be done.

## Writing down search spaces.

Aspects to be careful about:
* Make sure that the dimensions of the tensors involved in each part of the model are relevant.
    * This may involve mapping the input to some dimensions that are more manageable by the model.

On developing search spaces

It should not be difficult to write down some search spaces with some reasonable dimensions.
Reasonable dimensions can be computed based on the model on the dimensions of the model.

## Guidelines on writing search spaces
* For more complex search spaces, include a brief docstring about what that search space accomplishes.
* On writing search spaces, it it possible to pass values for all the hyperparameters, to have defaults for all of them, or somewhere in between. The creation of these search spaces so far has focused on the case where we pass mostly hyperparameters that are then used (this is nice to accomplish different hyperparameter sharing patterns) to instantiate the search space. So far, in some cases we use just arguments.

## Work items
* Group the work items into logical groups.

* Talk about the naming conventions employed for hyperparameters and normal values.
* Fix the broken abstraction for assessing the values for the inputs. If done just in a single place it is fine, but it is done every time that we call forward.
* Change the names to be more meaningful.
* Run the benchmark on the server.
* Add a few PyTorch examples.
* Write a better loader for CIFAR-10.
* Fix GraphViz to draw the values of the hyperparameters right next to the model.
* Develop the PyTorch functionality.
* Write down documentation.
* Merge some of the functionality as examples, or keep them in their own branches for now.
* Develop a good way of keeping track of important branches or of experiments that are relevant for the model.
* Think of a good way of adding regularization to the model. Either the current helper can be extended or we just want to have broad regularization over all the image.
* Fix the problem of it not being JSON serializable.
* Visualizer for ONNX models.
* Standards for ONNX models.
* Standards for data representation in disk.
* Hosting the models online.
* Serving models online in the browser.
* Separate items in research vs tooling.
* For handling the models, it is important to have a way of formatting the input such that it is in the correct dimension, e.g., for images or sentences.
* Make the search in ONNX and then export to the various languages. Would this be something interesting to consider?
* Refactor the draw graph functionality.
* Workflows based on the pretrained models. Make sure that I can interact with them easily.
* Make it easy to log new metrics for a new evaluator. Right now everything is done locally with the model.
* Perhaps make another one with resource logging specific functionality.
* Add some simple command line search visualization.
* Keep some of the information that we are not thinking about optimizing over in a different place, i.e., don't return it in results. Perhaps just save it in
* Add a way of adding scopes to have a nicer representation for the model.]
* The representation of the level interface computation can be done through JSON files. This can be done to describe both the data and the computation that we need to apply to it. First focus on serial computation, then think about how to construct arbitrary DAGs. This can be both for machine learning computation, as for
* Add a script for the shortest route to run models in our application. This implies installing VirtualBox, Vagrant, getting the image that has singularity installed, and getting our containers to run the software. On the server, it is only a matter of making that it has singularity installed.
* Make the documentation more minimal. Reduce the amount of obvious comments.
* Change the documentation to the same format at Pytorch and Tensorflow.
* Finish adding the containers and make sure that people can use them easily.
* Add some restrictions in terms of the naming to make sure that we can do splits on characters easily.
* Change the API of the surrogate models to work with lists that allow us to be more efficient when evaluating the surrogate function for multiple examples.
* Add an automatic formatter for the project.
* Add guidelines on how to setup Visual Studio Code on how to contribute for the project (e.g., formatter, linter, ...).
* Profile the code to find bottlenecks (e.g., in the surrogate function computation).
* Check why the surrogate models are not overfitting to the data.
* Perhaps a good way of handling varying dimensions in images is to map all images to the same dimensions.
* Go through torchvision to get some inspiration to develop search spaces.

* Common metrics to maintain in logging:
    * Training loss
    * Validation loss
    * Train performance metric
    * Validation performance metric
    * Learning rate (seq)
    * Total training time (seq)
    * Example training images predicted by the model.