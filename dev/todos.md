
Evaluation per day on a single GPU per evaluation length
* 1 model per day is equiv. 1440 min per model (24 hours per model)
* 2 models per day is equiv. 720 min per model (12 hours per model)
* 4 models per day is equiv. 360 min per model (6 hours per model)
* 8 models per day is equiv. 180 min per model (3 hours per model)
* 16 models per day is equiv. 90 min per model
* 32 models per day is equiv. 45 min per model
* 64 models per day is equiv. 22.5 min per model
* 128 models per day is equiv. 11.25 min per model
* 256 models per day is equiv. ~5.5 min per model
* 512 models per day is equiv. ~2.75 min per model
* 1024 models per day is equiv. ~1.38 min per model
* 2048 models per day is equiv. ~45 secs per model
* 4096 models per day is equiv. ~22.5 secs per model
* 8192 models per day is equiv. ~11.25 secs per model

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
* Go through torchvision to get some inspiration to develop search spaces for convolutional spaces.
* Add a build script that downloads a bunch of useful information.
* Make the SimpleClassifierEvaluator more generic.
* Figure out how to keep it sane with the different modules when building the image. The image out not be built very frequently.
* Think about how to tie the image to a specific commit.
* Think more about wrapping more of the logging and search functionality in some high level functions.
* Probably the recursive read needs some more information on getting it done.
* TODO: write error messages for the loggers, e.g., asserts.
* add some error checking or options to the read_log
* maybe move some of this file system manipulation to their own folder.
* integrate better the use of list files and list folders.
* check how to better integrate with the other models.
* add more user_data and functionality to load then.
* add the ability to have a function that is applied to each file type.
* Make it easy to read multiple folders simultaneously.
* Check http://www.sphinx-doc.org/en/stable/tutorial.html for notes on how to write documentation in Sphinx.
* Add the flop count to the Tensorflow simple evaluator.
* Refactor some of the Tensorflow code to pull some functional chunks.
* Make the classifier evaluator code to be more general, making it easy to run with different types of loss functions.
* List the contrib folders last in the documentation.
* Fix the documentation to make more extensive use of cross-referencing.
* Revisit the documentation for all but core.py.
* Add CMU logo and Petuum logos.
* Potentially change all names to qualified names.
* It is possible to change the inputs and outputs to have get_val and set_val
functions, rather than accessing the fields directly.
* Add a way of showing the documentation for the private methods for some aspects
of the code base, e.g., some of the private functions of the modules.
* Define debug modes for the models in the graph.
* Be more consistent in the application of see also in the documentation.
* While the visualization functionality is not fully finished, add some simple functionality to draw plots from search folders easily.
* Perhaps remove the type redundancy in the documentation.
* Write down some hyperparameter optimization vignettes comparing hyperopt and darch.
* Make this a lot cleaner.
* Add a test for dependent hyperparameters that involves making a chain of dependent hyperparameters.
* Make a test for dependent hyperparameters with a loop (should result in problems); needs loop detection.
* Test for dependent hyperp. that makes it depend on two other hyperparameters simultaneously.
* Add a lot more tests to core.
* Go over the tests as they are very poor right now.
* Add continuous valued hyperparameters.
* It may be possible to define the operations to use based on some dictionary that works across many different frameworks. This would help define new search spaces across different frameworks.
* Refactor the substitution modules to take a dictionary of hyperparameters and a dictionary of inputs to make it more consistent with the other one.
* To make certain calls more uniform, prefer functions that work over dictionaries to prevent packing and unpacking of results.
* Implement get_search_space as a wrapper around an unbuffered search space.
* Think more about the substitution module. Perhaps it should not depend on the inputs.
* Make type referencing more consistent. Right now, in some places we use absolute names while in other places, we use relative names.
* Configure the formatting and linting error checking level to make sure that it is appropriate for the people that will come in.
* Adding readme.md files in different folders may help the understanding of the project.
* Write a few examples by adapting the examples in the keras page.
* Implement a dataloader based on PyTorch.
* Add a few issues to the github to get the mechanism for contribution started.
* Put some of the issues on github to jump start contribution.
* Setting up a discourse channel for the platform. This can be hosted locally in my home page.
* Set a few tasks for the viewer.
* The comparison with hyperopt can be done through the same code.
* Make clear what is the most common workflow that this library will support. loading the data, setting the evaluator, creating the search space, instantiating the searcher, creating the logger, running the search process, visualize the results of the process.
* Show how to take an existing model and map it to something in our framework.
* Talk about how this can be hidden behind some high-level API.
* API should teach them
* siso_repeat can be refactored with siso_sequential.
* Make documentation more consistent (this has low priority for now).
* Transfer automatically the documentation to my website upon building it.
* Make sure I can copy the code easily while maintaining consistency with respect to the different models.
* Check that I'm consistently working with the correct functionality.
* Think of a few sections for the discourse page.
* Change the search spaces to be generic across frameworks.
* Improve error messages by checking cases where the current message is cryptic and add a better message that alludes to the likely cause of the problem.
* Needs to add error messages to the assert messages.
* change get_current_evaluation_logger to get_next_evaluation_logger.
* Check the logs for corrections.
* Compile the code.
* Write down about the directly structure of the code. This is important to be explicit about the structure of the folder and what is container in each of them.
* Decide on the capitalization, i.e., DArch vs Darch vs darch. I think that I prefer the first one.
* Make consistent vs with hyperp_value_lst
* Complete the documentation.
* Create a file that emphasizes the most interesting classes.
* Make sure that the appropriate helpers are in place.
* To make life easier, it would be nice to take the comments in the file and just write down the tutorials based on them. It would make it easier and improve consistency.
* Just say something about the log manager for plotting.
* Be a bit more clear on the inputs of the model and such.
* Change the Docker containers
* Change the deepo image to add the one with Jupyter.
* Add a table where markers can be added with numbers to keep track of what are the elements of the table that we care about.
* Think about adding debugging configurations that can be mentioned by pointing to the right ones.
* Remove sentiment_nn example as it is not very informative.
* Have a script that allows to rename imports from the contrib folder to imports of the dev folder or something similar.
* Have code to show the dependencies on folders in contrib and folders in dev.
* Better headers for the container recipe.
* Write the preliminary readme file for the project.
* Log visualization can be applied quite broadly. As long as the log manager knows how to expose a dictionary of information, we can do something about it.
* Functionality to just keep user data for the best architectures according to some measure. This guarantees that memory does not blow up.
* Managing different logs should be easy. This means that we can progressively add more data to an existing dataset. This will require the creation of tools to aggregate and manipulate log folders.
* Add more tools to manipulate different log functions.
* Work on including more tasks in the framework. This is useful to make sure that we can work with different models.
* Add functionality to generate a representation of the specification process of the search space.
* Visualization with better support to connect to assets that result from search.
* Do the wrapping of the search space at the level of the searcher. There is some amount of boilerplate that happens at that level. Right now, solved via the search space factory.
* Check if a multi input, multi output empty is necessary.
* Move the filesystem utils somewhere else.
* Use the term triage for the purpose of the contrib directory.
* Change the links to the deep_architect repo once we have all the models.
* Improve the way the padding of existing search spaces is done.
* In some cases it would be convenient to have functions that take the specificed hyperparameters and create the desired artifacts, e.g., search spaces for data augmentation schemes and search spaces for learning rate schedules.
* What are the current limitations in terms of models, i.e., what can be done
* Evolution searcher with general mutation scheme that works for all search spaces.
* A good way of reducing the effort of documenting the code is when finding something confusing, it is useful to document the code extensively then.
* Are the functions in the empty module necessary or not.
* Decide on better names for the variables that should be more consistent. This is problematic.
* The propagation for dependent hyperparameters is going to be suboptimal.
* Write more extensive tests for the new functionality for getting the hyperparameters.

Before release:
* Add more links to the project.
* Add some of the configuration defaults for VS Code to gua
* Create some additional Slack channel for discussion.rantee that the development environment is the closest possible to our development environment.
* Plan any breaking changes that you may have.
* Add to contributing information about what needs to be kept track of.
* Check code base vs codebase.
* Add pointers more extensively to the code that is there.
* Check that MCTS and SMBO work as well as we expect.
* Think about the sharing case for the model, i.e., what would happen in the case
* Refactor some of the names in the code to make things more readable.

Visualization:
* Make sure that an exception is never thrown.
* Handle invalid keys properly, by just keeping the previous state or some decent defaults.
* Remove row functionality.
* Copy row functionality.
* Save visualization functionality for the plots. The maximum number of plots can also be done in the config.
*

Core:
* Graph propagation has to be revisited. For example, for long chains of dependent hyperparameters, it is better to do things directly rather than through lazy evaluation.

* Common metrics to maintain in logging:
    * Training loss
    * Validation loss
    * Train performance metric
    * Validation performance metric
    * Learning rate (seq)
    * Total training time (seq)
    * Example training images predicted by the model.


Guidelines for documentation:
* Do not use Sphinx directives in the first line of the comment.
* Keep comments short.
* Do not comment the obvious.
* At a minimal add a docstring for the class.
* Complex functionality that is widely used in the rest of the library should be widely commented.
* Replicate the comment format that is used in the rest of the classes.
* Some directives to use are :func: (for functions), .. seealso::
(to point to some other parts that are important.), :meth: (for a method).
Check this reference: http://www.sphinx-doc.org/en/stable/markup/inline.html#inline-markup
This one is more relevant: http://www.sphinx-doc.org/en/stable/domains.html#python-roles
* Be more careful in the way that we define hyperparameters and

Darshan:
* Figure out how to pass dictionary of results to update searcher
    * R: This requires telling the searcher which keys to look at. This is
    makes sense for the case where the we are doing multi-objective optimization,
    but it might be less interesting for the single metric case.
* Darch controls Randomness
* Write test for core disconnect