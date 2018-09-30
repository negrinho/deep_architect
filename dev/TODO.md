
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


## Work items
* Group the work items into logical groups.

* Run the benchmark on the server.
* Add a few PyTorch examples.
* Write a better loader for CIFAR-10.
* Develop the PyTorch functionality.
* Write down documentation.
* Merge some of the functionality as examples, or keep them in their own branches for now.
* Think of a good way of adding regularization to the model. Either the current helper can be extended or we just want to have broad regularization over all the image.
* Visualizer for ONNX models.
* Standards for ONNX models.
* Standards for data representation in disk.
* Hosting the models online.
* Serving models online in the browser.
* Perhaps make another one with resource logging specific functionality.
* Add some simple command line search visualization.
* The representation of the level interface computation can be done through JSON files. This can be done to describe both the data and the computation that we need to apply to it. First focus on serial computation, then think about how to construct arbitrary DAGs. This can be both for machine learning computation, as for
* Add a script for the shortest route to run models in our application. This implies installing VirtualBox, Vagrant, getting the image that has singularity installed, and getting our containers to run the software. On the server, it is only a matter of making that it has singularity installed.
* Make the documentation more minimal. Reduce the amount of obvious comments.
* Change the documentation to the same format at Pytorch and Tensorflow.
* Finish adding the containers and make sure that people can use them easily.
* Add some restrictions in terms of the naming to make sure that we can do splits on characters easily.
* Change the API of the surrogate models to work with lists that allow us to be more efficient when evaluating the surrogate function for multiple examples.
* Go through torchvision to get some inspiration to develop search spaces for convolutional spaces.
* Add a build script that downloads a bunch of useful information.
* add some error checking or options to the read_log
* maybe move some of this file system manipulation to their own folder.
* integrate better the use of list files and list folders.
* check how to better integrate with the other models.
* add more user_data and functionality to load then.
* add the ability to have a function that is applied to each file type.
* Check http://www.sphinx-doc.org/en/stable/tutorial.html for notes on how to write documentation in Sphinx.
* Revisit the documentation for all but core.py.
* Add CMU logo and Petuum logos.
* Potentially change all names to qualified names.
* Add a way of showing the documentation for the private methods for some aspects
of the code base, e.g., some of the private functions of the modules.
* Define debug modes for the models in the graph.
* Be more consistent in the application of see also in the documentation.
* While the visualization functionality is not fully finished, add some simple functionality to draw plots from search folders easily.
* Perhaps remove the type redundancy in the documentation.
* Write down some hyperparameter optimization vignettes comparing hyperopt and deep_architect.
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
* Decide on the capitalization, i.e., DArch vs Darch vs deep_architect. I think that I prefer the first one.
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
* Use the term triage for the purpose of the contrib directory.
* Change the links to the deep_architect repo once we have all the models.
* Improve the way the padding of existing search spaces is done.
* In some cases it would be convenient to have functions that take the specificed hyperparameters and create the desired artifacts, e.g., search spaces for data augmentation schemes and search spaces for learning rate schedules.
* Evolution searcher with general mutation scheme that works for all search spaces.
* A good way of reducing the effort of documenting the code is when finding something confusing, it is useful to document the code extensively then.
* Are the functions in the empty module necessary or not.
* Decide on better names for the variables that should be more consistent. This is problematic.
* The propagation for dependent hyperparameters is going to be suboptimal.
* Write more extensive tests for the new functionality for getting the hyperparameters.
* Evaluation logging may change a little bit to allow for different types of logging information.
* Some auxiliary scripts to keep things running on the server.
* Make a few of the multiworker cases work smoothly.
* Cover the first part of the model running on a single machine and then show how can you get multiple machines working on the same problem. There are multiple ways of accomplishing this. The goal is to show case them.
* I think that the SearchLogger may be too inflexive regarding the use in the multiworker case.
* Change the eval data folderpath. I think that it is going to be important.


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