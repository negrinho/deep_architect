
# Guidelines for contributing

We strongly encourage and welcome contributions to DeepArchitect.

These contributions can result from your own research or
be implementations of existing algorithms.
If you have developed a searcher, search space, evaluator,
or any other component or functionality that would be useful to include
in DeepArchitect, please make a pull request that follows the guidelines
described in this document.
After reading this document, make sure that you understand:
* what are the different types of contributions that we identify;
* what is the folder structure for contributions;
* what is required in terms of tests and documentation for different types of contributions;
* what are the different levels of conformity that we require for different types of contributions.

If you have a feature that you would like to add to DeepArchitect but you
aren't sure if it should be included, open a
[GitHub issue](https://github.com/negrinho/darch/issues)
to discuss its scope and suitability.
This guarantees that your efforts are well-aligned with the project direction.
The best way to start a discussion is with a code snippet or pseudo-code
that illustrates the main use case of the feature that you want to implement.
You can also check the evergrowing list of work items
[here](https://github.com/negrinho/darch/blob/master/todos.md).

# Types of contributions

Most contributions will go into the `contrib` folder.
The `contrib` folder is used for functionality that is likely useful, but for
which we cannot necessarily guarantee that it will be maintained over time.
While code lies in the `contrib` folder, it is the responsibility of the code
owners to maintain it, i.e., that it is not broken.
If code in the `contrib` folder breaks and the code owner does not fix it
by issuing a pull request in a timely manner, we reserve the right to move
the code to the `dev` folder.
The `dev` folder serves to store code that contains a sketch of some interesting
functionality, but due to some reason, it is not fully functional yet or it
has not been refactored well enough to be integrated as part of `contrib`.
Unmaintained code will be moved to `dev` upon breakage.

Code that is part of the contrib folder may eventually be refactored into code
that is part of the darch folder.
Similarly, code in the dev folder may be refactored in code that goes in the
contrib folder.
In case this happens, it is the responsibility of the developers of DeepArchitect
to maintain it.
To create a new contrib folder, it is best it is best to first discuss scope to
make sure that contributions are done at a good level of detail.
For the dev folder, we do not impose these restrictions.
The `dev` folder should be used sparsely though.
We will only accept incomplete contributions to the dev folder if it is determined
that they showcase an important functionality and there is sufficient reason
to include even without it being completed in functionality or scope.
For cases where the functionality is indeed complete in terms of functionality
or scope, we recommend the contributor to refactor its contribution in
some thing that can be included in the contrib folder and reused by other
users of DeepArchitect.

Cross-pollination between `contrib` and `dev` folders is expected and encouraged.
One example of when this would make sense would be if a few contrib folders already
had some useful functionality, but a contributor wanted to extend it and
encapsulate it in a more coherent contrib subfolder.
This scheme allows DeepArchitect to evolve without committing to major
refactoring decisions upfront.

If the foreseen contribution is better seen as an extension or fix to an
existing contrib folder, please check with the active contributors for how to




<!-- only for the code that is derived from the model. -->

# Folder structure for contributions
<!-- say something about how we used a contrib folder scheme to work on our
contributions. -->

For minimizing coupling between contributions of different people, we adopt a
design similar to the one used in
[Tensorflow](https://github.com/tensorflow/tensorflow).
Namely, we have a `contrib` folder where each new sufficiently
different contribution gets assigned a subfolder.
The name of the subfolder should be chosen to reflect the functionality that
lies within.
All the library code contributed by the developer will be placed in this folder.
Main files that are meant to be run should not be placed in `darch/contrib`,
but rather in `examples/contrib`.
The same name should be used for both the subfolder in `darch/contrib` and
in `examples/contrib`.
The subfolder in `examples/contrib` is meant for runnable code related to
or making extensive use of the library code in the `darch/contrib` subfolder.
We recommend checking existing examples in the
[repo](https://github.com/negrinho/darch) for determining how to
structure and document a new example appropriately.
Options to run the code should be placed in a JSON configuration file in the
same subfolder as the other code.
This JSON configuration guarantees that the options that determine the behavior
of running the code can be kept separated from the code.
This allows us to include multiple configurations in a single JSON file, e.g.,
see [here](TODO).
Each key in the JSON configuration file corresponds to a different configuration.
We suggest the inclusion of a `debug` key.
The configuration in this key will be used to run a quick experiment to
validate the functionality of both the code under `contrib/examples` and
`darch/contrib`.


(i.e.,
the configuration should be such that running can be done in five minutes at most)




Code in contrib will be subject to more extensive code reviews.



If a contribution

The code owner file.
Minimizing redundancy for the file.


Naming for these files.

Examples of contributions and rationale about where they fit.


# Add


Summary



It is the responsibility of contributors to guarantee that their contributions
are up-to-date.
The


# Recommended code editor

The recommended code editor is [Visual Studio Code](https://code.visualstudio.com/)
with recommended plugins `ms-python.python`, `donjayamanne.githistory`,
`eamodio.gitlens`, `donjayamanne.jupyter`, `yzhang.markdown-all-in-one`,
`ban.spellright`. These can be installed through the extension tab or in the
command line (after Visual Studio Code has been installed) with
`code --install-extension $EXTENSION_NAME` where `EXTENSION_NAME` should be
replaced by the name of each of the extensions.




Code style:
In terms of naming convention and code style, we ask you to follow the naming
naming conventions and general style used throughout the rest of the code base.

Contributions will typically fit into some cases.

examples.
* Examples in new frameworks that are currently not covered in the toolbox.

search spaces.
* Complex

searchers.
* New searchers based on existing literature or


* getting a more complete library of search spaces.
Search spaces are compositional in our framework. The creation of new search
spaces.

Platform for research in architecture search.

* adding features that


TODO: come up with a way of managing the contrib folder.


to guarantee reproducibility when running the code, we recommend that the
correct git commit is used.

(what happens if certain desired functionality is not available).

If this tool has been useful to you, show some appreciation by contributing to
it or getting your colleagues to use it too.
Everyone benefits from open-source.


TODO: for each of the different modes of contribution, make

TODO: perhaps add a slack channel for contributing. First, show commitment by
implementing some of this functionality.

how to make things compatible without having too many dependencies.

I would like to add a new architecture search task to the battery of tasks covered.

New benchmark:



main.py
evaluator.py
search_space.py
searcher.py
config.json

if only a subset of these are used, then it is fine to just propose a benchmark
where each of the of the models.

easy to plug a new one.

what is the difference between the benchmarks an something else, for example,
if I want to substitute some components of the benchmark by something that I can
control. What is the problem here.

On writing tests.

Each contribution has its own tests associated. separate the contributions by
folder.

darch ; should I create a stackoverflow for this, or actually create a forum
for this. it is going to be very low volume.

The recommended editor for contributing is VS Code. We make available

structuring of the project. this is going to be important. the model is
going to be important.

# contributing guidelines:

Contributing a searcher

Contributing a search space

Contributing an evaluator

Other contributions


The main aspects to strive for are consistency.
The more standard the folder the code is placed in, the stricter the consistency standard.
This means


* folder.
main.py
evaluator.py
search_space.py
searcher.py

* contrib_examples


the standards for

config.json for running code.
accompanying tests.

making sure that all this is automated.

Talk about the importance of a config.json.

NOTE: can we have comments in the config.json.


the contrib folder is code that runs.

should new results be posted as examples.

We will make a consistent effort to make sure

Three levels: darch, contrib, dev

dev is used for functionality that is important



Aspects that are important. It is important to compartamentalize the results of
the model.
This is important because it will allow us to

A few guidelines for naming as for talking about the

The contrib folder was designed such that each folder

Readme file. config file.


examples/contrib

same name.

addressing space.

tests
 darch
 examples

this is the correct think


guideline

what are the tests that are going to run.

contrib is the library component of your contribution.
don't commit data files. your library should not require any data files.
if you main requires data files, then add a README.md


add the research toolbox here.
random_searcher.py

framework code.

evaluators.py
searchers.py
search_spaces.py

import darch.contrib.useful.

what happens if you depend on contrib functionality that is not being maintained.

those will serve as documentation and other thing.

**Important:** By submitting a pull request to this project, you agreeing to
license your contribution under the MIT license.



We follow the napoleon documentation and sphinx.
for tests, we use something else... this is going to be painful.

which ones should have list there.

Keep your commits small and localized and

Furthering progress on architecture search by

How to concentrate stuff in your folder.

Add some information about what remains to be done.
This is nice and functions as a wish list going into the future.

Documentation requirements.