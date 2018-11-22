# DeepArchitect examples directory

[mnist](https://github.com/negrinho/darch/tree/negrinho_cleanup/examples/mnist):
Simplest architecture search example in DeepArchitect for the MNIST classification
task showcasing simple search spaces, searchers, and evaluators.

[mnist_with_logging](https://github.com/negrinho/darch/tree/negrinho_cleanup/examples/mnist_with_logging):
Similar to [mnist](https://github.com/negrinho/darch/tree/negrinho_cleanup/examples/mnist),
but also showcasing the use of the logging functionality.

[simplest_multiworker](https://github.com/negrinho/darch/tree/negrinho_cleanup/examples/simplest_multiworker):
Simplest way of making use of multiple machines to do architecture search
in DeepArchitect: first samples multiple architectures from a search space,
and then distributes them them across the evaluators.

[benchmarks](https://github.com/negrinho/darch/tree/negrinho_cleanup/examples/benchmarks):
Compares the search performance of different search algorithms on different
search spaces.