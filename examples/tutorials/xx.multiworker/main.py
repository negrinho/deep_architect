
there is often the necessity of distributing the architecture search load
over multiple machines.
Typical use-cases are using a local machine with multiple GPUs, where we
intend to parallelize each evaluation on each of the GPUs.
Another possibility is to have multiple workers running simultaneous.


# talk about the design of the Python API and how is use to enable this.


The Searcher abstract class was designed with this use-case in mind, namely
that a searcher can sample an architecture and that its results can occur out of
order.
In this case, an architecture may be sample but its result may return to the
user only after its results have been generated.
A typical example where this would happen is in a case where multiple workers
are doing different evaluations simultaneously and may end out of order.

# TODO: perhaps talk about resuming different architecture search configurations.
# this is going to be important.

# The simplest possible