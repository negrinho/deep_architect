


###${MARKDOWN}
# # Introduction
#
# DeepArchitect is a framework for searching over computational graphs.
# The ideas and motivation to build such a framework came from the current
# deep learning workflow and its limitations.
# The current workflow places the burden of finding a good architecture directly
# on the expert.
# DeepArchitect aims to automate the task of finding a good architecture.
# The expert writes an expression that encodes a search space of architectures
# worth considering and an automatic search algorithm explores the search space
#
# DeepArchitect is built around the idea of a module.
# A module encapsulate computation that is dependent on hyperparameters and inputs.
#




# potentially change this as a tour of core.

# The most important conceptual decorations are in core.py.
# We will provide a walkthrough of those here for a new conceptual example.
# Let us assume that our goal is to search over

# writing down a simple search space to get started.

# searching over the structural aspect.


import deep_architect.core as co
import deep_architect.hyperparameters as hp




