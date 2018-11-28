
comments on the paper
comments on the encoding


### META:
# This tutorial is divided into sections, with one section per paper.
# In each section, the ideas of the paper regarding the search space are discussed
# and we discuss its implementation in DeepArchitect using the language to write
# search spaces.
# Finally, we discuss a summary about general guidelines to express search
# spaces in DeepArchitect.

# In this tutorial, we will show how to use DeepArchitect to implement
# search spaces from the literature.
# We will take the textual description of a paper in the architecture search
# literature and we will show how can we write it in DeepArchitect.
# This exercise serves to show to the reader the process by which one would go
# about writing existing search spaces from the literature in DeepArchitect.
# We will follow the description of the search space as it is done in the paper.
# We make no effort to implement idiosyncracies in the actual implementation.

# It should also give a sense to the reader of how the search space constructs
# can be used to express non-trivial search spaces.

# We believe that most of the search spaces that we have seen in the framework
# can be expressed straightforwardly in DeepArchitect.


# ### Zoph and Le 2017

# Let us take one of the first architecture search papers: Zoph and Le, 2017.
# In section 4.1, they describe their search space:

### TODO: add some copy from the paper.
# In our framework, if one layer has many input layers then all input layers are
# concatenated in the depth dimension. Skip connections can cause “compilation
# failures” where one layer is not compatible with another layer, or one layer
# may not have any input or output. To circumvent these issues, we employ three
# simple techniques. First, if a layer is not connected to any input layer then
# the image is used as the input layer. Second, at the final layer we take all
# layer outputs that have not been connected and concatenate them before sending
# this final hiddenstate to the classifier. Lastly, if input layers to be
# concatenated have different sizes, we pad the small layers with zeros so that the
# concatenated layers have the same sizes.


# 4.1 LEARNING CONVOLUTIONAL ARCHITECTURES FOR CIFAR-10

# Search space: Our search space consists of convolutional architectures, with
# rectified linear units as non-linearities (Nair & Hinton, 2010),
# batch normalization (Ioffe & Szegedy, 2015) and skip connections between layers
# (Section 3.3). For every convolutional layer, the controller RNN has to select
# a filter height in [1, 3, 5, 7], a filter width in [1, 3, 5, 7], and a number of
# filters in [24, 36, 48,64]. For strides, we perform two sets of experiments,
# one where we fix the strides to be 1, and one where we allow the controller to
# predict the strides in [1, 2, 3].

# NOTE: this is an important part of teh model. I think that it is most important
# to consider it.

# Dataset: In these experiments we use the CIFAR-10 dataset with data preprocessing and augmentation
# procedures that are in line with other previous results. We first preprocess the data by
# whitening all the images. Additionally, we upsample each image then choose a random 32x32 crop
# of this upsampled image. Finally, we use random horizontal flips on this 32x32 cropped image.



### Negrinho and Gordon, 2017

# TODO: short description of what DeepArchitect is about.
# This was the original paper for DeepArchitect having

#### TODO: do the example for DeepArchitect




# Let us take one of the examples from original DeepArchitect paper
# https://arxiv.org/abs/1704.08792.
# The main ideas from writing a DSL to express search spaces came from this paper.
# The ideas were considerably extended for this current implementation
# (e.g., multi-input multi-output modules, hyperparameter sharing, general
# framework support, ...).
# The original DeepArchitect paper presented the following search space

# def Module_fn(filter_ns, filter_ls, keep_ps, repeat_ns):
#     b = RepeatTied(
#     Concat([
#         Conv2D(filter_ns, filter_ls, [1], ["SAME"]),
#         MaybeSwap_fn( ReLU(), BatchNormalization() ),
#         Optional_fn( Dropout(keep_ps) )
#     ]), repeat_ns)
#     return b

# filter_nums = range(48, 129, 16)
# repeat_nums = [2 ** i for i in xrange(6)]
# mult_fn = lambda ls, alpha: list(alpha * np.array(ls))
# M = Concat([MH,
#         Conv2D(filter_nums, [3, 5, 7], [2], ["SAME"]),
#         Module_fn(filter_nums, [3, 5], [0.5, 0.9], repeat_nums),
#         Conv2D(filter_nums, [3, 5, 7], [2], ["SAME"]),
#         Module_fn(mult_fn(filter_nums, 2), [3, 5], [0.5, 0.9], repeat_nums),
#         Affine([num_classes], aff_initers) ])


# ### NOTE: I could develop this one.


# ### NOTE: now talk about a different search space.

# Neural Architecture Search with Reinforcement Learning (Zoph and Le. 2016)
# https://arxiv.org/abs/1611.01578


####

The description of the search space employed is spread out across the paper.
The main idea of the paper is to write architectures as hierarchical compositions
of motifs.

Architectures in this case are just motifs at a higher level of composition.
In the description of the paper, a motif is generated by connecting lower level
using an acyclical directed graph.
All motifs have a single input and a single output.
If multiple edges come into a node, the inputs are merged into a single
input and then passed as input to the motif.
Each of the lower motifs is placed in one of the nodes of the graph and the

Given some number of higher level motifs, these can be constructed

There are the high-level ideas for the paper.


NOTE: the motif is generated by putting the elements in topological order and
picking which ones should be there.



To paraphrase and summarize what they describe in the paper, they start with a
set of six operations (which they call primitives).


This is used to search for a cell that is then used in an architecture for
evaluation. Be


# We consider the following six primitives at the bottom level of the hierarchy (` = 1, M` = 6):
# • 1 × 1 convolution of C channels
# • 3 × 3 depthwise convolution
# • 3 × 3 separable convolution of C channels
# • 3 × 3 max-pooling
# • 3 × 3 average-pooling
# • identity

### TODO: what is going to be the plan for the other parts of the model.
# I think that this is going to be useful.

# NOTE: multiplying the initial number of filters.
# how to extract a search space from it. that seems kind of tricky?


For the hierarchical representation, we use three levels (L = 3),
with M1 = 6, M2 = 6, M3 = 1. Each of the level-2 motifs is a graph with
# |G(2)| = 4 nodes, and the level-3 motif is a graph with |G(3)| = 5 nodes.
# Each level-2 motif is followed by a 1×1 convolution
# with the same number of channels as on the motif input to reduce the number
# of parameters. For the flat representation, we used a graph with 11 nodes to
# achieve a comparable number of edges.


# NOTE: the sampling of the motifs is going to be important.
# I think that this is important.


The two main ideas are the

# Number of lower level motifs. (number of acyclical graphs between these models)


# L = 3;



# This is the hierarchical composition.


It is arguably more complicated to understand the textual description of the
search space in the paper than to express it in DeepArchitect.
This is a strong indicator of value of DeepArchitect for greatly improving the
reproducibility and reusability of architecture search research.

# NOTE: it would

# NOTE: most of these models where used.

# TODO: show some information in the model.


# TODO: show three architectures sampled from the search space.
# I think that this is going to be interesting.


#### Genetic CNN,

As in other cases that we have looked at, the paper proposes an ad-hoc encoding
for the search space that can be represented in the language for representing
search spaces of DeepArchitect.
The main aspect that is searched over in this search space is the connectivity
pattern.

We follow the description of the search space that can be found (here)[https://arxiv.org/pdf/1703.01513.pdf].
The description of the search space can be found on Section 3.1 and Section 3.1.1.
The first part contains an explanation of the encoding, while the second part
contains a discussion of the handling of some special cases.

Briefly, networks from the search space described are composed S stages.
Each stage takes the input from the previous (the input to the first stage is
the input to the network), applies a convolution, then the stage computation,
which is essentially a DAG of convolutions ending in a single convolutional node,
which is followed by a last convolution. There is a pooling layer between stages.
For each stage of the search space, we have to choose the number of nodes in the
DAG, e.g., in the search space represented in Figure~1, the first stage has
four nodes and the second stage has five nodes.
Each time we do a spatial reduction with a pooling layer, the number of filters is
increased, e.g., if we use a pooling layer with stride 2, the number of filters is
multiplied by two.
Neither the spatial dimension or the number of filters changes within a stage.
Each convolution is followed by ReLU nonlinearities and batch normalization.

The connection pattern of the DAG nodes is encoded by a binary string with
K_s(K _s - 1) / 2, where K_s is the number of nodes in the DAG in stage s.
The bit string is composed by K _s - 1 sections, where section i \in [K _s - 1],
encodes the connections of node i to the earlier nodes.

Section 3.1.1 describes a few edge cases. If a node does not get its input
from any of the earlier nodes in the stage, it gets its input from the initial
node of the stage. If a node in a stage is not used, then it is excluded from
the network.
We see that most of the complexity of the search space lies in the stage part.
For simplicity, we will focus on how to represent this part of the search
space and use the

# NOTE: multiple inputs to the same node are summed.


This description is better understood by reading directly the sections mentioned
along with Figure~1,
or looking at the encoding of the search space in DeepArchitect.




# define DeepArchitect




# TODO: I think that a good idea is to check that I can use this search space
# encoding in a separate file and see that it runs and that I can sample.

# TODO: copy 3.1; binary representation section.

# TODO: copy the 3.1.1 technical details section from the paper.

From reading this section, our understanding is that all stages have the
same number of nodes

# NOTE: what does it that the connections between the ordinary nodes and the
# the default nodes are not encoded.

Some of the advantages of the encoding of this search space in DeepArchitect
when compared to the ad-hoc one are that it is easy to change the search space
within using DeepArchitect, e.g., it is trivial to extend the search space to
also search over depth of the network or to search over the number of computational
nodes on each stage (tying them or not).
Arguably, doing all these extensions in an ad-hoc encoding is not straightforward
as we have to think about custom ways of encoding these options.
Within DeepArchitect, these transformations can be expressed in a straightforward
way with basic and substitution modules.

# FOR example, we see that trivial extensions of the search space by working with
# the model.

# TODO: perhaps show that this is indeed the case.


# Summary

This tutorial has two main purposes.
First, we aim to show that many (arguably, most) of the search spaces in the
literature can be represented straightforwardly using constructs from our
language, where understanding the definition of the search space is often more
difficult than actually writing down the search space in DeepArchitect.
Second, we argue that writing ad-hoc representations is a terribly innefficient
way of doing architecture search research.
Some of the main problems with ad-hoc encodings for search spaces is that they
are hard to reuse.
We recommend researchers to use DeepArchitect as a starting point for
research on architecture search to make the search spaces and the search
algorithms readily available for reuse and extension.

We made a serious effort to mimic represent the search spaces in DeepArchitect
in a way that is analog to the way that was used in the paper.
This means that we may need we may need to opt for a representation that is
more complex to


# TODO: distinguish place where we are talking and places where someone else is
# talking.

# TODO: another aspect that I want to show off is how to take the existing
# parts of the model, and put them together.
# I think that this is the right way of go




# and many more. If you would like me to write a search space that is not
# covered in this model, just let me know and I will consider adding it.


#

# In this tutorial we showed that it is possible to use the language constructs
# that we defined in DeepArchitect to implement the search spaces over architectures.
# In many cases, the textual description of the search space in the paper is
# more complex than the description in DeepArchitect, show casing the capabilities
# of our framework.


# The papers were chosen part based on chronology and part based on the number
# of citations that they


# TODO: searching for activation functions.

# We believe that DeepArchitect will make research in architecture search
# dramatically easier, allowing researchers to more easily build on previous
# work.

# We are confident that the primitives that were introduced are sufficient to
# represent a very large number of search spaces.

# TODO: check with max what does he have for the concatenation of search
# spaces.

# what do they talk about depth wise and separable convolution.

# in merging stuff, I think that it is a good idea to have them run.
# TODO: what is a depth

# they end up having three levels.
# NOTE: it should be easier to maintain if they are in multiple files.
# it should generate a high level tutorial file, but I don't know exactly
# what should be the structure for it.


# let us get started from scratch.
# we will use

# NOTE: this is just the search space for the cell, if we wish
# to write something

# try it for a stage, and try it for multiple stages.


# NOTE: some general comments about the motifs.