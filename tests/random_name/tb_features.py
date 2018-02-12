### simple featurization
class ProductFeaturizer:
    def __init__(self, fs):
        self.featurizers = fs
    
    def features(self, x):
        pass
    
    def dim(self):
        return sum( [f.dim() for f in self.featurizers] )

class HashFeaturizer:
    def __init__(self):
        pass

class OneHotFeaturizer:
    def __init__(self):
        pass

class BinarizationFeaturizer:
    def __init__(self):
        pass

# NOTE: certain things should be more compositional.

# TODO: perhaps add the is sparse information.

# what should be the objects.

# there is information about the unknown tokens and stuff like that.


# there are things that can be done through indexing.

# can featurize a set of objects, perhaps. 
# do it directly to each of the xs passed as 

# some auxiliary functions to create some of these.



# the construction of the dictionary can be done incrementally.

# TODO: a class that takes care of managing the domain space of the different 
# architectures.

# this should handle integer featurization, string featurization, float
# featurization
# 

# subset featurizers, and simple ways of featurizing models.

# features from side information.
# this is kind of a bipartite graph.
# each field may have different featurizers
# one field may have multiple featurizers
# one featurizers may be used in multiple fields
# one featurizer may be used across fields.
# featurizers may have a set of fields, and should be able to handle these 
# easily.

# features from history.
# features from different types of data. it should be easy to integrate.

# easy to featurize sets of elements, 

# handling gazeteers and stuff like that.
# it is essentially a dictionary. 
# there is stuff that I can do through

# I can also register functions that take something and compute something.
# and the feature is that. work around simple feature types.

# TODO: this is going to be quite interesting.

# come up with some reasonable interface for featurizers.
# TODO: have a few operations defined on featurizers.

# NOTE: that there may exist different featurizers.
# NOTE: I can have some form of function that guesses the types of elements.

# NOTE: for each csv field, I can register multiple features.

# NOTE that this is mainly for a row. what about for things with state.



# TODO: syncing folders
