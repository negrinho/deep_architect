import deep_architect.core as co
from six import iteritems


class SurrogateModel:
    """Abstract class for a surrogate model.
    """

    def eval(self, feats):
        """ Returns a prediction of performance (or other relevant metrics),
        given a feature representation of the architecture.
        """
        raise NotImplementedError

    def update(self, val, feats):
        """Updates the state of the surrogate function given the feature
        representation for the architecture and the corresponding ground truth
        performance metric.

        .. note::
            The data for the surrogate model may be kept internally. The update
            of the state of the surrogate function can be done periodically, rather
            than with each call to update. This can be done based on values for
            the configuration of the surrogate model instance.
        """
        raise NotImplementedError


# extract some simple features from the network. useful for smbo surrogate models.
def extract_features(inputs, outputs):
    """Extract a feature representation of a model represented through inputs and
    outputs.

    This function has been mostly used for performance prediction on fully
    specified models, i.e., after all the hyperparameters in the search space
    have specified. After this, there is a single model for which we can compute
    an appropriate feature representation containing information about the
    connections that the model makes and the values of the hyperparameters.

    Args:
        inputs (dict[str, deep_architect.core.Input]): Dictionary mapping names
            to inputs of the architecture.
        outputs (dict[str, deep_architect.core.Output]): Dictionary mapping names to outputs
            of the architecture.

    Returns:
        dict[str, list[str]]:
            Representation of the architecture as a dictionary where each
            key is associated to a list with different types of features.
    """
    module_memo = co.OrderedSet()

    module_feats = []
    connection_feats = []
    module_hyperp_feats = []

    # getting all the modules
    module_memo = []

    def fn(m):
        module_memo.append(m)

    co.traverse_backward(outputs.values(), fn)

    for m in module_memo:
        # module features
        m_feats = m.get_name()
        module_feats.append(m_feats)

        for ox_localname, ox in iteritems(m.outputs):
            if ox.is_connected():
                ix_lst = ox.get_connected_inputs()
                for ix in ix_lst:
                    # connection features
                    c_feats = "%s |-> %s" % (ox.get_name(), ix.get_name())
                    connection_feats.append(c_feats)

        # module hyperparameters
        for h_localname, h in iteritems(m.hyperps):
            mh_feats = "%s/%s : %s = %s" % (m.get_name(), h_localname,
                                            h.get_name(), h.get_value())
            module_hyperp_feats.append(mh_feats)

    return {
        'module_feats': module_feats,
        'connection_feats': connection_feats,
        'module_hyperp_feats': module_hyperp_feats,
    }
