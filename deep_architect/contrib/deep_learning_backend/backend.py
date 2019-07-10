TENSORFLOW = 0
TENSORFLOW_EAGER = 1
TENSORFLOW_KERAS = 2
PYTORCH = 3
KERAS = 4

_backend = None
_func_dict = None


# only set backend once
def set_backend(backend):
    global _backend, _func_dict
    if _backend is not None:
        raise RuntimeError('Backend is already specified')
    if type(backend) is not int or backend < 0 or backend > 3:
        raise ValueError('value of backend not valid')

    _backend = backend
    if backend is TENSORFLOW:
        from tf_ops import func_dict
    elif backend is TENSORFLOW_EAGER:
        import tensorflow as tf
        tf_version = tf.__version__.split('.')
        if int(tf_version[0]) < 1:
            raise RuntimeError('Tensorflow version too low')
        if int(tf_version[0]) == 1 and int(tf_version[1]) < 7:
            raise RuntimeError('Tensorflow version too low')
        tf.enable_eager_execution()
        from tfe_ops import func_dict
    elif backend is TENSORFLOW_KERAS:
        import tensorflow as tf
        from tf_keras_ops import func_dict
    elif backend is PYTORCH:
        from pytorch_ops import func_dict
    elif backend is KERAS:
        from keras_ops import func_dict
    _func_dict = func_dict
    if _func_dict is None:
        raise RuntimeError('Backend %s is not supported' % backend)


def get_backend():
    if _backend is None:
        raise RuntimeError('Backend is not set yet')
    return _backend


def get_func(fname):
    global _backend, _func_dict
    if _backend is None or _func_dict is None:
        raise RuntimeError('Backend is not set yet')
    if fname not in _func_dict:
        raise NotImplementedError(
            'Function %s has not been implemented for backend %s' (fname,
                                                                   _backend))
    return _func_dict[fname]
