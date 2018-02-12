### for simple NLP
def keep_short_sentences(sents, maxlen):
    return [s for s in sents if len(s) <= maxlen]
# TODO: change this for sequences maybe. simpler.

def count_tokens(sents):
    tk_to_cnt = {}
    for s in sents:
        for tk in s:
            if tk in tk_to_cnt:
                tk_to_cnt[tk] += 1
            else:
                tk_to_cnt[tk] = 1
    return tk_to_cnt

def keep_most_frequent_tokens(tk_to_cnt, num_tokens):
    top_items = sorted(tk_to_cnt.items(), 
        key=lambda x: x[1], reverse=True)[:num_tokens]
    return dict(top_items)

def remove_rare_tokens(tk_to_cnt, keep_thres):
    return {tk : c for (tk, c) in tk_to_cnt.iteritems() if c >= keep_thres}

def index_tokens(tokens, start_index=0):
    assert start_index >= 0
    num_tokens = len(tokens)
    tk_to_idx = dict(zip(tokens, range(start_index, num_tokens + start_index)))
    return tk_to_idx

def reverse_mapping(a_to_b):
    b_to_a = dict([(b, a) for (a, b) in a_to_b.iteritems()])
    return b_to_a

def reverse_nonunique_mapping(a_to_b):
    d = {}
    for (x, y) in a_to_b.iteritems():
        if y not in d:
            d[y] = []
        d[y].append(x)
    return d

def preprocess_sentence(sent, tk_to_idx, 
    unk_idx=-1, bos_idx=-1, eos_idx=-1, 
    bos_times=0, eos_times=0):
    """If no unk_idx is specified and there are tokens that are sentences that 
    have tokens that are not in the dictionary, it will throw an exception.
    it is also possible to look at padding using this type of functionality.
    """
    assert bos_idx == -1 or (bos_idx >= 0 and bos_times >= 0)  
    assert eos_idx == -1 or (eos_idx >= 0 and eos_times >= 0)  
    assert (unk_idx == -1 and all([tk in tk_to_idx for tk in sent])) or (unk_idx >= 0)

    proc_sent = []
    # adding the begin of sentence tokens
    if bos_idx != -1:
        proc_sent.extend([bos_idx] * bos_times)

    # preprocessing the sentence
    for tk in sent:
        if tk in tk_to_idx:
            idx = tk_to_idx[tk]
            proc_sent.append(idx)
        else:
            proc_sent.append(unk_idx)
            
    # adding the end of sentence token
    if eos_idx != -1:
        proc_sent.extend([eos_idx] * eos_times)

    return proc_sent

### TODO: this needs to be standardized. what is the typical information that 
# these functions take. 
# NOTE: perhaps focus on simple images, and
### for images; some data augmentation
import numpy as np
import cv2

def onehot_to_idx(y_onehot):
    y_idx = np.where(y_onehot > 0.0)[1]

    return y_idx

def idx_to_onehot(y_idx, num_classes):
    num_images = y_idx.shape[0]
    y_one_hot = np.zeros( (num_images, num_classes), dtype='float32')
    y_one_hot[ np.arange(num_images),  y_idx ] = 1.0

    return y_one_hot

# NOTE: many of 
# NOTE: this can be improved by dealing with arbitrary dimensions, but it is
# a simple example for now. kind of assumes 4 dimensions. it is OK, just 
# augment the image with an extra dimension.
# the application of this to four dimensional images is nice.
def center_crop(X, out_height, out_width):
    num_examples, height, width, num_channels = X.shape
    assert out_height <= height and out_width <= width

    start_i = (height - out_height) / 2
    start_j = (width - out_width) / 2
    X_out = X[:, start_i : start_i + out_height, start_j : start_j + out_width, :]  

    return X_out

# random crops for each of the images.
def random_crop(X, out_height, out_width):
    num_examples, height, width, num_channels = X.shape
    # the ouput dimensions have to be smaller or equal that the input dimensions.
    assert out_height <= height and out_width <= width

    start_is = np.random.randint(height - out_height + 1, size=num_examples)
    start_js = np.random.randint(width - out_width + 1, size=num_examples)
    X_out = []
    for ind in xrange(num_examples):
        st_i = start_is[ind]
        st_j = start_js[ind]

        X_out_i = X[ind, st_i : st_i + out_height, st_j : st_j + out_width, :]
        X_out.append(X_out_i)

    X_out = np.array(X_out)
    return X_out

def random_flip_left_right(X, p_flip):
    num_examples, height, width, num_channels = X.shape

    X_out = X.copy()
    flip_mask = np.random.random(num_examples) < p_flip
    X_out[flip_mask] = X_out[flip_mask, :, ::-1, :]

    return X_out

def per_image_whiten(X):
    """ Subtracts the mean of each image in X and renormalizes them to unit norm.

    """
    num_examples, height, width, num_channels = X.shape

    X_flat = X.reshape((num_examples, -1))
    X_mean = X_flat.mean(axis=1)
    X_cent = X_flat - X_mean[:, None]
    X_norm = np.sqrt( np.sum( X_cent * X_cent, axis=1) ) 
    X_out = X_cent / X_norm[:, None]
    X_out = X_out.reshape(X.shape) 

    return X_out

# Assumes the following ordering for X: (num_images, height, width, num_channels)
def zero_pad_border(X, pad_size):
    num_examples, height, width, num_channels = X.shape
    X_padded = np.zeros((num_examples, height + 2 * pad_size, width + 2 * pad_size, 
        num_channels), dtype='float32')
    X_padded[:, pad_size:height + pad_size, pad_size:width + pad_size, :] = X
    
    return X_padded

def random_scale_rotate(X, angle_min, angle_max, scale_min, scale_max):
    num_examples, height, width, num_channels = X.shape
    scales = np.random.uniform(scale_min, scale_max, size=num_examples)
    angles = np.random.uniform(angle_min, angle_max, size=num_examples)

    out_lst = []
    rot_center = (height / 2, width / 2)
    for i in xrange(num_examples):
        A = cv2.getRotationMatrix2D(rot_center, angles[i], scales[i])
        out = cv2.warpAffine(X[i], A, (height, width))
        out_lst.append(out)
    X_out = np.stack(out_lst)
    # it seems that if there is a single channel, it disappears.
    if num_channels == 1:
        X_out = np.expand_dims(X_out, 3)
    return X_out
