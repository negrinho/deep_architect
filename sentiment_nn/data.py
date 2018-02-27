
from create_sentiment_featuresets import create_feature_sets_and_labels

def load_data(use_small=False):
    pos_fpath = 'small_pos.txt' if use_small else 'pos.txt'
    neg_fpath = 'small_neg.txt' if use_small else 'neg.txt'
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels(pos_fpath, neg_fpath)
    train_size = int(len(train_x)*.8)
    train_x, val_x = train_x[:train_size], train_x[train_size:]
    train_y, val_y = train_y[:train_size], train_y[train_size:]
    return {'train' : (train_x, train_y), 'val' : (val_x, val_y)}