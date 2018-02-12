import numpy as np
import pandas as pd
import research_toolbox.tb_preprocessing as tb_pr
from sklearn.model_selection import train_test_split
import search_space as ss
import evaluator as ev
import darch.searchers as se
import darch.utils as ut

def extract_images(d):
    X_band_1 = np.array([np.array(band, dtype=np.float32).reshape(75, 75) 
        for band in d["band_1"]])
    X_band_2 = np.array([np.array(band, dtype=np.float32).reshape(75, 75) 
        for band in d["band_2"]])
    X = np.concatenate([
                X_band_1[:, :, :, np.newaxis], 
                X_band_2[:, :, :, np.newaxis]], axis=-1)
    return X

def load_data():
    train = pd.read_json("kaggle/iceberg/data/train.json")
    test = pd.read_json("kaggle/iceberg/data/test.json")
    
    Xtrain = extract_images(train)
    ytrain = np.array(train['is_iceberg'], dtype=np.int32)
    ytrain = tb_pr.idx_to_onehot(ytrain, 2)
    Xtest = extract_images(test)
    Xtrain = tb_pr.per_image_whiten(Xtrain)
    Xtest = tb_pr.per_image_whiten(Xtest)
    test_ids = list(test['id'])
    
    Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, 
        random_state=1, train_size=0.75)

    # # ## Fake data for fast debugging.
    # Xtrain = np.random.random((10, 75, 75, 2))
    # ytrain = np.zeros((10, 2), dtype=np.int32)
    # Xval = np.random.random((10, 75, 75, 2))
    # yval = np.zeros((10, 2), dtype=np.int32)
    # Xtest = np.random.random((10, 75, 75, 2))
    # ytrain[:, 0] = 1.0
    # yval[:, 0] = 1.0
    # test_ids = ['a'] * Xtest.shape[0]

    # Xtrain = Xtrain[:10]
    # ytrain = ytrain[:10]

    return Xtrain, ytrain, Xval, yval, Xtest, test_ids

if __name__ == '__main__':
    (Xtrain, ytrain, Xval, yval, Xtest, test_ids) = load_data()

    d = {'data' : {
            'train' : (Xtrain, ytrain), 
            'val' : (Xval, yval), 
            'test' : Xtest,
            'test_ids' : test_ids},
         'cfg' : {
             'in_d' : (75, 75, 2), 
             'num_classes' : 2, 
             'batch_size' : 128,
             'max_minutes_per_model' : 120.0,
             'max_num_epochs' : 1000,
             'sgd_momentum' : 0.99,
             'model_path' : './temp',
             'preds_path' : './temp/preds.txt'} } 
    
    search_space_fn = ss.get_ss2_fn(d['cfg']['num_classes'])
    # searcher = se.MCTSearcher(search_space_fn)
    searcher = se.RandomSearcher(search_space_fn)

    eval_fn = ev.get_eval_fn()
    for idx in xrange(16):
        d['cfg']['preds_path'] = './temp/preds%d.txt' % idx
        (inputs, outputs, hs, vs, cfg_d) = searcher.sample()
        # ut.draw_graph(outputs.values(), True, True)
        d['darch'] = {'inputs' : inputs, 'outputs' : outputs, 'hs' : hs}
        eval_fn(d)
        v = max(d['log']['val_acc'])
        searcher.update(v, cfg_d)

        
    # (inputs, outputs, hs) = search_space_fn()
    # se.random_specify(outputs.values(), hs)
    # ut.draw_graph(outputs.values())

    # searcher = se.RandomSearcher(search_space_fn, hyperparameters_fn)
    
    # for i in xrange(args.num_samples):
    #     model_folderpath = os.path.join(
    #         'experiments', exp_folderpath, 'model.%d' % i)
    #     # print(model_folderpath)
    #     if not os.path.exists(model_folderpath):
    #         os.makedirs(model_folderpath)
    #     evaluator.out_folderpath = model_folderpath

    #     se.run_searcher(searcher, evaluator, 1)

# TODO: add information about searchers.
# TODO: add information about other stuff.
# TODO: the hyperparameters have to have a way of distinguishing between
# different search spaces, for example, mnist and something else.