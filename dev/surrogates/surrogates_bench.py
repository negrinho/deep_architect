from darch import searchers as se, surrogates as su, core as co, search_logging as sl
# import benchmarks.datasets as datasets
import datasets
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import random
import time
import gc

# TODO: Change the below imports when using PyTorch
import darch.contrib.search_spaces.tensorflow.dnn as search_dnn
from darch.contrib.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from darch.contrib.search_spaces.tensorflow.common import D
from darch.contrib.datasets.dataset import InMemoryDataset
from darch.contrib.datasets.loaders import load_mnist


# Python >= 3.5

# Some requirements of surrogate models:
# 1. They should be fast to evaluate (justification)
# 2. They should be accurate (replacement)

class CharLSTMModel(torch.nn.Module):
    """ The actual CLSTM model
    """
    def __init__(self, character_list, hidden_size, embedding_size):
        super(CharLSTMModel, self).__init__()
        # One embedding feeds 4 LSTMs
        self.embeddings = torch.nn.Embedding(len(character_list), embedding_size)
        # Define ModuleList so modules are properly registered
        self.lstm_in = torch.nn.ModuleList()
        # Define ParameterList so parameters are properly registered
        self.h0 = torch.nn.ParameterList()
        self.c0 = torch.nn.ParameterList()
        for _ in range(4):
            # One LSTM for each input feature
            self.lstm_in.append(torch.nn.LSTM(embedding_size, hidden_size, 1, batch_first=True))
            # Learnable initial hidden states and cell states
            self.h0.append(torch.nn.Parameter(torch.randn(1, 1, hidden_size)))
            self.c0.append(torch.nn.Parameter(torch.randn(1, 1, hidden_size)))
        # LSTM to read the concatenated outputs of the input LSTMs
        self.lstm_out = torch.nn.LSTM(4 * hidden_size, hidden_size, 1)
        self.h_out = torch.nn.Parameter(torch.randn(1, 1, hidden_size))
        self.c_out = torch.nn.Parameter(torch.randn(1, 1, hidden_size))
        self.fc_out = torch.nn.Linear(hidden_size, 1)

    def pack(self, ins, lengths):
        lengths, indices = torch.sort(lengths, dim=0, descending=True)
        # TODO: When pytorch 0.4 comes out, change lengths.data.tolist() to just lengths
        packed = torch.nn.utils.rnn.pack_padded_sequence(ins[indices], lengths.data.tolist(), batch_first=True)
        return packed, indices

    def unsort(self, outs, indices, batch_dim=0):
        unsorted = outs.clone()
        unsorted.scatter_(batch_dim, indices.view(-1, 1).expand_as(outs), outs)
        return unsorted

    # not called for now, useful if we want to unpack the output of rnn (not hidden state)
    def unpack(self, outs, indices):
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(outs, batch_first=True)
        return self.unsort(unpacked, indices)

    def forward(self, feats):
        # Expects feats to be a list of 4 (tensor, seq_lengths) tuples
        batch_size = feats[0][0].size(0)
        outs = []
        for i in range(4): # 4 features in input, hardcoded for now
            out = self.embeddings(feats[i][0])
            packed, indices = self.pack(out, feats[i][1])
            _, (out, _) = self.lstm_in[i](packed, (self.h0[i].repeat(1, batch_size, 1), self.c0[i].repeat(1, batch_size, 1)))
            out = self.unsort(out, indices, batch_dim=1) # Torch docs say batch_dim is dim 1
            outs.append(out)
        out = torch.cat(outs, dim=2)
        _, (out, _) = self.lstm_out(out, (self.h_out.repeat(1, batch_size, 1), self.c_out.repeat(1, batch_size, 1)))
        out = self.fc_out(out[-1]) # Only care about the last layer in event LSTM is stacked
        return out

class CharLSTMSurrogate(su.SurrogateModel):
    """ The CharLSTM Surrogate Function
    """
    # Character LSTM: One LSTM for each feature vector,
    # For a total of 4, then concat the outputs and learn
    character_list = [chr(i) for i in range(ord('A'), ord('Z'))] + [
        chr(i) for i in range(ord('a'), ord('z') + 1)] + [
        chr(i) for i in range(ord('0'), ord('9') + 1)] + [
        '.', ':', '-', '_', '<', '>', '/', '=', '*', ' ', '|']
    char_to_index = {ch : idx for (idx, ch) in enumerate(character_list)}
    # Each input LSTM has an embedding size and hidden state of 128
    embedding_size = 128
    hidden_size = 128

    def __init__(self, batch_size=64, max_batches=16, refit_interval=1, use_cuda=True, val_data=None):
        self.model = CharLSTMModel(self.character_list, self.embedding_size, self.hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss_fn = torch.nn.MSELoss()
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.num_batches = 1
        self.refit_interval = refit_interval
        self.use_cuda = torch.cuda.is_available() and use_cuda

        self.feats, self.vals, self.feats_val, self.vals_val= [], [], [], []

        if self.use_cuda:
            self.model.cuda()

    def preprocess(self, feats):
        # Feats is a dictionary where values are lists of strings. Need to convert this into a tensor
        # Convert strings to tensor by mapping the characters to indices in the character_list
        # Final output is a list of 4 (Long Tensors, seq_len) tuples
        outputs = []
        for feat in feats.values():
            vec_feat = []
            for obj in feat:
                for char in obj:
                    vec_feat.append(self.char_to_index[char])
            output = torch.LongTensor(vec_feat).unsqueeze(0) # Make into row vector for batching later on
            if self.use_cuda:
                output = output.cuda()
            outputs.append((output, output.size(1)))
        return outputs

    def batch(self, feats_batch, val_batch, inference=False):
        # feats_batch is the raw list feats from extract_features, vals is a list of raw metrics from the logger
        # batch preprocesses the feats -> pads them -> batches them into one tensor
        # Preprocess
        for i, feats in enumerate(feats_batch):
            feats_batch[i] = self.preprocess(feats)
        # Pad and batch
        out_batch = [None] * 4
        for i, feat in enumerate(zip(*feats_batch)): # 4 iterations
            seqs, seq_lengths = map(list, zip(*feat))
            max_length = max(seq_lengths)
            for k, seq in enumerate(seqs):
                seqs[k] = torch.nn.functional.pad(seq, (0, max_length - seq.size(1), 0, 0)).data # Possibly incorrect use of F.pad
            feat = torch.cat(seqs, dim=0)
            seq_lengths = feat.new(seq_lengths) # Way to create a cuda.LongTensor if we're using gpu
            feat, seq_lengths = torch.autograd.Variable(feat, volatile=inference), torch.autograd.Variable(seq_lengths, volatile=inference) # no-op in torch v0.4
            out_batch[i] = feat, seq_lengths

        # Batch vals_batch if included
        if val_batch is not None:
            for i, val in enumerate(val_batch):
                val = torch.FloatTensor([[val]])
                if self.use_cuda:
                    val = val.cuda()
                val_batch[i] = val

            val_batch = torch.cat(val_batch, dim=0)
            val_batch = torch.autograd.Variable(val_batch, volatile=inference) # no-op in torch v0.4
        return out_batch, val_batch

    def mini_batch(self, feats, vals, batch_size, num_batches, inference=False):
        # feats and vals are the raw feats from extract_features and a list of vals
        # Yields num_batches many mini batches of size batch_size
        indices = np.arange(len(feats))
        if not inference:
            np.random.shuffle(indices)

        for i, start in enumerate(range(0, len(feats) - batch_size + 1, batch_size)):
            if i > num_batches:
                break
            subset_indicies = indices[start: start + batch_size]
            if vals is not None:
                yield self.batch([feats[i] for i in subset_indicies], [vals[i] for i in subset_indicies], inference=inference)
                # yield batch(feats[subset_indicies], vals[subset_indicies])
            else:
                yield self.batch([feats[i] for i in subset_indicies], [vals[i] for i in subset_indicies], inference=inference)
                # yield batch(feats[subset_indicies], None)

    def eval(self, feats):
        self.model.eval()
        feats = self.preprocess(feats)
        # Need to wrap tensors in autograd.Variable (no-op in torch v0.4)
        for i, feat in enumerate(feats):
            seq, seq_len = feat
            feats[i] = (torch.autograd.Variable(seq, volatile=True), torch.autograd.Variable(seq_len, volatile=True))
        return self.model(feats).data[0, 0]

    def update(self, val, feats):
        # Add datapoint to memory
        self.feats.append(feats)
        self.vals.append(val)
        # Refit only after seing refit_interval many new points
        if len(self.vals) % self.refit_interval == 0 and len(self.feats) > self.batch_size:
            self._refit()

    def _refit(self):
        # Sample batch_size many points from memory to train from
        self.model.train()
        for feats, vals in self.mini_batch(self.feats, self.vals, self.batch_size, self.num_batches):
            self.optimizer.zero_grad()  # Zero out the gradient buffer
            outs = self.model(feats) # Need to compute network output
            loss = self.loss_fn(outs, vals)
            loss.backward()
            self.optimizer.step()

        # Increase the number of batches by 1 (up to a max of max_batches)
        self.num_batches = min(min(self.num_batches + 1, self.max_batches), len(self.feats) // self.batch_size)

    def set_validation_data(self, val_data):
        for eval_log in val_data:
            self.feats_val.append(eval_log['features'])
            self.vals_val.append(eval_log['results']['validation_accuracy'])

    def validate(self):
        if len(self.feats_val) == 0:
            return None
        self.model.eval()
        total_loss = 0.0
        for feats, vals in self.mini_batch(self.feats_val, self.vals_val, self.batch_size, np.inf, inference=True):
            outs = self.model(feats)
            total_loss += torch.nn.MSELoss(size_average=False, reduce=True)(outs, vals).data[0]

        return total_loss / len(self.feats_val)

class CharLSTMSurrogateKeras(CharLSTMSurrogate):
    """ The CharLSTM Surrogate Function ported to Keras
    """
    # This class is largely a HACK, as it extends the torch-based
    # CharLSTMSurrogate (meaning we go from np -> torch -> np -> tf)
    # Character LSTM: One LSTM for each feature vector,
    # For a total of 4, then concat the outputs and learn
    character_list = [chr(i) for i in range(ord('A'), ord('Z'))] + [
        chr(i) for i in range(ord('a'), ord('z') + 1)] + [
        chr(i) for i in range(ord('0'), ord('9') + 1)] + [
        '.', ':', '-', '_', '<', '>', '/', '=', '*', ' ', '|']
    char_to_index = {ch : idx for (idx, ch) in enumerate(character_list)}
    # Each input LSTM has an embedding size and hidden state of 128
    embedding_size = 128
    hidden_size = 128
    def __init__(self, batch_size=2, max_batches=16, refit_interval=1, use_cuda=True, val_data=None):
        # HACK
        super().__init__(batch_size=batch_size, max_batches=max_batches,
            refit_interval=refit_interval, use_cuda=False, val_data=val_data)
        del self.model
        del self.optimizer
        del self.loss_fn
        # Define our model here
        try:
            import keras.backend as K
            from keras.models import Model
            from keras.layers import Input, Dense, LSTM, Embedding, Lambda
            import tensorflow as tf
        except:
            raise ValueError('Please Install Keras/Tensorflow')

        # Backend session work
        if use_cuda:
            gpu_ops = tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(gpu_options=gpu_ops)
            sess = tf.Session(config=config)
            K.tensorflow_backend.set_session(sess)

        # Define our layers
        character_list = self.character_list
        embedding_size = self.embedding_size
        hidden_size = self.hidden_size
        embeddings = Embedding(len(character_list), embedding_size)
        lstm = []
        feat = []
        for _ in range(4):
            lstm.append(LSTM(hidden_size))
            feat.append(Input((None, )))
        lstm_out = LSTM(hidden_size)
        stack = Lambda(lambda x: K.stack(x, axis=1))
        dense = Dense(1)
        # Define computation graph and model
        outs = []
        for i in range(4):
            embedded = embeddings(feat[i])
            out = lstm[i](embedded)
            outs.append(out)
        outs = stack(outs)
        outs = lstm_out(outs)
        outs = dense(outs)
        self.model = Model(inputs=feat, output=outs)
        self.model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
        
    def eval(self, feats):
        feats = self.preprocess(feats)
        output =  self.model.predict_on_batch(feats)
        return output

    def _refit(self):
        # Sample batch_size many points from memory to train from
        for feats, vals in self.mini_batch(self.feats, self.vals, self.batch_size, self.num_batches):
            for i in range(4):
                feats[i] = feats[i][0].data.numpy()
                print(feats)
            vals = vals.data.numpy()
            self.model.train_on_batch(feats, vals)
        
        # Increase the number of batches by 1 (up to a max of max_batches)
        self.num_batches = min(min(self.num_batches + 1, self.max_batches), len(self.feats) // self.batch_size)

    def validate(self):
        if len(self.feats_val) == 0:
            return None
        total_loss = 0.0
        for feats, vals in self.mini_batch(self.feats_val, self.vals_val, self.batch_size, np.inf, inference=True):
            for i in range(4):
                feats[i] = feats[i][0].data.numpy()
            vals = vals.data.numpy()
            total_loss += self.model.test_on_batch(feats, vals)[0] * len(vals)
        return total_loss / len(self.feats_val)


class SearchSpaceFactory:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def get_search_space(self):
        co.Scope.reset_default_scope()
        inputs, outputs = search_dnn.dnn_net(self.num_classes)
        return inputs, outputs, {'learning_rate_init' : D([1e-2, 1e-3, 1e-4, 1e-5])}

def savefig(filename):
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')

def test_clstm_surrogate():
    ## Params:
    dataset_size = 4  # Initial dataset size
    val_size = dataset_size // 2
    num_iters = 0

    ## TODO: Remove Tensorflow dependency so TF doesn't eat up all GPU memory
    ## TODO: Only use PyTorch in the benchmark?

    # ## Choose our dataset and split train/valid/test data
    # datasets.IRIS('.temp/data')
    # # For tensorflow usage, hardcoding in the data, once I change the contrib to PyTorch, can use the dataloaders
    # def iris_labels(label):
    #         if label == b'Iris-setosa':
    #             return 0
    #         elif label == b'Iris-versicolor':
    #             return 1
    #         elif label == b'Iris-virginica':
    #             return 2
    # data = np.loadtxt('.temp/data/iris.data', delimiter=',', usecols=[0,1,2,3])
    # labels = np.loadtxt('.temp/data/iris.data', delimiter=',', usecols=[4], converters={4: iris_labels})
    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33)
    # X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.50)

    # Steal MNIST Tensorflow example data:
    (X_train, y_train, X_valid, y_valid, X_test, y_test) = load_mnist('.temp/data/mnist')
    # Define datasets for contrib tensorflow evaluator
    train_dataset = InMemoryDataset(X_train, y_train, True)
    val_dataset = InMemoryDataset(X_valid, y_valid, False)
    test_dataset = InMemoryDataset(X_test, y_test, False)

    num_classes = 10 # Change this per dataset

    ## Declare the model evaluator
    evaluator = SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes,
        './temp', max_eval_time_in_minutes=1.0, log_output_to_terminal=True)

    ## Define our search space
    search_space_fn = SearchSpaceFactory(num_classes).get_search_space

    ## Define our surrogate models
    # clstm_sur = CharLSTMSurrogate()
    clstm_sur = CharLSTMSurrogateKeras()
    baseline_sur = su.HashingSurrogate(1024, 1)

    ## Choose our searching algorithm, using benchmarking surrogate model
    # We use a random searcher to populate our initial dataset
    # Assumes new model is better than benchmark, and better models -> better searcher
    searcher_rand = se.RandomSearcher(search_space_fn)
    searcher_modl = se.SMBOSearcher(search_space_fn, clstm_sur, 32, 0.2)

    ## Define our Logger (Almost like Replay Memory)
    search_logger = sl.SearchLogger('./logs', 'train', resume_if_exists=True)


    # Samples and evaluates a model from a searcher. Logs the result
    def sample_and_evaluate(searcher):
        # Sample from our searcher
        (inputs, outputs, hs, hyperp_value_lst, searcher_eval_token) = searcher.sample()
         # Get the true score by training the model sampled and log them
        results = evaluator.eval(inputs, outputs, hs)
        # Get the logger for this iteration and log configurations, features, and results
        eval_logger = search_logger.get_current_evaluation_logger()
        eval_logger.log_config(hyperp_value_lst, searcher_eval_token)
        eval_logger.log_features(inputs, outputs, hs)
        eval_logger.log_results(results)
        # Kinda silly, but not the bottleneck
        return sl.read_evaluation_folder(eval_logger.get_evaluation_folderpath())

    def validate(val_data):
        mse_clstm = clstm_sur.validate()
        total_squared_error_baseline = 0.0

        for eval_log in val_data:
            feats = eval_log['features']
            results = eval_log['results']
            total_squared_error_baseline += (baseline_sur.eval(feats) - results['validation_accuracy'])**2

        return total_squared_error_baseline / len(val_data), mse_clstm

    # trains both surrogates on a list of data: (feat,val) tuples
    def train_and_validate(train_data, val_data, validate_interval=10):
        baseline_mse, clstm_mse = [], []
        avg_time = 0
        for i, eval_log in enumerate(random.sample(train_data, len(train_data))):
            feats = eval_log['features']
            results = eval_log['results']
            # Take a backward step for both surrogate models (gradient step)
            baseline_sur.update(results['validation_accuracy'], feats)
            t0 = time.time()
            clstm_sur.update(results['validation_accuracy'], feats)
            avg_time += time.time() - t0
            # Validate
            if val_data is not None and i % validate_interval == 0:
                t0 = time.time()
                baseline_val, clstm_val = validate(val_data)
                print('Iteration {:4d}/{} | num batches: {} | Baseline MSE:{:.4f} | CLSTM MSE:{:.4f} | avg update time: {:.2f} ms | val time: {:.2f} ms'\
                    .format(i, len(train_data), clstm_sur.num_batches, baseline_val, clstm_val, avg_time * 1000 / validate_interval, (time.time() - t0) * 1000))
                baseline_mse.append(baseline_val)
                clstm_mse.append(clstm_val)
                avg_time = 0
        return baseline_mse, clstm_mse

    ## Training loop
    # We train two surrogate models with one searcher based on the non-baseline model
    # Each iteration we sample a model from the searcher
    # We want to populate our dataset with some initial configurations and evaluations
    if search_logger.current_evaluation_id < dataset_size:
        print('Not enough data found, training preliminary models.')
        while search_logger.current_evaluation_id < dataset_size:
            sample_and_evaluate(searcher_rand)

    print('Sufficient data to begin training loop. Dataset size: {}'.format(search_logger.current_evaluation_id))

    # Partition the dataset so the first val_size points are the validation set
    dataset = sl.read_search_folder(search_logger.search_folderpath)
    val_data = dataset[:val_size]
    train_data = dataset[val_size:]
    clstm_sur.set_validation_data(val_data)

    # Train on the data already there
    baseline_mse, clstm_mse = train_and_validate(train_data, val_data)

    # For an additional num_iter iterations, sample and evaluate models using the model searcher

    print('Trained on dataset for 1 epoch. Begin adding new evaluations.')

    for _ in range(num_iters):
        eval_log = sample_and_evaluate(searcher_modl)
        baseline_mse_next, clstm_mse_next = train_and_validate([eval_log], val_data)
        baseline_mse += baseline_mse_next
        clstm_mse += clstm_mse_next

    # Plot the MSEs
    plt.plot(np.arange(len(baseline_mse)), baseline_mse)
    plt.plot(np.arange(len(clstm_mse)), clstm_mse)
    plt.legend(['Baseline MSE', 'CharSLTM MSE'], loc='lower right')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    savefig('surrogate_benchmark')
    print('Baseline MSE:\n{}\nCLSTM MSE:\n{}\n'.format(baseline_mse, clstm_mse))


if __name__ == '__main__':
    test_clstm_surrogate()