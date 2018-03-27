from darch import searchers as se, surrogates as su, core as co, search_logging as sl
import benchmarks.datasets as datasets
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import random

# TODO: Change the below imports when using PyTorch
import darch.contrib.search_spaces.tensorflow.dnn as search_dnn
from darch.contrib.evaluators.tensorflow.classification import SimpleClassifierEvaluator
from darch.contrib.search_spaces.tensorflow.common import D
from darch.contrib.datasets.dataset import InMemoryDataset
from darch.contrib.datasets.loaders import load_mnist
from sklearn.model_selection import train_test_split


# Python >= 3.5

# Some requirements of surrogate models:
# 1. They should be fast to evaluate (justification)
# 2. They should be accurate (replacement)

class CLSTMSurrogateModel(torch.nn.Module):
    """ The actual CLSTM model
    """
    def __init__(self, character_list, hidden_size, embedding_size):
        super(CLSTMSurrogateModel, self).__init__()
        # One embedding feeds 4 LSTMs
        self.embeddings = torch.nn.Embedding(len(character_list), embedding_size)
        # Define ModuleList so modules are properly registered
        self.lstm_in = torch.nn.ModuleList()
        # Define ParameterList so parameters are properly registered
        self.h0 = torch.nn.ParameterList()
        self.c0 = torch.nn.ParameterList()
        for i in range(4):
            # One LSTM for each input feature
            self.lstm_in.append(torch.nn.LSTM(embedding_size, hidden_size, 1))
            # Learnable initial hidden states and cell states
            self.h0.append(torch.nn.Parameter(torch.randn(1, 1, hidden_size)))
            self.c0.append(torch.nn.Parameter(torch.randn(1, 1, hidden_size)))
        # LSTM to read the concatenated outputs of the input LSTMs
        self.lstm_out = torch.nn.LSTM(4 * hidden_size, hidden_size, 1)
        self.h_out = torch.nn.Parameter(torch.randn(1, 1, hidden_size))
        self.c_out = torch.nn.Parameter(torch.randn(1, 1, hidden_size))
        self.fc_out = torch.nn.Linear(hidden_size, 1)

    def forward(self, feats):
        out_0 = self.embeddings(feats[0]).unsqueeze(1)
        out_1 = self.embeddings(feats[1]).unsqueeze(1)
        out_2 = self.embeddings(feats[2]).unsqueeze(1)
        out_3 = self.embeddings(feats[3]).unsqueeze(1)
        _, (out_0, _) = self.lstm_in[0](out_0, (self.h0[0], self.c0[0]))
        _, (out_1, _) = self.lstm_in[1](out_1, (self.h0[1], self.c0[1]))
        _, (out_2, _) = self.lstm_in[2](out_2, (self.h0[2], self.c0[2]))
        _, (out_3, _) = self.lstm_in[3](out_3, (self.h0[3], self.c0[3]))
        out = torch.cat((out_0, out_1, out_2, out_3), dim=2)
        _, (out, _) = self.lstm_out(out, (self.h_out, self.c_out))
        out = self.fc_out(out)
        return out



class CLSTMSurrogate(su.SurrogateModel):
    """ The CLSTM Surrogate Function
    """
    # Character LSTM: One LSTM for each feature vector,
    # For a total of 3, then concat the outputs and learn
    character_list = [chr(i) for i in range(ord('A'), ord('Z'))] + [
        chr(i) for i in range(ord('a'), ord('z') + 1)] + [
        chr(i) for i in range(ord('0'), ord('9') + 1)] + [
        '.', ':', '-', '_', '<', '>', '/', '=', '*', ' ', '|']
    char_to_index = {ch : idx for (idx, ch) in enumerate(character_list)}
    # Each input LSTM has an embedding size and hidden state of 128
    embedding_size = 128
    hidden_size = 128

    def __init__(self, max_batch_size=1024, refit_interval=1, use_cuda=True):
        self.model = CLSTMSurrogateModel(self.character_list, self.embedding_size, self.hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss_fn = torch.nn.MSELoss()
        self.feats = []
        self.vals = []
        self.batch_size = 1
        self.max_batch_size = max_batch_size
        self.refit_interval = refit_interval
        self.use_cuda = torch.cuda.is_available() and use_cuda

        if self.use_cuda:
            self.model.cuda()

    def preprocess(self, feats):
        # Feats is a dictionary where values are lists of strings. Need to convert this into a tensor
        # Convert strings to tensor by mapping the characters to indicies in the character_list
        # Final output is a list of 4 Long Tensors
        outputs = []
        for i, feat in enumerate(feats.values()):
            vec_feat = []
            for obj in feat:
                for char in obj:
                    vec_feat.append(self.char_to_index[char])
            output = torch.LongTensor(vec_feat)
            if self.use_cuda:
                output = output.cuda()
            output = torch.autograd.Variable(output) # This will become a no-op in PyTorch 0.4
            outputs.append(output)
        return outputs

    def eval(self, feats):
        return self.model(self.preprocess(feats)).data[0, 0, 0]

    def update(self, val, feats):
        # Add datapoint to memory
        self.feats.append(self.preprocess(feats))
        self.vals.append(val)
        # Refit only after seing refit_interval many new points
        if len(self.vals) % self.refit_interval == 0:
            self._refit()
        
    def _refit(self):
        # Sample batch_size many points from memory to train from
        for feats, val in random.sample(list(zip(self.feats, self.vals)), self.batch_size):  
            self.optimizer.zero_grad()  # Zero out the gradient buffer
            out = self.model(feats) # Need to compute network output
            # Wrap true value in a Float Tensor Variable
            val = torch.FloatTensor([[[val]]])
            if self.use_cuda:
                val = val.cuda()
            val = torch.autograd.Variable(val) # This will become a no-op in PyTorch 0.4

            loss = self.loss_fn(out, val)
            loss.backward()
            self.optimizer.step()
        # Increase the batch_size by a power of two
        self.batch_size = min(len(self.feats), min(self.batch_size * 2, self.max_batch_size))


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
    dataset_size = 4096 # Initial dataset size
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
    clstm_sur = CLSTMSurrogate()
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
        total_squared_error_baseline = 0.0
        total_squared_error_clstm = 0.0
        for eval_log in val_data:
            feats = eval_log['features']
            results = eval_log['results']
            total_squared_error_baseline += (baseline_sur.eval(feats) - results['val_acc'])**2
            total_squared_error_clstm += (clstm_sur.eval(feats) - results['val_acc'])**2
        return total_squared_error_baseline / len(val_data), total_squared_error_clstm / len(val_data)
        

    # trains both surrogates on a list of data: (feat,val) tuples
    def train_and_validate(train_data, val_data, validate_interval=10):
        baseline_mse, clstm_mse = [], []
        for i, eval_log in enumerate(random.sample(train_data, len(train_data))):
            feats = eval_log['features']
            results = eval_log['results']
            # Take a backward step for both surrogate models (gradient step)
            baseline_sur.update(results['val_acc'], feats)
            clstm_sur.update(results['val_acc'], feats)
            # Validate data
            # TODO: Validate every x iterations
            if val_data is not None and i % validate_interval == 0:
                baseline_val, clstm_val = validate(val_data)
                print('Iteration {:4d}/{} \t|\t Baseline MSE:{}\t CLSTM MSE:{}'.format(i, len(train_data), baseline_val, clstm_val))
                baseline_mse.append(baseline_val)
                clstm_mse.append(clstm_val)
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
    plt.legend(['Baseline MSE', 'CSLTM MSE'], loc='lower right')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    savefig('surrogate_benchmark')
    print('Baseline MSE:\n{}\nCLSTM MSE:\n{}\n'.format(baseline_mse, clstm_mse))


if __name__ == '__main__':
    test_clstm_surrogate()