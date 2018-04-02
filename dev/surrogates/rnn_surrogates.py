import darch.surrogates as su
from six import itervalues
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as opt
import numpy as np
import torch

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

    # NOTE: val_data is unused for now.
    def __init__(self, batch_size=64, max_batches=16, refit_interval=1, use_cuda=True, val_data=None):
        self.model = CharLSTMModel(self.character_list, self.embedding_size, self.hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.mse = torch.nn.MSELoss()
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.num_batches = 1
        self.refit_interval = refit_interval
        self.use_cuda = torch.cuda.is_available() and use_cuda

        # NOTE: why are there four of these?
        # NOTE: that these are lists.
        self.feats = []
        self.vals = []
        self.feats_val = []
        self.vals_val = []

        if self.use_cuda:
            self.model.cuda()

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
            loss = self.mse(outs, vals)
            loss.backward()
            self.optimizer.step()

        # Increase the number of batches by 1 (up to a max of max_batches)
        self.num_batches = min(min(self.num_batches + 1, self.max_batches), len(self.feats) // self.batch_size)

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

    ### TODO: I don't know this should be here. perhaps all this can be outside
    # this model or
    def batch(self, feats_batch, val_batch, inference=False):
        # feats_batch is the raw list feats from extract_features, vals is a list of raw metrics from the logger
        # batch preprocesses the feats -> pads them -> batches them into one tensor

        # Preprocess
        for i, feats in enumerate(feats_batch):
            feats_batch[i] = self.preprocess(feats)
        # Pad and batch
        for i, feat in enumerate(zip(*feats_batch)):
            seqs, seq_lengths = map(list, zip(*feat))
            max_length = max(seq_lengths)
            for k, seq in enumerate(seqs):
                seqs[k] = torch.nn.functional.pad(seq, (0, max_length - seq.size(1), 0, 0)).data # Possibly incorrect use of F.pad
            feat = torch.cat(seqs, dim=0)
            seq_lengths = feat.new(seq_lengths) # Way to create a cuda.LongTensor if we're using gpu
            feat, seq_lengths = torch.autograd.Variable(feat, volatile=inference), torch.autograd.Variable(seq_lengths, volatile=inference) # no-op in torch v0.4
            feats_batch[i] = feat, seq_lengths

        # Batch vals_batch if included
        if val_batch is not None:
            for i, val in enumerate(val_batch):
                val = torch.FloatTensor([[val]])
                if self.use_cuda:
                    val = val.cuda()
                val_batch[i] = val

            val_batch = torch.cat(val_batch, dim=0)
            val_batch = torch.autograd.Variable(val_batch, volatile=inference) # no-op in torch v0.4
        return feats_batch, val_batch

    def mini_batch(self, feats, vals, batch_size, num_batches, inference=False):
        # feats and vals are the raw feats from extract_features and a list of vals
        # Yields num_batches many mini batches of size batch_size
        indices = np.arange(len(feats))
        if not inference:
            np.random.shuffle(indices)

        for i, start in enumerate(range(0, len(feats) - batch_size + 1, batch_size)):
            if i > num_batches:
                break
            subset_indices = indices[start: start + batch_size]
            if vals is not None:
                yield self.batch([feats[i] for i in subset_indices], [vals[i] for i in subset_indices], inference=inference)
            else:
                yield self.batch([feats[i] for i in subset_indices], [vals[i] for i in subset_indices], inference=inference)

### Simplified model.
def process_feats_for_charlstm(ch2idx, feats):
    all_feats_lst = []
    for fs in itervalues(feats):
        all_feats_lst.extend(fs)
    target_len = max([len(f) for f in all_feats_lst]) + 2
    # print target_len

    idxs_lst = []
    for feat in all_feats_lst:
        idxs = [ch2idx['*']]
        for ch in feat:
            idxs.append(ch2idx[ch])
        while len(idxs) < target_len:
            idxs.append(ch2idx['*'])
        idxs_lst.append(idxs)
    # num_feats (batch) x feat_size (time)
    # print idxs_lst
    return idxs_lst

class RNNModel(nn.Module):
    def __init__(self, num_characters, embedding_size, hidden_size):
        nn.Module.__init__(self)
        self.ch_embs = nn.Embedding(num_characters, embedding_size)
        self.rnn = torch.nn.LSTM(embedding_size, hidden_size, 1)
        # self.rnn = torch.nn.GRU(embedding_size, hidden_size, 1)
        self.reg = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, vec):
        # vec: feat_len x num_feats
        out = self.ch_embs(vec)
        (h0, c0) = (Variable(torch.zeros(1, out.size(1), out.size(2))),
            Variable(torch.zeros(1, out.size(1), out.size(2))))

        # print "0", vec.size()
        # print "1", out
        # print "2", self.rnn(out, (h0, c0))
        # print "3", out.size()
        # print "4", self.rnn(out, (h0, c0)).size()
        # assert False

        _, (_, out) = self.rnn(out, (h0, c0))
        out = self.reg(out.mean(1))
        return out

class RNNSurrogate(su.SurrogateModel):
    def __init__(self, embedding_size, hidden_size,
            validation_frac=0.25, refit_interval=1,
            use_surrogate_after_num_samples=16,
            max_num_updates_per_refit=1024, patience=2,
            compute_validation_every_num_updates=128):

        assert int(1.0 / validation_frac) <= use_surrogate_after_num_samples

        self.ch_lst = [chr(i) for i in range(ord('A'), ord('Z'))] + [
            chr(i) for i in range(ord('a'), ord('z') + 1)] + [
            chr(i) for i in range(ord('0'), ord('9') + 1)] + [
                '.', ':', '-', '_', '<', '>', '/', '=', '*', ' ', '|']
        self.ch2idx = {ch : idx for (idx, ch) in enumerate(self.ch_lst)}

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.validation_frac = validation_frac
        self.use_surrogate_after_num_samples = use_surrogate_after_num_samples
        self.max_num_updates_per_refit = max_num_updates_per_refit
        self.patience = patience
        self.compute_validation_every_num_updates = compute_validation_every_num_updates
        self.refit_interval = refit_interval

        self.model = RNNModel(len(self.ch_lst), self.embedding_size, self.hidden_size)
        self.optimizer = opt.Adam(self.model.parameters(), 1e-2)
        self.loss_fn = nn.MSELoss()
        self.feats_lst = []
        self.val_lst = []
        self.vec_lst = []

    def update(self, val, feats):
        self.feats_lst.append(feats)
        self.val_lst.append(val)

        idxs_lst = process_feats_for_charlstm(self.ch2idx, feats)
        vec = torch.LongTensor(idxs_lst).t()
        self.vec_lst.append(vec)

        n = len(self.feats_lst)
        if (n == self.use_surrogate_after_num_samples) or (
            n > self.use_surrogate_after_num_samples and n % self.refit_interval == 0):
            self._refit()

    def eval(self, feats):
        if len(self.feats_lst) >= self.use_surrogate_after_num_samples:
            idxs_lst = process_feats_for_charlstm(self.ch2idx, feats)
            vec = Variable(torch.LongTensor(idxs_lst).t())
            pred_val = self.model(vec).data[0, 0]
        else:
            pred_val = np.mean(self.val_lst) if len(self.val_lst) > 0 else 0.0
        return pred_val

    def _refit(self):
        n = len(self.feats_lst)
        for _ in xrange(16):
            i = np.random.randint(n)
            true_val = Variable(torch.FloatTensor([self.val_lst[i]]))
            vec = Variable(self.vec_lst[i])
            pred_val = self.model(vec)
            loss = self.loss_fn(pred_val, true_val)

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
            self.optimizer.step()

# NOTE: do the loss fn or something like that. loss_fn.
# that is the only thing that needs to be done. as long as it improves
# in the training set, it works.

# NOTE: this tries to sort the models correctly rather than predict performance.
class RankingRNNSurrogate(RNNSurrogate):
    def __init__(self, embedding_size, hidden_size):
        RNNSurrogate.__init__(self, embedding_size, hidden_size)
        self.bce = nn.BCEWithLogitsLoss()

    def _refit(self):
        n = len(self.feats_lst)
        if n > 1:
            for _ in xrange(64):
                i = np.random.randint(n)
                while True:
                    i_other = np.random.randint(n)
                    if i_other != i:
                        break

                vec = Variable(self.vec_lst[i])
                score = self.model(vec)
                vec_other = Variable(self.vec_lst[i_other])
                score_other = self.model(vec_other)
                # print score.size(), score_other.size()

                label = float(self.val_lst[i] >= self.val_lst[i_other])
                label = Variable(torch.FloatTensor([[label]]))

                # s = score.data[0, 0]
                # s_other = score_other.data[0, 0]
                # lab = label.data[0, 0]
                # print s, s_other, lab, self.val_lst[i], self.val_lst[i_other], ((s > s_other) and lab == 0.0) or ((s < s_other) and lab == 1.0)

                self.optimizer.zero_grad()
                loss = self.bce(score - score_other, label)
                loss.backward()
                # torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
                self.optimizer.step()

# TODO: add a random pair function
# essentially the same with a different training scheme.
