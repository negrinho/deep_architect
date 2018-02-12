from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import darch.modules as mo
import darch.helpers.pytorch as ph
import darch.core as connect_sequentially
import darch.utils as ut
from darch.hyperparameters import *
import darch.searchers as se
import darch.core as co

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--use-darch', action='store_true', default=False,
                    help='use the darch model insteand (default: 0)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

# original model.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# generated search space.
def get_model():
    PyTFn = mo.WrappedSISOCompiledFn
    PyTMo = ph.PyTSISOModule
    relu_fn = lambda : PyTMo('ReLU', {}, lambda : nn.ReLU()) 
    logsoftmax_fn = lambda : PyTMo('LogSoftmax', {}, lambda : nn.LogSoftmax())  
    dropout2d_fn = lambda : PyTMo('Dropout2d', {}, lambda : nn.Dropout2d())
    flatten_fn = lambda : PyTFn('Flatten', {}, lambda : lambda In: In.view(In.size(0), -1))
    dropout_fn = lambda : PyTMo('Dropout', {}, lambda : nn.Dropout())
    
    linear_fn = lambda feature_sizes: PyTMo('Linear', 
        {'out_features' : Discrete(feature_sizes)}, 
        lambda In, out_features: nn.Linear(In.size(1), out_features))
    max_pool_2d_fn = lambda kernel_sizes : PyTMo('MaxPool2d', 
            {'kernel_size' : Discrete(kernel_sizes)}, 
            lambda kernel_size: nn.MaxPool2d(kernel_size))
    conv2d_fn = lambda out_sizes, kernel_sizes: PyTMo('Conv2d', 
            {'kernel_size' : Discrete(kernel_sizes), 'out_channels' : Discrete(out_sizes)}, 
            lambda In, kernel_size, out_channels: nn.Conv2d(In.size(1), out_channels, kernel_size))
    
    xs = [conv2d_fn([10], [5]), max_pool_2d_fn([2]), relu_fn(), 
        conv2d_fn([20], [5]), dropout2d_fn(), max_pool_2d_fn([2]), relu_fn(),
        flatten_fn(), linear_fn([50]), relu_fn(), dropout_fn(), 
        linear_fn([10]), logsoftmax_fn()
    ]
    ut.connect_sequentially(xs)
    # ut.draw_graph(self.xs)
    se.random_specify(xs[-1].outputs.values())
    return ph.PyTNetContainer(xs[0].inputs, xs[-1].outputs)

if args.use_darch:
    model = get_model()
    model({'In' : Variable( iter(train_loader).next()[0] )})
else:
    model = Net()

# model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        if args.use_darch:
            output = model({'In' : data})['Out']
        else:
            output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        if args.use_darch:
            output = model({'In' : data})['Out']
        else:
            output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

import time
t = time.time()
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
print(time.time() - t)
