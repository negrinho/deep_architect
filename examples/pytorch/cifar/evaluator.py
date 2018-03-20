import os
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import darch.helpers.pytorch as hpt


# TODO check/delete comments
def start_fn(d):
    outputs = d['darch']['outputs']

    # Init dataset
    if not os.path.isdir(d['args'].data_path):
        os.makedirs(d['args'].data_path)

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])
    if d['args'].dataset == 'cifar10':
        train_data = dset.CIFAR10(d['args'].data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(d['args'].data_path, train=False, transform=test_transform, download=True)
    else:
        train_data = dset.CIFAR100(d['args'].data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(d['args'].data_path, train=False, transform=test_transform, download=True)
    d['train_loader'] = torch.utils.data.DataLoader(train_data, batch_size=d['args'].batch_size, shuffle=True,
                                                    num_workers=d['args'].prefetch, pin_memory=True if d['args'].ngpu
                                                                                                       > 0 else False)
    d['test_loader'] = torch.utils.data.DataLoader(test_data, batch_size=d['args'].test_bs, shuffle=False,
                                                   num_workers=d['args'].prefetch, pin_memory=True if d['args'].ngpu
                                                                                                       > 0 else False)

    # Init checkpoints
    if not os.path.isdir(d['args'].save):
        os.makedirs(d['args'].save)

    # Init model, criterion, and optimizer
    # net = CifarResNeXt(hs['cardinality'], hs['depth'], nlabels, hs['base_width'], hs['widen_factor'])
    # d['net'] = net
    # print(net)
    data, _ = next(iter(d['train_loader']))
    d['net']({ 'In': torch.autograd.Variable(data) })  # make a preemptive forward pass to initialize things

    if d['args'].ngpu > 1:
        d['net'] = torch.nn.DataParallel(d['net'], device_ids=list(range(d['args'].ngpu)))
        hpt.cuda(outputs.values())
        # data = data.cuda()
        # TODO make darch-compatible for multi-gpu

    elif d['args'].ngpu > 0:
        hpt.cuda(list(outputs.values()))

    optimizer = torch.optim.SGD(hpt.parameters(outputs.values()), d['learning_rate'], momentum=d['momentum'],
                                weight_decay=d['decay'], nesterov=True)
    d['optimizer'] = optimizer


# train function (forward, backward, update)
def train_fn(d):
    net = d['net']
    net.train()
    loss_avg = 0.0
    for batch_idx, (data, target) in enumerate(d['train_loader']):
        if d['args'].ngpu > 0:
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
        else:
            data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)

        # forward
        output = net({ 'In': data })

        # backward
        d['optimizer'].zero_grad()
        loss = F.cross_entropy(output['Out'], target)
        loss.backward()
        d['optimizer'].step()

        # exponential moving average
        loss_avg = loss_avg * 0.2 + loss.data[0] * 0.8
    d['train_loss'] = loss_avg


def is_over_fn(d):
    if 'cur_epoch' not in d:
        d['cur_epoch'] = 0
    d['cur_epoch'] += 1
    return d['args'].epochs < d['cur_epoch']


def end_fn(d):
    d['cur_epoch'] = 0
    net = d['net']
    net.eval()
    loss_avg = 0.0
    correct = 0
    for batch_idx, (data, target) in enumerate(d['test_loader']):
        if d['args'].ngpu > 0:
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
        else:
            data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)

        # forward
        output = net({ 'In': data })
        loss = F.cross_entropy(output['Out'], target)

        # accuracy
        pred = output['Out'].data.max(1)[1]
        correct += pred.eq(target.data).sum()

        # test loss average
        loss_avg += loss.data[0]

    d['test_loss'] = loss_avg / len(d['test_loader'])
    d['test_accuracy'] = correct / len(d['test_loader'].dataset)


def get_eval_fn():
    def eval_fn(d):
        start_fn(d)
        while not is_over_fn(d):
            train_fn(d)
        end_fn(d)
    return eval_fn
