# -*- coding: utf-8 -*-
from __future__ import division

""" 
Trains a ResNeXt Model on Cifar10 and Cifar 100. Implementation as defined in:

Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.


MIT License

Copyright (c) 2017 Pau Rodriguez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import argparse
import darch.searchers as se
import darch.helpers.pytorch as hpt
from examples.pytorch.cifar.evaluator import get_eval_fn
import examples.pytorch.cifar.search_space as ss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('data_path', type=str, help='Root for the Cifar dataset.')
    parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'], help='Choose between Cifar10/100.')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_bs', type=int, default=10)
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./', help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    # Architecture
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--log', type=str, default='./', help='Log folder.')
    args = parser.parse_args()

    d = {
        'args': args,
        'darch': None,
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'decay': args.decay
    }

    # Main loop
    search_space_fn = ss.get_ss_fn(10)  # TODO change to parse nr from input
    searcher = se.MCTSearcher(search_space_fn)

    eval_fn = get_eval_fn()
    for _ in range(10):
        (inputs, outputs, hs, vs, cfg_d) = searcher.sample()
        d['net'] = hpt.PyTNetContainer(inputs, outputs)
        d['darch'] = {'inputs' : inputs, 'outputs' : outputs, 'hs' : hs}
        eval_fn(d)
        print(d['train_loss'], d['test_loss'], d['test_accuracy'])
