#!/usr/bin/env python3
from dataloader import create_split_loaders
from model import GenericRNN
import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy
import os
import sys

from utility import *

config = {'chunk_size':100, 'type_number':93, 'hidden':100, 
          'learning_rate':0.001, 'early_stop':True, 'patience_threshold':10, 
          'epoch_num':10, 'N':50, 'M':500, 'seed':1, 'model':'LSTM',
          'model_path':'model_weights', 'T':1}

def init(seed=None, chunk_size=None, type_number=None, hidden=None, learning_rate=None,
         early_stop=None, patience_threshold=None, epoch_num=None, model_path=None, N=None,
         M=None, model=None, T=1, **kwargs):
    global net
    global c_list
    global forward

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 1, "pin_memory": True}
        print("CUDA is supported")
    else:
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # receive train, validation, test data
    train, valid, test, c_to, one_to = create_split_loaders(chunk_size,extras)
    
    # construct network
    net = GenericRNN(type_number, hidden, type_number, model)
    # if model already exists, then load the previous one
    if os.path.exists(model_path+'.pkl'):
        print('loading previous model...')
        load_net_state(net, model_path+'.pkl')
    net = net.to(computing_device)

    data = []
    with open('pa4Data/train.txt', 'r') as file:
        for line in file:
            data += line
    c_list = list(set(data))
    c_list.sort()
    c_list = np.array(c_list)

    def forward(c, h):
        return net.forward(c_to(c).view(1, -1).to(computing_device), h)

def sample(seed=None, chunk_size=None, type_number=None, hidden=None, learning_rate=None,
           early_stop=None, patience_threshold=None, epoch_num=None, model_path=None, N=None,
           M=None, model=None, T=1, **kwargs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('====== T = {}, seed = {} ======'.format(T, seed))
    with torch.no_grad():
        start_str = '<start>\n'
        out_str, h = start_str, None
        for ii in range(200):
            # if the net prediction will still be inside start_str, we want to
            # ignore the output; we are effectively just priming the net with a
            # good hidden initialization
            if ii < len(out_str) - 1:
                out, h = forward(start_str[ii], h)
                continue
            out, h = forward(out_str[-1], h)
            p = out.cpu()
            if T != 'argmax':
                p /= T # Apply temperature
                p = F.softmax(p, dim=1)
                p = np.array(p).flatten()
                p /= sum(p)
                sampled_out = np.random.choice(c_list, p=p)
            else:
                p = p.argmax()
                sampled_out = c_list[p]
            out_str += sampled_out
        print(out_str)

if __name__ == '__main__':
    init(**config)
    for T in [0.5, 0.7, 1, 2, 'argmax']:
        for seed in [1, 2]:
            config['T'], config['seed'] = T, seed
            sample(**config)
