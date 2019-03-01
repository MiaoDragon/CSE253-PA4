#!/usr/bin/env python3
from dataloader import create_split_loaders
from model import GenericRNN
import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy
import os
from utility import *

config = {'chunk_size':100, 'type_number':93, 'hidden':100, 
          'learning_rate':0.001, 'early_stop':True, 'patience_threshold':10, 
          'epoch_num':10, 'N':50, 'M':500, 'seed':1, 'model':'LSTM',
          'model_path':'model_weights', 'T':1}

def sample(seed=None, chunk_size=None, type_number=None, hidden=None, learning_rate=None,
        early_stop=None, patience_threshold=None, epoch_num=None, model_path=None, N=None,
         M=None, model=None, T=1, **kwargs):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 1, "pin_memory": True}
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
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

    with torch.no_grad():
        out_str, h = '<', None
        for ii in range(chunk_size):
            out, h = net.forward(
                c_to(out_str[-1]).view(1, -1).to(computing_device), h)
            p = out.cpu()
            p /= T # Apply temperature
            p = F.softmax(p, dim=1)
            p = np.array(p).flatten()
            p /= sum(p)
            sampled_out = np.random.choice(c_list, p=p)
            out_str += sampled_out
        print(out_str)

if __name__ == '__main__':
    sample(**config)
