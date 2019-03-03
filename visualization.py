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

import math
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

config = {'chunk_size':100, 'type_number':93, 'hidden':100, 
          'learning_rate':0.001, 'early_stop':True, 'patience_threshold':10, 
          'epoch_num':10, 'N':50, 'M':500, 'seed':1, 'model':'LSTM',
          'model_path':'model_weights', 'T':1}

def visualize(seed=None, chunk_size=None, type_number=None, hidden=None, learning_rate=None,
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

    data = ''
    with open('pa4Data/generated_music0.txt', 'r') as file:
        for line in file:
            for char in line:
                data += char

    activations = []
    with torch.no_grad():
        in_char, h = '<', None
        for ii in range(len(data)):
            activation, h = net.rnn(c_to(data[ii]).view(1, 1, -1).to(computing_device), h)
            activations.append(activation.cpu().numpy().reshape(-1))

    activations = np.array(activations)
    print(activations.shape)

    for hidden_unit_index in range(100):
        #hidden_unit_index = 0
        act = activations[:,hidden_unit_index]
        x_len = int(math.sqrt(len(act)))
        y_len = int(len(act) / x_len )
        if y_len * x_len < len(act):
            y_len += 1

        act = np.pad(act, (0, y_len * x_len - len(act)), 'constant', constant_values=0)
        act = act.reshape(x_len, y_len)
        print(act.shape)
        print(act)
        for i in range(y_len * x_len - len(act)):
            data += ' '

        fig, ax = plt.subplots()
        im = ax.imshow(act, cmap='coolwarm')
        # Loop over data dimensions and create text annotations.
        for i in range(x_len):
            for j in range(y_len):
                text = ax.text(j, i, data[i * y_len + j], ha="center", va="center", color="w")

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, cmap="coolwarm")
        cbar.ax.set_ylabel('activation', rotation=-90, va="bottom")

        fig.tight_layout()
        plt.savefig('plots/heatmap_' + str(hidden_unit_index) + '.png')


if __name__ == '__main__':
    visualize(**config)
