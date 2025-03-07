import numpy as np
from dataloader import create_split_loaders
from model import GenericRNN
import torch

import torch.nn as nn
import copy
import os
from utility import *

config = {'chunk_size':100, 'type_number':93, 'hidden':100,
          'learning_rate':1e-3, 'weight_decay':0, 'early_stop':True,
          'patience_threshold':10, 'epoch_num':0, 'N':1000, 'M':1000,
          'seed':1, 'model':'LSTM', 'model_path':'model_weights',
          'num_workers': 4, 'pin_memory': True}

def test(seed=None, chunk_size=None, type_number=None, hidden=None, learning_rate=None,
        early_stop=None, patience_threshold=None, epoch_num=None, model_path=None, N=None,
        M=None, model=None, num_workers=8, pin_memory=True, **kwargs):
    """test the model.
    """
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": num_workers, "pin_memory": pin_memory}
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")

    torch.manual_seed(seed)

    # receive train, validation, test data
    train, valid, test, c_to, one_to = create_split_loaders(chunk_size,extras)
    # if model already exists, then load the previous one
    net = GenericRNN(type_number, hidden, type_number, model)
    if os.path.exists(model_path+'.pkl'):
        print('loading previous model...')
        load_net_state(net, model_path+'.pkl')
    net = net.to(computing_device)

    # use cross entropy loss
    def one_hot_CE(pred, target):
        # convert target from one hot to LongTensor, then apply CE
        target = target.argmax(dim=1)
        return nn.CrossEntropyLoss()(pred, target)
    criterion = one_hot_CE
    # if model already exists, then load the previous optimizer state
    prev_total_loss, prev_avg_minibatch_loss, prev_val_loss = [], [], []
    if os.path.exists(model_path+'.pkl'):
        print('loading optimizer state...')
        # notice when saving prev_val_loss, we ignored the first val_loss
        prev_total_loss, prev_avg_minibatch_loss, prev_val_loss = load_loss(model_path+'.pkl')

    total_loss = [] + prev_total_loss
    avg_minibatch_loss = [] + prev_avg_minibatch_loss
    test_loss = [float('inf')] + prev_val_loss #assume large error at the begining
    with torch.no_grad():
        loss_test = 0
        count_val = 0
        val_state_0 = None
        for val in test:
            count_val += 1
            if val[0].size()[0] != chunk_size:
                break
            valid_batch = val[0].to(computing_device)
            valid_target = val[1].to(computing_device)
            valid_predict, val_state_0 = net(valid_batch, val_state_0)
            loss_test += criterion(valid_predict, valid_target)
        loss_test /= count_val
        #test_loss.append(-np.log(loss_test.item()))
        test_loss.append(loss_test.item())
        print('test loss: %f' % (test_loss[-1]))

if __name__ == '__main__':
    test(**config)
