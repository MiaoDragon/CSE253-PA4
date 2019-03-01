#!/usr/bin/env python3
from dataloader import create_split_loaders
from model import GenericRNN
import torch

import torch.nn as nn
import copy
import os
from utility import *

config = {'chunk_size':100, 'type_number':93, 'hidden':100,
          'learning_rate':1e-3, 'weight_decay':0, 'early_stop':True,
          'patience_threshold':10, 'epoch_num':10, 'N':100, 'M':1000,
          'seed':1, 'model':'LSTM', 'model_path':'model_weights',
          'num_workers': 0, 'pin_memory': True}

def train(seed=None, chunk_size=None, type_number=None, hidden=None, learning_rate=None,
        early_stop=None, patience_threshold=None, epoch_num=None, model_path=None, N=None,
        M=None, model=None, num_workers=None, pin_memory=None, weight_decay=None,
        **kwargs):
    """Train a model.
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

    # construct network
    net = GenericRNN(type_number, hidden, type_number, model)
    # if model already exists, then load the previous one
    if os.path.exists(model_path+'.pkl'):
        print('loading previous model...')
        load_net_state(net, model_path+'.pkl')
    net = net.to(computing_device)

    # use cross entropy loss
    def one_hot_CE(pred, target):
        # convert target from one hot to LongTensor, then apply CE
        # print(one_to(target))
        target = target.argmax(dim=1)
        # print(target.tolist())
        # print(pred.shape, target.shape)
        return nn.CrossEntropyLoss()(pred, target)
    criterion = one_hot_CE
    # Using Adam
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # if model already exists, then load the previous optimizer state
    prev_total_loss, prev_avg_minibatch_loss, prev_val_loss = [], [], []
    if os.path.exists(model_path+'.pkl'):
        print('loading optimizer state...')
        load_opt_state(optimizer, model_path+'.pkl')
        # notice when saving prev_val_loss, we ignored the first val_loss
        prev_total_loss, prev_avg_minibatch_loss, prev_val_loss = load_loss(model_path+'.pkl')

    total_loss = [] + prev_total_loss
    avg_minibatch_loss = [] + prev_avg_minibatch_loss
    val_loss = [float('inf')] + prev_val_loss #assume large error at the begining
    last_valid = float('inf')  # for early stopping
    best_net = -1
    # store best loss
    best_loss = float('inf')
    # number of consecutive epochs validation loss has increased - resets
    # whenever the loss decreases again
    stop_counter = 0
    epoch_cnt = 0
    for epoch in range(epoch_num):
        average_loss = 0
        state_0 = None
        for minibatch_ind, minibatch in enumerate(train, 1):
            if minibatch[0].size()[0] != chunk_size:
                break
            optimizer.zero_grad()
            train_batch = minibatch[0]
            target_batch = minibatch[1]
            train_batch = train_batch.to(computing_device)
            target_batch = target_batch.to(computing_device)
            predict_batch, state_0 = net(train_batch, state_0)

            # debug code
            debug = False
            if minibatch_ind % N == 0 and debug:
                print('=== INPUT ===========================================')
                print(one_to(train_batch) +'<FAKE_END>')
                print('=== TARGET ==========================================')
                print(one_to(target_batch) +'<FAKE_END>')
                print('=== OUTPUT ==========================================')
                # print((predict_batch*10).round().type(torch.int).tolist())
                print(one_to(predict_batch) +'<FAKE_END>')
                print('=== LOSS ============================================')
                print(loss.item())
                print('\n')

            if isinstance(state_0, tuple):
                state_0 = list(state_0)
                for i in range(len(state_0)):
                    state_0[i] = state_0[i].detach()
                state_0 = tuple(state_0)
            else:
                state_0 = state_0.detach()
            loss = criterion(predict_batch, target_batch)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            average_loss += loss.item()
            if minibatch_ind % N == 0:
                avg_minibatch_loss.append(average_loss / N)
                average_loss = 0
                print('epoch %d minibatch %d average train loss: %f' % (epoch, minibatch_ind, avg_minibatch_loss[-1]))
        # validation
        with torch.no_grad():
            loss_val = 0
            count_val = 0
            val_state_0 = None
            for val in valid:
                count_val += 1
                if val[0].size()[0] != chunk_size:
                    break
                valid_batch = val[0].to(computing_device)
                valid_target = val[1].to(computing_device)
                valid_predict, val_state_0 = net(valid_batch, val_state_0)
                loss_val += criterion(valid_predict, valid_target)
            loss_val /= count_val
            val_loss.append(loss_val.item())
            print('epoch %d minibatch %d val loss: %f' % (epoch, minibatch_ind, val_loss[-1]))
            if loss_val < best_loss:
                print('best model is updated')
                best_loss = loss_val
                best_net = copy.deepcopy(net)
        save_state(best_net, optimizer, total_loss, avg_minibatch_loss, val_loss[1:], \
                   seed, config['model_path']+'.pkl')
        if early_stop:
            if loss_val > val_loss[-1]:
                stop_counter += 1
            else:
                stop_counter = 0
            if stop_counter >= patience_threshold:
                break

if __name__ == '__main__':
    train(**config)
