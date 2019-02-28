#!/usr/bin/env python3
from dataloader import create_split_loaders
from model import GenericRNN
import torch

import torch.nn as nn
import copy
import os
from utility import *

config = {'chunk_size':100, 'type_number':93, 'hidden':100,
          'learning_rate':0.001, 'early_stop':True, 'patience_threshold':10,
          'epoch_num':10, 'N':50, 'M':500, 'seed':1, 'model':'LSTM',
          'model_path':'model_weights'}

def train(seed=None, chunk_size=None, type_number=None, hidden=None, learning_rate=None,
        early_stop=None, patience_threshold=None, epoch_num=None, model_path=None, N=None,
        M=None, model=None, **kwargs):
    """Train a model.
    """
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
    criterion = nn.BCELoss()

    # Using Adam
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
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

    best_net = -1
    # store best loss
    best_loss = float('inf')
    # number of consecutive epochs validation loss has increased - resets
    # whenever the loss decreases again
    stop_counter = 0
    epoch_cnt = 0
    for epoch in range(epoch_num):

        count = 0
        average_loss = 0
        state_0 = None
        for minibatch in train:
            count += 1
            if minibatch[0].size()[0] != chunk_size:
                break
            optimizer.zero_grad()
            train_batch = minibatch[0]
            target_batch = minibatch[1]
            train_batch = train_batch.to(computing_device)
            target_batch = target_batch.to(computing_device)
            predict_batch, state_0 = net(train_batch, state_0)
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
            if count % N == 0:
                training_record.append(average_loss / N)
                avg_minibatch_loss.append(average_loss / N)
                average_loss = 0
                print('training error after %d chunks:' % (count))
                print(training_record)
            # validation
            if count % M == 0:
                with torch.no_grad():
                    loss_val = 0
                    count_val = 0
                    state_0 = None
                    for val in valid:
                        count_val += 1
                        if val[0].size()[0] != chunk_size:
                            break
                        valid_batch = val[0].to(computing_device)
                        valid_target = val[1].to(computing_device)
                        valid_predict, state_0 = net(valid_batch, state_0)
                        loss_val += criterion(valid_predict, valid_target)
                    loss_val /= count_val
                    val_loss.append(loss_val.item())
                    validation_record.append(loss_val.item())
                    print('validation error after %d chunks:' % (count))
                    print(validation_record)
                    if loss_val < best_loss:
                        print('best model is updated')
                        best_loss = loss_val
                        best_net = copy.deepcopy(net)
                save_state(best_net, optimizer, total_loss, avg_minibatch_loss, val_loss[1:], \
                           seed, config['model_path']+'.pkl')
                if early_stop:
                    if loss_val > last_valid:
                        increasement += 1
                    else:
                        increasement = 0
                    last_valid = loss_val
                    if increasement >= increase_limit:
                        break
        if early_stop:
            if increasement >= increase_limit:
                break

if __name__ == '__main__':
    train(**config)
