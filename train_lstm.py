#!/usr/bin/env python
from dataloader import create_split_loaders
from model import GenericRNN
import torch

import torch.nn as nn
import copy
import os
from utility import *

config = {'chunk_size':100, 'type_number':93, 'hidden':100, 
          'learning_rate':0.001, 'early_stop':True, 'increase_limit':3, 
          'epoch_num':1, 'N':50, 'M':100, 'seed':1, 'model':'LSTM', 'model_path':'model_weights'}

def train(config):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 1, "pin_memory": True}
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")
    
    seed = config['seed']
    torch.manual_seed(seed)
    
    # the size of every chunk
    chunk_size = config['chunk_size']
    # number of types, it is a constant
    type_number = config['type_number']
    # number of features in hidden layer
    hidden = config['hidden']
    # learning rate
    learning_rate = config['learning_rate']
    # whether we use early stop
    early_stop = config['early_stop']
    # after validation loss increase how many times do we stop training
    increase_limit = config['increase_limit']
    # number of epoch
    epoch_num = config['epoch_num']
    # receive train, validation, test data
    train, valid, test, c_to, one_to = create_split_loaders(chunk_size,extras)
    
    # construct network
    net = GenericRNN(type_number, hidden, type_number, config['model'])
    # if model already exists, then load the previous one
    if os.path.exists(config['model_path']+'.pkl'):
        print('loading previous model...')
        load_net_state(net, config['model_path']+'.pkl')
    net = net.to(computing_device)
    
    # use cross entropy loss
    criterion = nn.BCELoss()
    # keep tracking of the traininig loss
    training_record = []
    # keep tracking of the validation loss
    validation_record = []
    
    # Using Adam
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # if model already exists, then load the previous optimizer state
    prev_total_loss, prev_avg_minibatch_loss, prev_val_loss = [], [], []
    if os.path.exists(config['model_path']+'.pkl'):
        print('loading optimizer state...')
        load_opt_state(optimizer, config['model_path']+'.pkl')
        # notice when saving prev_val_loss, we ignored the first val_loss
        prev_total_loss, prev_avg_minibatch_loss, prev_val_loss = load_loss(config['model_path']+'.pkl')
    
    total_loss = [] + prev_total_loss
    avg_minibatch_loss = [] + prev_avg_minibatch_loss
    val_loss = [1000000000] + prev_val_loss #assume large error at the begining

    last_valid = float('inf')
    best_net = -1
    # after how many batches do we record the training loss
    N = config['N']
    # after how many batches do we examine whether validation loss increase
    M = config['M']
    # store best loss
    best_loss = float('inf')
    increasement = 0
    epoch_cnt = 0
    for epoch in range(epoch_num):
        count = 0
        average_loss = 0
        for minibatch in train:
            count += 1
            if minibatch[0].size()[0] != chunk_size:
                break
            predict_all = torch.zeros(chunk_size, type_number)
            target_all = torch.zeros(chunk_size, type_number)
            optimizer.zero_grad()
            for ii in range(chunk_size):
                train_batch = torch.zeros(1, 1, type_number)
                train_batch[0] = minibatch[0][ii]
                train_batch = train_batch.to(computing_device)
                target = minibatch[1][ii]
                target = target.to(computing_device)
                if ii == 0:
                    predict = net.predict(train_batch)
                else:
                    # teacher forcing
                    teacher = torch.ones(1, 1, type_number)
                    teacher[0] = minibatch[1][ii - 1]
                    teacher = teacher.to(computing_device)
                    predict = net.predict(train_batch, teacher)
                predict_all[ii] = predict
                target_all[ii] = target
            # calculate loss
            loss = criterion(predict_all, target_all)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            average_loss += loss.item()
            if count % N == 0:
                training_record.append(average_loss / N)
                avg_minibatch_loss.append(average_loss / N)
                average_loss = 0
                print('epoch ' + str(epoch_cnt) + ' minibatch ' + str(count) + ' keep tracking of training error:')
                print(training_record[-1])
#             print(loss.item())
            # validation 
            if count % M == 0:
                with torch.no_grad():
                    loss_val = 0
                    count_val = 0
                    for val in valid:
                        count_val += 1
                        if val[0].size()[0] != chunk_size:
                            break
                        predict_valid = torch.zeros(chunk_size, type_number)
                        target_valid = torch.zeros(chunk_size, type_number)
                        for ii in range(chunk_size):
                            valid_batch = torch.zeros(1, 1, type_number)
                            valid_batch[0] = val[0][ii]
                            valid_batch = valid_batch.to(computing_device)
                            target = val[1][ii]
                            target = target.to(computing_device)
                            predict = net.predict(valid_batch)
                            predict_valid[ii] = predict
                            target_valid[ii] = target
                        loss_val += criterion(predict_valid, target_valid)
                    loss_val /= count_val
                    val_loss.append(loss_val.item())
                    validation_record.append(loss_val.item())
                    print('epoch ' + str(epoch_cnt) + ' minibatch ' + str(count) + ' keep tracking of validation error')
                    print(validation_record[-1])
                    if loss_val < best_loss:
                        print('best model is updated')
                        best_loss = loss_val
                        best_net = copy.deepcopy(net)
                save_state(best_net, optimizer, total_loss, avg_minibatch_loss, val_loss[1:], seed, config['model_path']+'.pkl')
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

        epoch_cnt += 1


train(config)

