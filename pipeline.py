import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, utils

from utility import *
from augment_dataloader import create_split_loaders
import plot_util

import random
import time
import os

# TODO: handle other models
import model1
import model2

def train(config):
    # config: dictionary containing field:
    #         model:  -- pytorch model class
    #         learning_rate
    #         num_epochs
    #         batch_size
    #         seed    -- random seed, int
    #         p_val   -- Percent of the overall dataset to reserve for validation
    #         p_test  -- Percent of the overall dataset to reserve for testing
    #         transforms  -- list of transforms (each one corresponds to one dataset)
    #         criterion  -- loss function (can be found in utility)
    #         early_stop -- early stop or not
    #         early_stop_threshold -- number of times validation loss increase to early stop
    #         train_loader, val_loader, test_loader
    #         model_path  -- where to save trained model, specifiy only the path and name
    #                        then epoch information will be appended at the last
    # Setup: initialize the hyperparameters/variables
    num_epochs = config['num_epochs']          # Number of full passes through the dataset
    batch_size = config['batch_size']         # Number of samples in each minibatch
    learning_rate = config['learning_rate']
    seed = config['seed']                # Seed the random number generator for reproducibility
    p_val = config['p_val']              # Percent of the overall dataset to reserve for validation
    p_test = config['p_test']             # Percent of the overall dataset to reserve for testing
    model_name = config['model_name']

    transforms = config['transforms']
    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        print("CUDA NOT supported")

    # Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = config['train_loader'], config['val_loader'], config['test_loader']

    # set random seed for pytorch, numpy and random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Instantiate a model to run on the GPU or CPU based on CUDA support
    model = config['model']
    # if model already exists, then load the previous one
    if os.path.exists(config['model_path']+'.pkl'):
        print('loading previous model...')
        load_net_state(model, config['model_path']+'.pkl')
    model = model.to(computing_device)
    print("Model on CUDA?", next(model.parameters()).is_cuda)

    # Define the loss criterion and instantiate the gradient descent optimizer
    criterion = config['criterion']


    optimizer = torch.optim.Adam(params=filter(lambda w: w.requires_grad, model.parameters()),\
                                 lr=learning_rate)

    # if model already exists, then load the previous optimizer state
    prev_total_loss, prev_avg_minibatch_loss, prev_val_loss = [], [], []
    if os.path.exists(config['model_path']+'.pkl'):
        print('loading optimizer state...')
        load_opt_state(optimizer, config['model_path']+'.pkl')
        # notice when saving prev_val_loss, we ignored the first val_loss
        prev_total_loss, prev_avg_minibatch_loss, prev_val_loss = load_loss(config['model_path']+'.pkl')
    # keep track of the current best model by validation loss
    optimalModel = model
    optimalLoss = 1000000000
    # Track the loss across training
    total_loss = [] + prev_total_loss
    avg_minibatch_loss = [] + prev_avg_minibatch_loss
    val_loss = [1000000000] + prev_val_loss #assume large error at the begining
    stop_ct = 0


    # interval params
    N = 50 # number of minibatches between avg loss output
    val_interval = 10*N # number of minibatches between val checkpts
    prev_minibatch_ind = (len(val_loss) - 1) * val_interval

    # Begin training procedure
    for epoch in range(num_epochs):

        N_minibatch_loss = 0.0
        start = time.time()

        # Get the next minibatch of images, labels for training
        for minibatch_count, (images, labels) in enumerate(train_loader, 1):
            if epoch == 0 and minibatch_count <= prev_minibatch_ind:
                continue
            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            images, labels = images.to(computing_device), labels.to(computing_device)

            # Zero out the stored gradient (buffer) from the previous iteration
            optimizer.zero_grad()

            # Perform the forward pass through the network and compute the loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Automagically compute the gradients and backpropagate the loss through the network
            loss.backward()

            # Update the weights
            optimizer.step()

            # Add this iteration's loss to the total_loss
            total_loss.append(loss.item())
            N_minibatch_loss += loss.item()

            if minibatch_count % N == 0:

                # Print the loss averaged over the last N mini-batches
                N_minibatch_loss /= N
                print('Epoch %d, average minibatch %d loss: %.3f' %
                    (epoch + 1, minibatch_count, N_minibatch_loss))

                # Add the averaged loss over N minibatches and reset the counter
                avg_minibatch_loss.append(N_minibatch_loss)
                N_minibatch_loss = 0.0
                print('minibatch %d time: %f' % (minibatch_count, time.time()-start))
                start = time.time()
            # validation check for every 500 minibatches
            if minibatch_count % val_interval == 0:
                print('-------validation-------')
                with torch.no_grad():
                    subLoss = 0.0
                    for minibatch_count_V, (images_V, labels_V) in enumerate(val_loader, 1):
                        images_V, labels_V = images_V.to(computing_device), labels_V.to(computing_device)
                        outputs_V = model(images_V)
                        loss_V = criterion(outputs_V, labels_V)
                        subLoss += loss_V.item()
                    subLoss = subLoss / minibatch_count_V  # averaged over number of minibatches
                    val_loss.append(subLoss)
                    if val_loss[-1] >= val_loss[-2]:
                        # current validation loss larger than previous one, increase the counter
                        stop_ct += 1
                    else:
                        # loss decreases, reset the counter to 0
                        stop_ct = 0
                    if val_loss[-1] < optimalLoss:
                        # remember the best model with the minimal val loss
                        optimalLoss = val_loss[-1]
                        optimalModel = model
                print('validation loss after %d epoches, %d minibatches: %f' % (epoch+1, minibatch_count, val_loss[-1]))
                # save weights after validation test
                print('saving weight...')
                save_state(optimalModel, optimizer, total_loss, avg_minibatch_loss, val_loss[1:], \
                           seed, config['model_path']+'.pkl')
                if config['early_stop'] and stop_ct >= config['early_stop_threshold']:
                    # early stop after passing the threshold
                    print("Finished training after ", epoch+1, " epoch updates.")
                    break
        print("Finished", epoch + 1, "epochs of training")

        if config['early_stop'] and stop_ct >= config['early_stop_threshold']:
            break

    print("Training complete after", epoch, "epochs")
    print("Average minibatches training loss: ", avg_minibatch_loss)
    print("Validation loss: ", val_loss[1:])
    # plot train loss and val loss
    plot_util.plot(avg_minibatch_loss, val_loss[1:], name=model_name)


def test(config):
    # config: dictionary containing field:
    #         model:  -- pytorch model class
    #         learning_rate
    #         num_epochs
    #         batch_size
    #         seed    -- random seed, int
    #         p_val   -- Percent of the overall dataset to reserve for validation
    #         p_test  -- Percent of the overall dataset to reserve for testing
    #         transforms  -- list of transforms (each one corresponds to one dataset)
    #         criterion  -- loss function (can be found in utility)
    #         train_loader, val_loader, test_loader
    #         model_path  -- where to save trained model, specifiy only the path and name
    #                        then epoch information will be appended at the last
    #         model_name  -- to use when plotting
    # Setup: initialize the hyperparameters/variables
    batch_size = config['batch_size']         # Number of samples in each minibatch
    learning_rate = config['learning_rate']
    seed = config['seed']                # Seed the random number generator for reproducibility
    p_val = config['p_val']              # Percent of the overall dataset to reserve for validation
    p_test = config['p_test']             # Percent of the overall dataset to reserve for testing

    transforms = config['transforms']
    model_name = config['model_name']
    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        print("CUDA NOT supported")

    # Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = config['train_loader'], config['val_loader'], config['test_loader']

    # set random seed for pytorch, numpy and random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Instantiate a BasicCNN to run on the GPU or CPU based on CUDA support
    model = config['model']
    load_net_state(model, config['model_path']+'.pkl')
    model = model.to(computing_device)
    print("Model on CUDA?", next(model.parameters()).is_cuda)

    #TODO: Define the loss criterion and instantiate the gradient descent optimizer
    confMtrx = [[ 0 for j in range(0,15)] for i in range(0,15)]
    ##class contingency matrix stores the TP, TN, FP, FN for each class in a row
    classContingencyMatrix = [[0 for t in range(0,4)] for r in range(0,14)]
    ##switch the model to the optimalModel
    model.eval()
    with torch.no_grad():
        for minibatch_count_T, (images_T, labels_T) in enumerate(test_loader, 0):
            images_T, labels_T= images_T.to(computing_device), labels_T.to(computing_device)
            outputs_T = model.predict(images_T)
            outputs_T = ((outputs_T>= 0.5)).float()
            for r in range(0,len(outputs_T)):
                confMtrx = confMtrxUpdate(outputs_T[r].cpu().numpy(), labels_T[r].cpu().numpy(), confMtrx)
                for c in range(0,14):
                    pred = (outputs_T[r][c]).item()
                    target = (labels_T[r][c]).item()
                    ##switch cases
                    ##TP
                    if (pred == 1.0 and target == 1.0):
                        classContingencyMatrix[c][0] += 1
                    ##TN
                    elif (pred == 0.0 and target == 0.0):
                        classContingencyMatrix[c][1] += 1
                    ##FP
                    elif (pred == 1.0 and target == 0.0):
                        classContingencyMatrix[c][2] += 1
                    ##FN
                    else:
                        classContingencyMatrix[c][3] += 1

    classDict = {0: "Atelectasis", 1: "Cardiomegaly", 2: "Effusion",
                    3: "Infiltration", 4: "Mass", 5: "Nodule", 6: "Pneumonia",
                    7: "Pneumothorax", 8: "Consolidation", 9: "Edema",
                    10: "Emphysema", 11: "Fibrosis",
                    12: "Pleural_Thickening", 13: "Hernia"}
    print("---------------------------------")
    ##Evaluate i)-iv)
    ##v) will be done when creating a table for the report
    for c in range(0,14):
        result = classContingencyMatrix[c]
        ##print class name
        print("Class: %s"% classDict[c])
        ##calculate the accuracy
        denom = sum(result)
        if denom != 0:
            s_accuracy = (result[0] + result[1])/denom
        else:
            s_accuracy  = 0
        print("\t Accuracy: %f" % s_accuracy )
        ##calculate the precision
        denom = result[2]+result[0]
        if denom != 0:
            s_precision = (result[0])/denom
        else:
            s_precision = 0
        print("\t Precision: %f" % s_precision)
        ##calculate the recall
        denom = result[0]+result[3]
        if denom != 0:
            s_recall = (result[0])/denom
        else:
            s_recall = 0
        print("\t Recall: %f" % s_recall)
        ##calculate BCR
        BCR = (s_precision+s_recall)/2.0
        print("\t BCB: %f" % BCR)
        print("\n")

    print("---------------------------------")
    ##normalize the confusion matrix row-wise
    # print(confMtrx)
    confMtrxNorm = []
    for row in range(0,15):
        subSum = sum(confMtrx[row])
        if subSum != 0:
            confMtrxNorm.append([e/subSum for e in confMtrx[row]])
        else:
            confMtrxNorm.append(confMtrx[row])
    print(confMtrxNorm)
    plot_util.confMtrxPlot(confMtrxNorm, name=model_name)

    print("---------------------------------")
    plot(config)

def plot(config):
    print('loading loss data')
    if os.path.exists(config['model_path']+'.pkl'):
        _, prev_avg_minibatch_loss, prev_val_loss = \
            load_loss(config['model_path']+'.pkl')
    # plot train loss and val loss
    def to_dict(a_list, interval):
        return {interval*(i+1) : e for i,e in enumerate(a_list)}
    plot_util.plot(to_dict(prev_avg_minibatch_loss, 50),
                   to_dict(prev_val_loss, 500), name=config['model_name'])
    print('loss plot saved')


def main(test_mode=False, plot_mode=False, use_weighted=False, use_dropout=False,
       model_base_name='model1', model=None, in_size=526, color='L',
       dropout_scale=1., learning_rate=1e-2):
    config = {}
    config['use_dropout'] = use_dropout
    config['use_weighted'] = use_weighted

    # TODO: handle other models
    # if model_base_name != 'model1':
    #     raise ValueError('Unrecognized model, "{}"'.format(model_base_name))

    # NOTE: for now, we just assume that, if model_base_name != 'model1', then
    # the model is provided as a param (and if model_base_name is not changed,
    # we ignore the model param)
    if model_base_name == 'model1':
        model = model1.Net(in_size=in_size, num_class=14,
                           use_dropout=config['use_dropout'],dropout_scale=dropout_scale)
    elif model_base_name == 'model2':
        model = model2.Model2(model2.make_layers(model2.cfg['G'], batch_norm=True), num_classes=14, use_dropout=config['use_dropout'])
    config['model'] = model
    model_name = model_base_name
    if config['use_weighted']:
        model_name += '_weighted'
    if config['use_dropout']:
        model_name += '_dropout'
        if dropout_scale != 1.:
            model_name += '_%f' % (dropout_scale)
    config['model_name'] = model_name

    os.makedirs('model_weights', exist_ok=True)
    config['model_path'] = 'model_weights/' + model_name

    config['learning_rate'] = learning_rate
    config['num_epochs'] = 20
    config['batch_size'] = 16
    config['p_val'] = 0.1
    config['p_test'] = 0.2
    config['transforms'] = [transforms.Compose([transforms.Resize(in_size), transforms.ToTensor(), \
                                                transforms.Lambda(zscore)]),
                            transforms.Compose([transforms.Resize(in_size), \
                                                transforms.RandomAffine(degrees=5, translate=None, \
                                                scale=None, shear=None, resample=False, fillcolor=0),\
                                                transforms.ToTensor(), \
                                                transforms.Lambda(zscore)]),
                            transforms.Compose([transforms.Resize(in_size), \
                                                transforms.RandomAffine(degrees=0, translate=None, \
                                                scale=(0.9,1.1), shear=None, resample=False, fillcolor=0),\
                                                transforms.ToTensor(), \
                                                transforms.Lambda(zscore)])
                            ]
    #config['seed'] = np.random.randint(low=0,high=1000)
    config['seed'] = 50
    # if model_path already exists, set seed according to previous seed
    # to make sure the same split is applied as training
    if os.path.exists(config['model_path']+'.pkl'):
        print('loading previous seed...')
        config['seed'] = load_seed(config['model_path']+'.pkl')

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 16, "pin_memory": True}
        #config['model'] = torch.nn.DataParallel(config['model'])
        #print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        #print("CUDA NOT supported")



    batch_size, p_val, p_test, transforms_, seed = \
        config['batch_size'], config['p_val'], \
        config['p_test'], config['transforms'], config['seed']

    config['train_loader'],config['val_loader'],config['test_loader'],pos_weight = \
        create_split_loaders(batch_size, seed, transforms=transforms_,
                             p_val=p_val, p_test=p_test,
                             shuffle=True, show_sample=False,
                             extras=extras,
                             color=color)
    if use_cuda:
        pos_weight = pos_weight.cuda(computing_device)


    unweighted_loss = nn.BCEWithLogitsLoss()
    weighted_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    config['criterion'] = weighted_loss if config['use_weighted'] else unweighted_loss

    config['early_stop'] = True
    config['early_stop_threshold'] = 10

    function = train
    if test_mode:
        function = test
    if plot_mode:
        function = plot
    function(config)

if __name__ == "__main__":
    # TODO: parse commandline args
    main()
