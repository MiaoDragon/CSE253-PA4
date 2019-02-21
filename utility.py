import torch
import torch.nn as nn
import numpy as np
def zscore(img):
    # author: Yinglong Miao
    # can be used as a transform lambda function
    # input: tensor
    # subtract mean, then divide by std
    return (img - img.mean()) / img.std()

def save_state(net, opt, train_loss, avg_minibatch_loss, val_loss, seed, fname):
    # save model state, optimizer state, train_loss, val_loss, random_seed
    states = {
        'state_dict': net.state_dict(),
        'optimizer': opt.state_dict(),
        'train_loss': train_loss,
        'avg_minibatch_loss': avg_minibatch_loss,
        'val_loss': val_loss,
        'seed': seed
    }
    torch.save(states, fname)

def load_net_state(net, fname):
    checkpoint = torch.load(fname)
    net.load_state_dict(checkpoint['state_dict'])

def load_opt_state(opt, fname):
    checkpoint = torch.load(fname)
    opt.load_state_dict(checkpoint['optimizer'])

def load_loss(fname):
    checkpoint = torch.load(fname)
    return checkpoint['train_loss'], checkpoint['avg_minibatch_loss'], checkpoint['val_loss']

def load_seed(fname):
    checkpoint = torch.load(fname)
    return checkpoint['seed']

def confMtrxUpdate(pred, label, confMtrx):
    predClass = np.where(pred==1)[0]##what classes you've predicted
    labelClass = np.where(label==1)[0]##what the true classes are
    if len(predClass)==0 and len(labelClass)==0:
        confMtrx[14][14] +=1
        return confMtrx
    else:
        ##work on the common ones
        commonList = np.intersect1d(predClass, labelClass)
        for c in commonList:
            confMtrx[c][c] += 1
        ##eliminate the common one, work on the uncommon ones
        predUnq = [i for i in predClass if i not in commonList]
        labelUnq = [i for i in labelClass if i not in commonList]
        if len(predUnq) == 0 and len(labelUnq) == 0:
            return confMtrx ##predictions and true values match
        if len(predUnq) == 0:
            predUnq.append(14)
        if len(labelUnq) == 0:
            labelUnq.append(14)
        for i in predUnq:
            for j in labelUnq:
                confMtrx[i][j]+=1
    return confMtrx
