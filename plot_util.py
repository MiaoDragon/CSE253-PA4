import time
import seaborn as sn
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

import os

def _convert_to_xy(data):
    if isinstance(data, dict):
        x,y = zip(*(sorted(data.items())))
    elif isinstance(data, list):
        x = np.arange(n)+1
        y = data
    elif isinstance(data, np.ndarray):
        x = np.arange(n)+1
        y = list(data) # Assume 1D
    return (x,y)

# Data params should be lists or 1D arrays or dicts. `name` is a string
# identifier for the output figure PNG; if not provided, it will default to
# using the current datetime.
def plot(train_losses, val_losses, name=None):
    fig = plt.figure(figsize=(8,4), dpi=80)
    ax = fig.add_subplot(111)
    title = 'loss over epochs'
    xlabel, ylabel = 'epochs', 'loss'
    nbins = 10

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    x,y = _convert_to_xy(train_losses)
    plt.plot(x,y,label='Training')
    n = max(x)

    x,y = _convert_to_xy(val_losses)
    plt.plot(x,y,label='Validation')
    n = max(max(x),n)

    ticks = (np.arange(nbins) + 1) * n//nbins
    plt.xticks(ticks)

    ax.set_ylim(bottom=0)
    ax.margins(0)
    ax.legend()

    if name is None:
        name = time.strftime("%m-%d-%Y_%H-%M-%S")

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/run_'+name+'.png')

##plot the confusion matrix
##input is the confusion matrix and name of the task
classDict = {0: "Atelectasis", 1: "Cardiomegaly", 2: "Effusion",
                3: "Infiltration", 4: "Mass", 5: "Nodule", 6: "Pneumonia",
                7: "Pneumothorax", 8: "Consolidation", 9: "Edema",
                10: "Emphysema", 11: "Fibrosis",
                12: "Pleural_Thickening", 13: "Hernia", 14:"No disease present"}
def confMtrxPlot(confMtrx, name = None):
    df_cm = pd.DataFrame(confMtrx, index = [classDict[i] for i in range(0,15)],
                  columns = [classDict[i] for i in range(0,15)])
    plt.figure(figsize = (12,8))
    a = sn.heatmap(df_cm, annot=True)
    a.set(xlabel = "actual disease",ylabel = "predicted disease" )
    os.makedirs('plots', exist_ok=True)
    if name is None:
        plt.savefig('plots/confusionMatrix.png')
    else:
        plt.savefig('plots/confusionMatrix_'+name+'.png')
