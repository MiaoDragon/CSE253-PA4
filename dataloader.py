# dataloader for training, validation, and testing set
# from training dataset, construct an one-hot encoding dictionary for each character
# provide a function to achieve this
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter


class MusicDataset(Dataset):
    def __init__(self, filename, c_to_one_hot, one_hot_to_c):
        """
        # Input:
        - filename: path to load data
        - c_list:   unique character list (None if using the one obtained from this)
        - one_hot_dict: dictionary for one-hot encoding (None if using the one obtained
                                                         from this data)
        # keep raw data, unique character set, dictionary from character to indices
        # and number of characters in each dataset
        """
        data = []
        with open(filename, 'r') as file:
            for line in file:
                data += line
        self.data = data
        self.c_to_one_hot = c_to_one_hot
        self.one_hot_to_c = one_hot_to_c
    def __len__(self):
        # return the length of each dataset - 1
        # minus one as the last data has no target value
        return len(self.data) - 1
    def __getitem__(self, ind):
        # return the one-hot encoding of the data at ind
        # if ind is at the end of training set, testing set or val set, then just
        # use None as label
        # convert data to one_hot
        x = self.c_to_one_hot(self.data[ind])
        y = self.c_to_one_hot(self.data[ind+1])
        return x, y

def obtain_map():
    """
    obtain unique character list, ind_dict that maps characters to indices
    from training dataset
    """
    data = []
    with open('pa4Data/train.txt', 'r') as file:
        for line in file:
            data += line

    # debug
    # ctr = Counter(data[:30000])
    # print(ctr)
    # print(ctr['#'])
    # print(ctr['&'])
    # print(ctr["'"])
    # print(len(ctr))

    c_list = list(set(data))
    c_list.sort()
    # print(c_list)
    ind_dict = {}
    for i in range(len(c_list)):
        ind_dict[c_list[i]] = i
    def c_to_one_hot(c):
        one_hot = torch.zeros(len(c_list))
        one_hot[ind_dict[c]] = 1.
        return one_hot
    def one_hot_to_c(one_hot):
        """
            input is tensor
            one_hot to character
            Note that since we are using max function here, the input can also be
            logistics (probability)
        """
        _, indices = one_hot.max(1)
        # print(one_hot.shape)
        # print(indices.shape)
        s = ''
        for ind in indices:
            # print(ind.item())
            s += c_list[ind.item()]
        return s#''.join(c_list[ind] for ind in indices)
    return c_to_one_hot, one_hot_to_c

def create_split_loaders(chunk_size, extras={}):
    # obtain maps that map from characters to one-hot-encoding, and backward
    c_to_one_hot, one_hot_to_c = obtain_map()
    # create dataset object for train, val, test dataset
    train_dataset = MusicDataset(filename='pa4Data/train.txt', \
                                 c_to_one_hot=c_to_one_hot, one_hot_to_c=one_hot_to_c)
    val_dataset = MusicDataset(filename='pa4Data/val.txt', \
                                 c_to_one_hot=c_to_one_hot, one_hot_to_c=one_hot_to_c)
    test_dataset = MusicDataset(filename='pa4Data/test.txt', \
                                 c_to_one_hot=c_to_one_hot, one_hot_to_c=one_hot_to_c)
    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    # Define the training, test, & validation DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=chunk_size,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=chunk_size,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=chunk_size,
                             num_workers=num_workers, pin_memory=pin_memory)

    # Return the training, validation, test, and mapping functions
    return (train_loader, val_loader, test_loader, c_to_one_hot, one_hot_to_c)
