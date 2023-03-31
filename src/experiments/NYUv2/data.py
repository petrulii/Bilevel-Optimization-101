import fnmatch
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pprint import pprint
from datasets import load_dataset


class NYUv2(Dataset):
    """
    Code from:  https://github.com/lorenmt/mtan/blob/master/im2im_pred/create_dataset.py
                and https://github.com/AvivNavon/AuxiLearn/blob/master/experiments/nyuv2/data.py
    The (pre-processed) data is available here: https://www.dropbox.com/s/p2nn02wijg7peiy/nyuv2.zip?dl=0
    """
    def __init__(self, root, train=True):
        self.train = train
        self.root = os.path.expanduser(root)

        # read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

    def __getitem__(self, index):
        # get image name from the pandas df
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0))
        return image.float(), semantic.float(), depth.float(), normal.float()

    def __len__(self):
        return self.data_len

def nyu_dataloaders(datapath, validation_indices=None):
    """
    NYU dataloaders
        param datapath: path to the dataset directory
        param validation_indices: indices that form the validation set
    """
    ds = load_dataset(datapath)
    pprint(vars(ds))
    exit(0)
    nyuv2_train_set = NYUv2(root=datapath, train=True)
    nyuv2_test_set = NYUv2(root=datapath, train=False)
    train_set_numpy = nyuv2_train_set.data.numpy()
    test_set_numpy = nyuv2_test_set.data.numpy()
    return train_set_numpy, test_set_numpy
