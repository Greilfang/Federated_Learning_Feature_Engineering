import os
import pandas as pd
import random
from sklearn.utils import shuffle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import torch

from kits.transformations import Binaries, Unaries

useful_tag = 1

useless_tag = 0


def read_convert_csvs(path):
    datasets = []
    csvs = os.listdir(path)
    print(csvs)
    for csv in csvs:
        csv_frame = pd.read_csv(path + '/{}'.format(csv))
        csv_frame = shuffle(csv_frame).sample(frac=0.93)
        dataset = {
            'name': csv[:-4],
            'data': csv_frame.values[:, :-1],
            'target': csv_frame.values[:, -1].astype("int"),
            'property': csv_frame.columns.values
        }
        print(dataset['name'])
        print(dataset['data'].shape)
        print('------------------------------------')
        datasets.append(dataset)
    return datasets


def read_domain_csv(path, type):
    dataset = None
    csv_frame = pd.read_csv(path)
    if type == "tensor":
        dataset = {
            'name': path[:-4],
            'data': torch.from_numpy(csv_frame.values[:, :-1].astype("float")).float(),
            'target': torch.from_numpy(csv_frame.values[:, -1].astype("int")).long(),
            'property': list(csv_frame.columns.values)[:-1]
        }
    elif type == "array":
        dataset = {
            'name': path[:-4],
            'data': csv_frame.values[:, :-1],
            'target': csv_frame.values[:, -1].astype("int"),
            'property': csv_frame.columns.values[:-1]
        }
    return dataset


def get_random_feature(n_feature, n_top):
    col_rand_ind = np.arange(n_feature)
    np.random.shuffle(col_rand_ind)
    col_rand_ind = col_rand_ind[:n_top]
    return col_rand_ind


class QuantileSketchDataset(Dataset):
    def __init__(self, json_path=None, transform=None):
        # 把数据集读取出来
        if json_path is not None:
            with open(json_path, 'r') as f:
                self.dataset = json.load(f)
        else:
            self.dataset = None

#
# if __name__ == "__main__":
#     read_convert_csvs("../raw")
