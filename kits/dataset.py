import os
import pandas as pd
import random
from sklearn.utils import shuffle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json

from kits.transformations import Binaries, Unaries

useful_tag = np.array([1])

useless_tag = np.array([0])


def read_convert_csvs(path):
    datasets = []
    csvs = os.listdir(path)
    print(csvs)
    for csv in csvs:
        csv_frame = pd.read_csv(path + '/{}'.format(csv))
        csv_frame = shuffle(csv_frame)
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



class QuantileSketchDataset(Dataset):
    def __init__(self, json_path = None, transform=None):
        # 把数据集读取出来
        if json_path is not None:
            with open(json_path, 'r') as f:
                self.dataset = json.load(f)
        else:
            self.dataset = None

#
# if __name__ == "__main__":
#     read_convert_csvs("../raw")