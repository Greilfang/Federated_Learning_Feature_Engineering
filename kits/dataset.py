import os
import pandas as pd
import random
from sklearn.utils import shuffle
import numpy as np


def read_convert_csvs(path):
    datasets = []
    csvs = os.listdir(path)
    print(csvs)
    for csv in csvs:
        csv_frame = pd.read_csv(path+'/{}'.format(csv))
        csv_frame = shuffle(csv_frame)
        dataset = {
            'name':csv[:-4],
            'data':csv_frame.values[:,:-1],
            'target':csv_frame.values[:,-1],
            'property':csv_frame.columns.values
        }
        print(dataset['name'])
        print(dataset['data'].shape)
        print('------------------------------------')
        datasets.append(dataset)
    return datasets

#
# if __name__ == "__main__":
#     read_convert_csvs("../raw")