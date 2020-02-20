import pandas as pd
import numpy as np
import syft as sy
import torch
import os
from role.server_client import ParameterServer, Client
from kits.transformations import Unaries, Binaries
from kits.dataset import read_convert_csvs


class Params:
    def __init__(self):
        self.attempts = 5000
        self.epochs = 50
        self.cli_num = 5
        self.norm_bound = 10
        self.no_cuda = False


class ClientParams:
    def __init__(self):
        self.threshold = 0.01
        self.n_trees = 10
        self.n_bins = 400


class NetParams:
    def __init__(self):
        self.size = 400
        self.n_cls = 2
        self.n_cores = 256


hook = sy.TorchHook(torch)


class FederatedLFE:
    def __init__(self):
        self.params = Params()
        self.server = ParameterServer(NetParams())
        self.clients = list()
        for i in range(self.params.cli_num):
            self.clients.append(Client(i, ClientParams()))

    def generate_qsa(self, path):
        datasets = read_convert_csvs(path)
        index = 0
        # 分发数据集到各个节点
        while not len(datasets) == 0:
            dataset = datasets.pop()
            self.clients[index].get_dataset(dataset)
            index = (index + 1) % self.params.cli_num

        for client in self.clients:
            client.generate_qsa()

    def train_fedavg_mlp(self):
        # 联邦训练mlp
        pass

    def learn_feature_engineering(self):
        pass

    def check_improvement(self):
        pass


if __name__ == "__main__":
    bob = sy.VirtualWorker(hook, id="bob")
    x = torch.tensor([1, 2, 3, 4, 5])
    y = torch.tensor([1, 1, 1, 1, 1])
    x_ptr = x.send(bob)
    y_ptr = y.send(bob)
    z_ptr = x_ptr + x_ptr
    z = z_ptr.get()
    print(z)