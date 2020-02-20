import pandas as pd
import numpy as np
import syft as sy
import torch
import torch.utils.data
import os
from role.server_client import ParameterServer, Client
from kits.transformations import Unaries, Binaries
from kits.dataset import read_convert_csvs, QuantileSketchDataset
from torch.utils.data import DataLoader, TensorDataset


class Params:
    def __init__(self):
        self.attempts = 5000
        self.epochs = 10
        self.cli_num = 5
        self.norm_bound = 10
        self.batch_size = 32
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Totally use ", torch.cuda.device_count(), "GPUs")
            self.server.net = torch.nn.DataParallel(self.server.net)

        # 读取数据集
        json_path = "demo"
        quantile_sets = QuantileSketchDataset(json_path)
        for name in Unaries.name:
            self.server.nets[name].to(device)
            quantile_set = quantile_sets[name]
            tensor_quantile_set = TensorDataset(quantile_set['data'], quantile_set['target'])
            loader = DataLoader(
                dataset=quantile_set,
                batch_size=self.params.batch_size,
                shuffle=True,
                num_workers=2
            )
            optimizer = torch.optim.SGD(self.server.nets[name].parameters(), lr=0.01, weight_decay=0.001)
            loss_func = torch.nn.CrossEntropyLoss().cuda()

            for epoch in range(self.params.epochs):
                for step, (batch_x, batch_y) in enumerate(loader):
                    pred_y = self.server.nets[name](batch_x)
                    loss = loss_func(pred_y, batch_y)
                    print("step {}".format(step), "loss {}".format(loss))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

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