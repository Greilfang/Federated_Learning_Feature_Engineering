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
    data_center = FederatedLFE()
    data_center.generate_qsa("./raw")
    # bob = sy.VirtualWorker(hook, id="bob")import pandas as pd
import numpy as np
import syft as sy
import torch
import torch.utils.data
import os
from role.server_client import ParameterServer, Client,get_sketch
from kits.transformations import Unaries, Binaries
from random import choice
from kits.dataset import read_convert_csvs, QuantileSketchDataset
from torch.utils.data import DataLoader, TensorDataset


class Params:
    def __init__(self):
        self.attempts = 5000
        self.epochs = 400
        self.cli_num = 5
        self.norm_bound = 10
        self.batch_size = 256
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


# hook = sy.TorchHook(torch)


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
            client.load_qsa_set()
            # 生成数据集
            client.generate_qsa()
            # 保存数据集
            client.save_qsa_set()

    def train_fedavg_mlp(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Totally use ", torch.cuda.device_count(), "GPUs")
            self.server.net = torch.nn.DataParallel(self.server.net)

        # 读取数据集
        for client in self.clients[4:]:
            client.load_qsa_set()
        # 把数据集送到每个节点
        for name in (Unaries.name + Binaries.name):
            print('MLP:{}'.format(name))
            print('-' * 40)
            fed_data_dict, fed_target_dict = dict(), dict()
            optimizers = dict()
            datasets = list()
            for client in self.clients[4:]:
                quantile_set = {
                    'data': torch.tensor(client.qsa_set[name]['data']).float(),
                    'target': torch.tensor(client.qsa_set[name]['target']).long()
                }
                print("{}".format(client.worker.id), ":{}".format(quantile_set['data'].shape))
                # 把数据集送到节点,并保留指针
                fed_data_dict[client.worker.id] = quantile_set['data'].send(client.worker)
                fed_target_dict[client.worker.id] = quantile_set['target'].send(client.worker)
                # 配置数据集
                dataset = sy.BaseDataset(fed_data_dict[client.worker.id], fed_target_dict[client.worker.id])
                datasets.append(dataset)
                # 设置优化器
                optimizers[client.worker.id] = torch.optim.SGD(self.server.nets[name].parameters(), lr=0.01,
                                                               weight_decay=0.001, momentum=0.9)
            fed_dataset = sy.FederatedDataset(datasets)
            print('federated_worker:', fed_dataset.workers)

            # self.server.nets[name] = torch.load('{}.pkl'.format(name))
            self.server.nets[name].to(device)

            loader = sy.FederatedDataLoader(
                federated_dataset=fed_dataset,
                batch_size=self.params.batch_size,
                shuffle=True,
                drop_last=False
            )

            loss_func = torch.nn.CrossEntropyLoss().cuda()

            for epoch in range(self.params.epochs+1):
                loss_accum = 0
                for step, (batch_x, batch_y) in enumerate(loader):
                    # 下发模型
                    self.server.nets[name].send(batch_x.location)
                    # 获取优化器
                    optimizer = optimizers[batch_x.location.id]

                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    pred_y = self.server.nets[name](batch_x)
                    # 回收损失函数
                    loss = loss_func(pred_y, torch.max(batch_y, 1)[0]).get()
                    # 回收模型
                    self.server.nets[name].get()
                    # print("step {}".format(step), "loss {}".format(loss))
                    loss_accum = loss_accum + loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if epoch % 5 == 0:
                    print('epoch: {} loss accumavg:{}'.format(epoch, loss_accum / len(loader)))
                    print('-' * 60)
                if epoch % 50 == 0:
                    torch.save(self.server.nets[name], '{}.pkl'.format(name))

    def learn_feature_engineering(self):
        # 加载目标数据集
        for client in self.clients:
            client.load_target_dataset("dataset_path")

        # 加载训练模型
        for name in (Unaries.name+Binaries.name):
            self.server.nets[name] = torch.load("sandbox/{}.pkl".format(name))

        for attempt in range(self.params.attempts):
            tr_name = choice(Unaries.name + Binaries.name)
            tr_ind=(Unaries.name+Binaries.name).index(tr_name)
            inds = np.random.randint(0, len(self.clients), size=2)
            ind_1,ind_2 = inds
            new_qsa = None
            if ind_1 == ind_2:
                # 对一个client做特征变换
                client = self.clients[ind_1]
                #返回两列
                feature_num = client.dataset['data'].shape[2]
                col_rand_ind = np.arange(feature_num)
                np.random.shuffle(col_rand_ind)
                col_rand_ind = col_rand_ind[0:2]
                idx1,idx2 = col_rand_ind
                f1,f2 = client.dataset[idx1],client.dataset[idx2]
                transformation = (Unaries.func+Binaries.func)[tr_ind]
                new_feature = transformation(f1,f2)
                new_qsa = get_sketch(client.params.n_bins,new_feature,client.dataset['target'])
            else:
                # 对两个client做特征变换
                pass
            new_qsa = torch.tensor(new_qsa).float()



    def check_improvement(self):
        pass


if __name__ == "__main__":
    data_center = FederatedLFE()
    # data_center.generate_qsa("./raw")
    data_center.train_fedavg_mlp()
    # bob = sy.VirtualWorker(hook, id="bob")
    # x = torch.tensor([1, 2, 3, 4, 5])
    # y = torch.tensor([1, 1, 1, 1, 1])
    # x_ptr = x.send(bob)
    # y_ptr = y.send(bob)
    # z_ptr = x_ptr + x_ptr
    # z = z_ptr.get()
    # print(z)
    # x = torch.tensor([1, 2, 3, 4, 5])
    # y = torch.tensor([1, 1, 1, 1, 1])
    # x_ptr = x.send(bob)
    # y_ptr = y.send(bob)
    # z_ptr = x_ptr + x_ptr
    # z = z_ptr.get()
    # print(z)
