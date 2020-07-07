import pandas as pd
import numpy as np
import syft as sy
import torch
import torch.utils.data
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from role.server_client import ParameterServer, Client, get_tensor_sketch
from kits.transformations import Unaries, Binaries
from kits.encryption import *
from random import choice
from kits.dataset import read_convert_csvs, QuantileSketchDataset, get_random_feature
from torch.utils.data import DataLoader, TensorDataset
import argparse
from kits.dataset import read_domain_csv
import time

parser = argparse.ArgumentParser(description='Process some integers.')
# 尝试搜索有用特征次数，由于有14个变换，所以n_attempt=100对应14*100次
parser.add_argument("-n_attempt", type=int, default=1000)
# 训练参数
parser.add_argument("-n_epoch", type=int, default=80)
# 客户端个数，不能超过这个数字
parser.add_argument("-n_client", type=int, default=3)
# 训练参数
parser.add_argument("-batch", type=int, default=128)
# 是否调用cuda
parser.add_argument("-cuda", type=bool, default=True)
# 训练参数
parser.add_argument("-threshold", type=float, default=0.01)
# 用几棵树衡量模型
parser.add_argument("-n_trees", type=int, default=5)
# sketch的bin的个数
parser.add_argument("-n_bins", type=int, default=200)
#  几分类
parser.add_argument("-n_cls", type=int, default=2)
# 网络参数
parser.add_argument("-n_cores", type=int, default=512)
# qsa数据集位置
parser.add_argument("-qsa_root", type=str, default="new_raw/")
# 模型位置
parser.add_argument("-model_root", type=str, default="model/")

args = parser.parse_args()


class Params:
    def __init__(self):
        self.attempts = args.n_attempt
        self.epochs = args.n_epoch
        self.cli_num = args.n_client
        self.batch_size = args.batch
        self.no_cuda = args.cuda
        self.model_root = args.model_root
        self.qsa_root = args.qsa_root


class ClientParams:
    def __init__(self):
        self.threshold = args.threshold
        self.n_trees = args.n_trees
        self.n_bins = args.n_bins
        self.qsa_root = args.qsa_root


class NetParams:
    def __init__(self):
        self.size = args.n_bins
        self.n_cls = 2
        self.n_cores = args.n_cores


class FederatedLFE:
    def __init__(self):
        self.params = Params()
        self.server = ParameterServer(NetParams())
        self.clients = list()
        for i in range(self.params.cli_num):
            # 加载并生成了Client
            self.clients.append(Client(i, ClientParams()))

    def generate_qsa(self):
        datasets = read_convert_csvs(self.params.qsa_root)
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
        for client in self.clients:
            client.load_qsa_set()
        # 把数据集送到每个节点
        for name in (Unaries.name + Binaries.name):
            print('MLP:{}'.format(name))
            print('-' * 40)
            fed_data_dict, fed_target_dict = dict(), dict()
            optimizers = dict()
            datasets = list()
            for client in self.clients:
                quantile_set = {
                    'data': torch.tensor(client.qsa_set[name]['data']).float(),
                    'target': torch.tensor(np.array(client.qsa_set[name]['target']).reshape(-1)).long()
                }
                print("{}".format(client.worker.id), ":{}".format(quantile_set['data'].shape))
                # 把数据集送到节点,并保留指针
                fed_data_dict[client.worker.id] = quantile_set['data'].send(client.worker)
                fed_target_dict[client.worker.id] = quantile_set['target'].send(client.worker)
                # 配置数据集
                dataset = sy.BaseDataset(fed_data_dict[client.worker.id], fed_target_dict[client.worker.id])
                datasets.append(dataset)
                # 设置优化器
                optimizers[client.worker.id] = torch.optim.SGD(self.server.nets[name].parameters(), lr=0.0001,
                                                               momentum=0.9)
            fed_dataset = sy.FederatedDataset(datasets)
            print('federated_worker:', fed_dataset.workers)

            refine_flag = 1
            if refine_flag:
                load_federated_model(str(self.params.model_root) + '{}.pkl'.format(name), self.server.nets[name],
                                     optimizers=optimizers, device=device, lr=0.00002, mode="train")
            else:
                self.server.nets[name].to(device)

            loader = sy.FederatedDataLoader(
                federated_dataset=fed_dataset,
                batch_size=self.params.batch_size,
                shuffle=True,
                drop_last=False
            )

            loss_func = torch.nn.CrossEntropyLoss().cuda()

            for epoch in range(self.params.epochs + 1):
                loss_accum = 0
                for step, (batch_x, batch_y) in enumerate(loader):
                    # 下发模型
                    self.server.nets[name].send(batch_x.location)
                    # 获取优化器
                    optimizer = optimizers[batch_x.location.id]
                    optimizer.zero_grad()
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    pred_y = self.server.nets[name](batch_x)
                    # 回收损失函数
                    loss = loss_func(pred_y, batch_y)
                    loss.backward()
                    # 回收模型
                    self.server.nets[name].get()
                    loss = loss.get()
                    loss_accum = loss_accum + loss
                    optimizer.step()
                if epoch % 1 == 0:
                    print('epoch: {} loss avg:{} \naccum:{}'.format(epoch, loss_accum / len(loader), loss_accum))
                    print('-' * 60)
                if epoch % 10 == 0:
                    # torch.save(self.server.nets[name], str(self.params.model_root) + '{}.pkl'.format(name))
                    save_federated_model(str(self.params.model_root) + '{}.pkl'.format(name), self.server.nets[name],
                                         optimizers)

    def learn_feature_engineering(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 加载目标数据集
        self.clients[0].load_target_dataset("sandbox/new_qiye.csv")
        self.clients[1].load_target_dataset("sandbox/new_zhengfu.csv")
        self.clients[2].load_target_dataset("sandbox/new_gonggong.csv")
        # 加载训练模型
        for name in (Unaries.name + Binaries.name):
            load_model(str(self.params.model_root) + "{}.pkl".format(name), self.server.nets[name],
                       device=device, mode="eval")

        t1 = time.time()
        for attempt in range(self.params.attempts):
            for tr_name, tr_func in zip(Binaries.name, Binaries.func):
                # 寻找两个client
                ind_1, ind_2 = np.random.randint(0, self.params.cli_num, size=2)
                # 如果是一个client
                if ind_1 == ind_2:
                    client = self.clients[ind_1]
                    feature_num = client.domain_dataset['data'].shape[1]
                    idx1, idx2 = get_random_feature(feature_num, 2)
                    f1, f2 = client.domain_dataset['data'][:, idx1], client.domain_dataset['data'][:, idx2]
                    qsa1 = get_tensor_sketch(client.params.n_bins, f1, client.domain_dataset['target']).to(device)
                    qsa2 = get_tensor_sketch(client.params.n_bins, f2, client.domain_dataset['target']).to(device)
                    new_qsa = torch.cat((qsa1, qsa2), 0).get()
                    output = self.server.nets[tr_name](new_qsa)

                    new_feature_name = "{} {} {}".format(client.domain_dataset['property'][idx1], tr_name,
                                                         client.domain_dataset['property'][idx2])
                    if has_improvement(output) and new_feature_name not in client.domain_dataset['property']:
                        new_feature = tr_func(f1, f2)
                        if new_feature is None: continue
                        print(new_feature_name)
                        print('-' * 40)
                        new_feature = torch.unsqueeze(new_feature, 1)
                        client.domain_dataset['data'] = torch.cat((client.domain_dataset['data'], new_feature), 1)
                        client.domain_dataset['property'].append(new_feature_name)
                # 如果是两个client
                else:
                    client_1, client_2 = self.clients[ind_1], self.clients[ind_2]
                    fn1 = client_1.domain_dataset['data'].shape[1]
                    idx1 = get_random_feature(fn1, 1)[0]
                    f1 = client_1.domain_dataset['data'][:, idx1]
                    qsa1 = get_tensor_sketch(client_1.params.n_bins, f1, client_1.domain_dataset['target']).to(device)
                    f1, mt = encrypt_feature_in_rand(client_1.domain_dataset['data'][:, idx1], tr_name)
                    f1, qsa1 = f1.move(client_2.worker), qsa1.move(client_2.worker)
                    mt = mt.get()

                    fn2 = client_2.domain_dataset['data'].shape[1]
                    idx2 = get_random_feature(fn2, 1)[0]
                    f2 = client_2.domain_dataset['data'][:, idx2]
                    qsa2 = get_tensor_sketch(client_2.params.n_bins, f2, client_2.domain_dataset['target']).to(device)
                    new_qsa = torch.cat((qsa1, qsa2), 0).get()
                    output = self.server.nets[tr_name](new_qsa)

                    new_feature_name = "{} {} {}".format(client_1.domain_dataset['property'][idx1], tr_name,
                                                         client_2.domain_dataset['property'][idx2])

                    if has_improvement(output) and new_feature_name not in self.server.feature_names:
                        f1, f2 = f1.get(), f2.get()
                        new_feature = tr_func(f1, f2)
                        if new_feature is None: continue
                        print(new_feature_name)
                        print('-' * 40)
                        new_feature = decrypt_feature_in_rand(new_feature, mt, tr_name)
                        client = self.get_another_free_client(ind_1, ind_2)
                        new_feature = torch.unsqueeze(new_feature, 1)
                        new_feature = new_feature.send(client.worker)
                        client.domain_dataset['data'] = torch.cat((client.domain_dataset['data'], new_feature), 1)
                        self.server.feature_names.append(new_feature_name)
                        client.domain_dataset['property'].append(new_feature_name)

            #一元变换
            for tr_name, tr_func in zip(Unaries.name, Unaries.func):
                ind = np.random.randint(0, self.params.cli_num, size=1)[0]
                client = self.clients[ind]
                feature_num = client.domain_dataset['data'].shape[1]
                idx = get_random_feature(feature_num, 1)[0]
                f1 = client.domain_dataset['data'][:, idx]
                new_qsa = get_tensor_sketch(client.params.n_bins, f1, client.domain_dataset['target']).to(
                    device)
                new_qsa = new_qsa.get()
                new_feature_name = "{}({})".format(tr_name, client.domain_dataset['property'][idx])

                output = self.server.nets[tr_name](new_qsa)
                # 判断是否有提升
                if has_improvement(output) and new_feature_name not in client.domain_dataset['property']:
                    new_feature = tr_func(f1)
                    if new_feature is None: continue
                    print("new_feature:", new_feature_name)
                    print('-' * 40)
                    new_feature = torch.unsqueeze(new_feature, 1)
                    client.domain_dataset['data'] = torch.cat((client.domain_dataset['data'], new_feature), 1)
                    client.domain_dataset['property'].append(new_feature_name)
                    # print(new_feature_name)
                    # print('---------------------------------------------------------------------')
        t2 = time.time()
        print((t2 - t1) / self.params.attempts)

    def get_another_free_client(self, ind1, ind2):
        for i in range(self.params.cli_num):
            if not i == ind1 and not i == ind2:
                return self.clients[i]

    def valid_beforehand(self):
        self.clients[0].domain_dataset = read_domain_csv("sandbox/new_qiye.csv", type="array")
        self.clients[1].domain_dataset = read_domain_csv("sandbox/new_zhengfu.csv", type="array")
        self.clients[2].domain_dataset = read_domain_csv("sandbox/new_gonggong.csv", type="array")

        data, target = None, None
        for client in self.clients:
            data = client.domain_dataset['data'] if data is None else np.column_stack(
                (data, client.domain_dataset['data']))
            target = client.domain_dataset['target']

        scores = list()
        for i in range(10):
            score = cross_val_score(RandomForestClassifier(n_estimators=20), data, target, cv=10, scoring='f1').mean()
            scores.append(score)
        print('original feature num:', data.shape)
        print("benchscore:", np.mean(scores))

    def valid_afterhand(self):
        for client in self.clients:
            client.domain_dataset['data'] = client.domain_dataset['data'].get()
            client.domain_dataset['target'] = client.domain_dataset['target'].get()

        data, target = None, None
        for client in self.clients:
            data = client.domain_dataset['data'] if data is None else torch.cat((data, client.domain_dataset['data']),
                                                                                1)
            target = client.domain_dataset['target']

        data, target = np.array(data), np.array(target)
        scores = list()
        for i in range(10):
            score = cross_val_score(RandomForestClassifier(n_estimators=20), data, target, cv=10, scoring='f1').mean()
            scores.append(score)
        print('learned feature num:', data.shape)
        print("learned score:", np.mean(scores))


def has_improvement(output):
    output = torch.nn.functional.softmax(output, dim=0)
    result_tensors = torch.max(output, 0)
    # print(result_tensors)
    if result_tensors[1] == 1 and result_tensors[0] > 0.9:
        # print("-----------------------------------------------")
        print(result_tensors[0])
        return True
    return False


def save_federated_model(path, model, optimizers):
    optimizer_states = dict()
    for name in optimizers.keys():
        optimizer_states[name] = optimizers[name].state_dict()
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_states': optimizer_states
    }, path)


def load_federated_model(path, model, device, mode, optimizers=None, lr=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    if optimizers is not None:
        optimizer_states = checkpoint['optimizer_states']
        for name in optimizer_states:
            optimizers[name].load_state_dict(optimizer_states[name])
            if lr is not None:
                for param_group in optimizers[name].param_groups:
                    param_group['lr'] = lr
            print(optimizers[name])
    if mode == "train":
        model.train()
    elif mode == "eval":
        model.eval()


def save_model(path, model, optimizer):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, path)


def load_model(path, model, device, mode, optimizer=None, lr=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    if optimizer is not None:
        optimizer_state = checkpoint['optimizer_state']
        optimizer.load_state_dict(optimizer_state)
        if lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        print(optimizer)

    if mode == "train":
        model.train()
    elif mode == "eval":
        model.eval()


if __name__ == "__main__":
    '''加载联邦特征工程模块'''
    data_center = FederatedLFE()
    '''训练mlp的
    # data_center.train_fedavg_mlp()
    '''

    '''产生训练mlp用的qsa的
    data_center.generate_qsa()
    '''

    ''' 
    一个特征数据集，作用是把3个数据集拼起来，得到一个准确率
    3个数据集名字在里面目前是硬编码的,你可以改一下，从外面传进来
    '''
    data_center.valid_beforehand()

    '''
    联邦自动化特征工程模块，你要写一个同态加密模块换掉
    '''
    data_center.learn_feature_engineering()

    '''
    将产生完新特征的拼接起来，再测一次准确率， 
    '''
    data_center.valid_afterhand()
    # frame = pd.read_csv("sandbox/cpu_small.csv")
