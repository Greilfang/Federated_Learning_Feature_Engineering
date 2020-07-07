from sklearn.model_selection import cross_val_score

from kits.transformations import Unaries, Binaries
from kits.dataset import useful_tag, useless_tag, read_domain_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from role.net import MLP, SubMLP
import syft as sy
import numpy as np
import torch
import pickle
import os

hook = sy.TorchHook(torch)


class ParameterServer:
    def __init__(self, n_params):
        self.worker = sy.VirtualWorker(hook, id="parameter_server")
        self.nets = dict()
        self.feature_names = []
        for name in Binaries.name:
            self.nets[name] = SubMLP(n_params, "binary")
        for name in Unaries.name:
            self.nets[name] = SubMLP(n_params, "unary")


def get_sketch(n_bins, feature, labels):
    idx0, idx1 = np.where(labels == 0), np.where(labels == 1)
    sketch0, _ = np.histogram(a=feature[idx0], bins=n_bins)
    sketch1, _ = np.histogram(a=feature[idx1], bins=n_bins)
    sketch0 = -10 + 20 * (sketch0 - np.min(sketch0)) / (np.max(sketch0) - np.min(sketch0))
    sketch1 = -10 + 20 * (sketch1 - np.min(sketch1)) / (np.max(sketch1) - np.min(sketch1))
    quantile_sketch = np.concatenate((sketch0, sketch1))
    return list(quantile_sketch)


def get_tensor_sketch(n_bins, feature, labels):
    if feature.location is None:
        supr, infr = torch.max(feature), torch.min(feature)
    else:
        supr, infr = torch.max(feature).get(), torch.min(feature).get()
    idx0, idx1 = torch.where(labels == 0), torch.where(labels == 1)

    sketch0 = torch.histc(feature[idx0], bins=n_bins, min=float(infr), max=float(supr))
    sketch1 = torch.histc(feature[idx1], bins=n_bins, min=float(infr), max=float(supr))

    sketch0 = -10 + 20 * ((sketch0) - torch.min(sketch0)) / (torch.max(sketch0) - torch.min(sketch0))
    sketch1 = -10 + 20 * ((sketch1) - torch.min(sketch1)) / (torch.max(sketch1) - torch.min(sketch1))
    quantile_sketch = torch.cat((sketch0, sketch1), 0)
    return quantile_sketch


def modify_dataset(dataset, feature):
    # dataset['data'][:, col_rand_ind[0]] = feature
    dataset['data'] = np.column_stack((dataset['data'], feature))


def reverse_dataset(dataset):
    # dataset['data'][:, col_rand_ind] = feature
    # dataset['data'] = dataset['data'][:, :-1]
    dataset['data'] = np.delete(dataset['data'], -1, axis=1)


class Client:
    @staticmethod
    def trans_per_capita_set(col_num):
        return min([int(col_num), 80]), min([int(col_num * (col_num - 1) / 2), 80])

    def __init__(self, worker_id, client_params):
        self.worker = sy.VirtualWorker(hook, id="client_" + str(worker_id))
        self.params = client_params
        self.qsa_path = 'worker_{}'.format(worker_id)
        self.datasets = list()
        self.qsa_set = dict()
        self.domain_dataset = {'data': None, 'target': None}
        for name in Binaries.name:
            self.qsa_set[name] = {'data': list(), 'target': list()}
        for name in Unaries.name:
            self.qsa_set[name] = {'data': list(), 'target': list()}

    def save_qsa_set(self):
        with open(self.qsa_path, 'wb') as f:
            pickle.dump(self.qsa_set, f)

    def load_qsa_set(self):
        if os.access(self.qsa_path, os.R_OK):
            with open(self.qsa_path, 'rb') as f:
                self.qsa_set = pickle.load(f)
            for name in Binaries.name:
                self.qsa_set[name]['data'] = list(self.qsa_set[name]['data'])
                self.qsa_set[name]['target'] = list(self.qsa_set[name]['target'])
            for name in Unaries.name:
                self.qsa_set[name]['data'] = list(self.qsa_set[name]['data'])
                self.qsa_set[name]['target'] = list(self.qsa_set[name]['target'])

    def get_dataset(self, dataset):
        self.datasets.append(dataset)

    # 对一个dataset获得指定数目的二元样本
    def get_binary_num_qsa(self, attempts, dataset, bench_score):
        threshold = self.params.threshold
        estimated_classifier = RandomForestClassifier(n_estimators=self.params.n_trees)
        pos, neg = 0, 0
        labels = dataset['target']
        for attempt in range(attempts):
            col_rand_ind = np.arange(dataset['data'].shape[1])
            np.random.shuffle(col_rand_ind)
            col_rand_ind = col_rand_ind[0:2]
            features = dataset['data'][:, col_rand_ind]
            f1, f2 = features[:, 0], features[:, 1]
            for name, func in zip(Binaries.name, Binaries.func):
                # if (pos + neg) % 100 == 0:
                #     print("binary  pos:{} neg:{}".format(pos, neg))
                f_t = func(f1, f2)
                if f_t is None:
                    continue
                sketch0 = get_sketch(self.params.n_bins, f1, labels)
                sketch1 = get_sketch(self.params.n_bins, f2, labels)
                if sketch0 is None or sketch1 is None:
                    continue
                quantile_sketch_vector = sketch0 + sketch1

                modify_dataset(dataset, f_t)
                estimated_score = get_avg_score(estimated_classifier, dataset['data'], dataset['target'], 10)
                if estimated_score - bench_score > threshold:
                    self.qsa_set[name]['data'].append(quantile_sketch_vector)
                    self.qsa_set[name]['target'].append(useful_tag)
                    pos = pos + 1
                elif estimated_score < bench_score:
                    self.qsa_set[name]['data'].append(quantile_sketch_vector)
                    self.qsa_set[name]['target'].append(useless_tag)
                    neg = neg + 1
                reverse_dataset(dataset)
        print("binary  pos:{} neg:{}".format(pos, neg))

    def get_unary_num_qsa(self, attempts, dataset, bench_score):
        threshold = self.params.threshold
        pos, neg = 0, 0
        estimated_classifier = RandomForestClassifier(n_estimators=self.params.n_trees)
        labels = dataset['target']
        for attempt in range(attempts):
            col_rand_ind = np.arange(dataset['data'].shape[1])
            np.random.shuffle(col_rand_ind)
            col_rand_ind = col_rand_ind[0:1]
            features = dataset['data'][:, col_rand_ind]
            f1 = features[:, 0]
            for name, func in zip(Unaries.name, Unaries.func):
                # if (pos + neg) % 100 == 0:
                #     print("unary  pos:{} neg:{}".format(pos, neg))
                f_t = func(f1)
                if f_t is None:
                    continue
                quantile_sketch_vector = get_sketch(self.params.n_bins, f1, labels)
                if quantile_sketch_vector is None:
                    continue
                modify_dataset(dataset, f_t)
                estimated_score = get_avg_score(estimated_classifier, dataset['data'], dataset['target'], 10)
                if estimated_score - bench_score > threshold:
                    self.qsa_set[name]['target'].append(useful_tag)
                    self.qsa_set[name]['data'].append(quantile_sketch_vector)
                    pos = pos + 1
                elif estimated_score < bench_score:
                    self.qsa_set[name]['target'].append(useless_tag)
                    self.qsa_set[name]['data'].append(quantile_sketch_vector)
                    neg = neg + 1
                reverse_dataset(dataset)
        print("unary  pos:{} neg:{}".format(pos, neg))

    def generate_qsa(self):
        bench_classfier = RandomForestClassifier(n_estimators=self.params.n_trees)
        for dataset in self.datasets:
            print('id:{} dataset_name:{}'.format(self.worker.id, dataset['name']))
            col_num = dataset['data'].shape[1]
            u_attempts, b_attempts = Client.trans_per_capita_set(col_num)
            avg_score = get_avg_score(bench_classfier, dataset['data'], dataset['target'], 10)
            print("benchmark:", avg_score)

            if avg_score > 0.5:
                self.get_binary_num_qsa(b_attempts, dataset, avg_score)
                self.get_unary_num_qsa(u_attempts, dataset, avg_score)
        for t_name, t_set in self.qsa_set.items():
            t_set['data'] = np.array(t_set['data'])

    # 加载目标数据集
    def load_target_dataset(self, path):
        self.domain_dataset = read_domain_csv(path, type="tensor")
        self.domain_dataset['data'] = self.domain_dataset['data'].send(self.worker)
        self.domain_dataset['target'] = self.domain_dataset['target'].send(self.worker)

        # print(self.domain_dataset['name'], ' ', self.domain_dataset['data'].size())
        # print(self.domain_dataset['data'].location)


def get_avg_score(classifier, data, target, times):
    scores = list()
    for i in range(times):
        scores.append(cross_val_score(classifier, data, target, cv=10, scoring='f1').mean())
    return np.mean(scores)


if __name__ == "__main__":
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(list(a))
    print(a)
