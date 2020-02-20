from sklearn.model_selection import cross_val_score

from kits.transformations import Unaries, Binaries
from kits.dataset import useful_tag, useless_tag
from sklearn.ensemble import RandomForestClassifier
from role.net import MLP
import syft as sy
import numpy as np
import torch

hook = sy.TorchHook(torch)


class ParameterServer:
    def __init__(self, n_params):
        self.worker = sy.VirtualWorker(hook, id="parameter_server")
        self.neural_network = MLP(n_params)


def get_sketch(n_bins, feature, labels):
    quantile_sketch_vector = np.zeros((2, n_bins + 1))
    supr, infr = max(feature), min(feature)
    blank = supr - infr
    if blank == 0: return None
    for fv, cv in zip(feature, labels):
        idx = int((fv - infr) / blank * n_bins)
        quantile_sketch_vector[cv, idx] += 1
    return quantile_sketch_vector[:, :-1]


def modify_dataset(dataset, col_rand_ind, feature):
    dataset[:, col_rand_ind] = 0
    dataset[:, col_rand_ind[0]] = feature


def reverse_dataset(dataset, col_rand_ind, feature):
    dataset[:, col_rand_ind] = feature


class Client:
    @staticmethod
    def trans_per_capita_set(col_num):
        return col_num * 2, col_num * (col_num - 1)

    def __init__(self, worker_id, client_params):
        self.worker = sy.VirtualWorker(hook, id="client_" + str(worker_id))
        self.params = client_params
        self.base_model = RandomForestClassifier(n_estimators=self.params.n_trees)
        self.datasets = list()
        self.qsa_set = dict()
        for name in Binaries.name:
            self.qsa_set[name] = {'data': list(), 'target': list()}
        for name in Unaries.name:
            self.qsa_set[name] = {'data': list(), 'target': list()}

    def get_dataset(self, dataset):
        self.datasets.append(dataset)

    # 对一个dataset获得指定数目的二元样本
    def get_binary_num_qsa(self, attempts, dataset, bench_score):
        estimated_classifier = RandomForestClassifier(n_estimators=self.params.n_trees)
        for attempt in range(attempts):
            col_rand_ind = np.arange(dataset['data'].shape[1])
            col_rand_ind = np.random.shuffle(col_rand_ind)[0:2]
            features = dataset['data'][:, col_rand_ind]
            f1, f2 = features[:, 0], features[:, 1]
            labels = dataset['target']
            for name, func in zip(Binaries.name, Binaries.func):
                f_t = func(f1, f2)
                if f_t is None:
                    continue
                quantile_sketch_vector = get_sketch(self.params.n_bins, f_t, labels)
                if quantile_sketch_vector is None:
                    continue
                modify_dataset(dataset, col_rand_ind, f_t)
                estimated_score = cross_val_score(estimated_classifier, dataset['data'], dataset['target'], cv=5,
                                                  scoring='f1').mean()
                self.qsa_set[name]['data'].append(quantile_sketch_vector)
                if estimated_score - bench_score > self.params.threshold:
                    self.qsa_set[name]['target'].append(useful_tag)
                else:
                    self.qsa_set[name]['target'].append(useless_tag)
                reverse_dataset(dataset, col_rand_ind, features)

    def get_unary_num_qsa(self, attempts, dataset, bench_score):
        estimated_classifier = RandomForestClassifier(n_estimators=self.params.n_trees)
        for attempt in range(attempts):
            col_rand_ind = np.arange(dataset['data'].shape[1])
            col_rand_ind = np.random.shuffle(col_rand_ind)[0:1]
            feature = dataset['data'][:, col_rand_ind]
            labels = dataset['target']
            for name, func in zip(Unaries.name, Unaries.func):
                f_t = func(feature)
                if f_t is None:
                    continue
                quantile_sketch_vector = get_sketch(self.params.n_bins, f_t, labels)
                if quantile_sketch_vector is None:
                    continue
                modify_dataset(dataset, col_rand_ind, f_t)
                estimated_score = cross_val_score(estimated_classifier, dataset['data'], dataset['target'], cv=5,
                                                  scoring='f1').mean()
                self.qsa_set[name]['data'].append(quantile_sketch_vector)
                if estimated_score - bench_score > self.params.threshold:
                    self.qsa_set[name]['target'].append(useful_tag)
                else:
                    self.qsa_set[name]['target'].append(useless_tag)
                reverse_dataset(dataset, col_rand_ind, feature)

    def generate_qsa(self):
        bench_classfier = RandomForestClassifier(n_estimators=self.params.n_trees)
        for dataset in self.datasets:
            col_num = dataset['data'].shape[1]
            u_attempts, b_attempts = Client.trans_per_capita_set(col_num)
            bench_score = cross_val_score(bench_classfier, dataset['data'], dataset['target'], cv=5,
                                          scoring='f1').mean()
            self.get_binary_num_qsa(b_attempts, dataset, bench_score)
            self.get_unary_num_qsa(u_attempts, dataset, bench_score)

# if __name__ == "__main__":
#     a = np.array([[1,2],[3,4]])
#     print(a)
#     a[:,1]=0
#     print(a)
