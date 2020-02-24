from sklearn.model_selection import cross_val_score
from kits.transformations import Unaries, Binaries
from kits.dataset import useful_tag, useless_tag
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from role.net import MLP
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
        for name in Binaries.name:
            self.nets[name] = MLP(n_params)
        for name in Unaries.name:
            self.nets[name] = MLP(n_params)


def get_sketch(n_bins, feature, labels):
    quantile_sketch_vector = np.zeros((2, n_bins + 1))
    try:
        supr, infr = max(feature), min(feature)
    except:
        print('wrong')
        print(feature)
    blank = supr - infr
    if blank == 0: return None
    for fv, cv in zip(feature, labels):
        idx = int((fv - infr) / blank * n_bins)
        quantile_sketch_vector[cv, idx] = quantile_sketch_vector[cv, idx] + 1
    return list(quantile_sketch_vector[:, :-1].flatten())


def modify_dataset(dataset, col_rand_ind, feature):
    dataset['data'][:, col_rand_ind] = 0
    dataset['data'][:, col_rand_ind[0]] = feature


def reverse_dataset(dataset, col_rand_ind, feature):
    dataset['data'][:, col_rand_ind] = feature


class Client:
    @staticmethod
    def trans_per_capita_set(col_num):
        return min([col_num * 2, 400]), min([col_num * (col_num - 1), 300])

    def __init__(self, worker_id, client_params):
        self.worker = sy.VirtualWorker(hook, id="client_" + str(worker_id))
        self.params = client_params
        self.qsa_path = 'worker_{}'.format(worker_id)
        # self.base_model = RandomForestClassifier(n_estimators=self.params.n_trees)
        self.base_model = DecisionTreeClassifier()
        self.datasets = list()
        self.qsa_set = dict()
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
        print('benchscore:', bench_score)

        if bench_score > 0.95:
            self.params.threshold = 0.008
        elif bench_score < 0.4:
            self.params.threshold = 0.04
        elif bench_score < 0.6:
            self.params.threshold = 0.02
        else:
            self.params.threshold = 0.01
        # estimated_classifier = RandomForestClassifier(n_estimators=self.params.n_trees)
        estimated_classifier = DecisionTreeClassifier()
        pos, neg = 0, 0
        for attempt in range(attempts):
            col_rand_ind = np.arange(dataset['data'].shape[1])
            np.random.shuffle(col_rand_ind)
            col_rand_ind = col_rand_ind[0:2]
            features = dataset['data'][:, col_rand_ind]
            f1, f2 = features[:, 0], features[:, 1]
            labels = dataset['target']
            for name, func in zip(Binaries.name, Binaries.func):
                if (pos + neg) % 50 == 0:
                    print("binary  pos:{} neg:{}".format(pos, neg))
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
                    pos = pos + 1
                else:
                    self.qsa_set[name]['target'].append(useless_tag)
                    neg = neg + 1
                reverse_dataset(dataset, col_rand_ind, features)

    def get_unary_num_qsa(self, attempts, dataset, bench_score):
        if bench_score > 0.95:
            self.params.threshold = 0.008
        elif bench_score < 0.4:
            self.params.threshold = 0.04
        elif bench_score < 0.6:
            self.params.threshold = 0.02
        else:
            self.params.threshold = 0.01
        pos, neg = 0, 0
        estimated_classifier = DecisionTreeClassifier()
        for attempt in range(attempts):
            col_rand_ind = np.arange(dataset['data'].shape[1])
            np.random.shuffle(col_rand_ind)
            col_rand_ind = col_rand_ind[0:1]
            features = dataset['data'][:, col_rand_ind]
            f1 = features[:, 0]
            labels = dataset['target']
            for name, func in zip(Unaries.name, Unaries.func):
                if (pos + neg) % 50 == 0:
                    print("unary  pos:{} neg:{}".format(pos, neg))
                f_t = func(f1)
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
                    pos = pos + 1
                else:
                    self.qsa_set[name]['target'].append(useless_tag)
                    neg = neg + 1
                reverse_dataset(dataset, col_rand_ind, features)

    def generate_qsa(self):
        # bench_classfier = RandomForestClassifier(n_estimators=self.params.n_trees)
        bench_classfier = DecisionTreeClassifier()
        for dataset in self.datasets:
            print('id:{} dataset_name:{}'.format(self.worker.id, dataset['name']))
            col_num = dataset['data'].shape[1]
            u_attempts, b_attempts = Client.trans_per_capita_set(col_num)
            print("ubatp:", u_attempts, b_attempts)
            bench_score = cross_val_score(bench_classfier, dataset['data'], dataset['target'], cv=5,
                                          scoring='f1').mean()
            self.get_binary_num_qsa(b_attempts, dataset, bench_score)
            self.get_unary_num_qsa(u_attempts, dataset, bench_score)
        for t_name, t_set in self.qsa_set.items():
            t_set['data'] = np.array(t_set['data'])

    # 加载目标数据集
    def load_target_dataset(self,path):
        return None


if __name__ == "__main__":
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(list(a))
    print(a)
