from kits.transformations import Unaries, Binaries
import syft as sy
import torch

hook = sy.TorchHook(torch)


class ParameterServer:
    def __init__(self):
        self.worker = sy.VirtualWorker(hook, id="parameter_server")


class Client:

    @staticmethod
    def trans_per_capita_set(col_num):
        return col_num * 2, col_num * (col_num - 1)

    def __init__(self, worker_id):
        self.worker = sy.VirtualWorker(hook, id="client_" + str(worker_id))
        self.datasets = list()
        self.qsa_set = dict()

    def get_dataset(self, dataset):
        self.datasets.append(dataset)

    def get_binary_num_qsa(self, num):
        pass

    def get_unary_num_qsa(self, num):
        pass

    def generate_qsa(self):
        for dataset in self.datasets:
            col_num = dataset['data'].shape[1]
            unary_num, binary_num = Client.trans_per_capita_set(col_num)

            self.get_binary_num_qsa(binary_num)
            self.get_unary_num_qsa(unary_num)
