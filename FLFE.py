import pandas as pd
import numpy as np
import syft as sy
import torch

from kits.transformations import unaries, binaries


class Params:
    def __init__(self):
        self.attempts = 5000
        self.epochs = 50
        self.cli_num = 3
        self.improvement_threshold = 0.01
        self.norm_bound = 10
        self.no_cuda = False


hook = sy.TorchHook(torch)


class FederatedLFE:
    def __init__(self):
        self.params = Params()
        self.workers = list()
        for i in range(self.params.cli_num):
            self.workers.append(sy.VirtualWorker(hook, id="worker" + str(i)))

    def generate_qsa(self):
        pass

    def train_fedavg_mlp(self):
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
