import torch


class MLP(torch.nn.Module):
    def __init__(self, params):
        super(MLP, self).__init__()
        self.params = params
        self.fc1 = torch.nn.Linear(params.size, params.n_cores)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(params.n_cores, 1)

    def forward(self, qsa):
        qsa = qsa.view(-1, self.params.n_cls * self.params.size)
        qsa_out = self.dropout(self.fc1(qsa))
        qsa_out = torch.nn.functional.relu(qsa_out)
        return torch.nn.functional.softmax(self.fc2(qsa_out))
