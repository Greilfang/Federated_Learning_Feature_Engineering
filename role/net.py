import torch


class MLP(torch.nn.Module):
    def __init__(self, params, ttype):
        super(MLP, self).__init__()
        self.params = params
        if ttype == 'unary':
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(params.size * params.n_cls, params.n_cores),
                torch.nn.Dropout(0.2),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(params.n_cores, 2)
            )
        elif ttype == 'binary':
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(params.size * params.n_cls * 2, params.n_cores * 2),
                torch.nn.Dropout(0.2),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(params.n_cores * 2, 2)
            )

    def forward(self, qsa):
        qsa_out = self.linear(qsa)
        return qsa_out


class SubMLP(torch.nn.Module):
    def __init__(self, params,type):
        super(SubMLP, self).__init__()
        self.params = params
        if type == "unary":
            input_size = params.size * params.n_cls
        elif type == "binary":
            input_size = 2 * params.size * params.n_cls
        n_cores = params.n_cores

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(input_size, n_cores),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(n_cores, 2),
        )

    def forward(self, x):
        out = self.linear(x)
        return out
