import torch
import torch.nn as nn
from cores.models import PTGB, PooLedSubgraphGNN


class RPGraphFMPretrainer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.ptg_bank = PTGB(configs.M, configs.d, configs.hid_dim)
        self.encoder = PooLedSubgraphGNN(configs.conv_name, configs.n_layers,
                                         configs.in_dim, configs.hid_dim,
                                         configs.normalize, configs.bias,
                                         configs.norm_str, configs.act_str, configs.drop)
        self.M = configs.M

    def forward(self, batch_graph):
        pass