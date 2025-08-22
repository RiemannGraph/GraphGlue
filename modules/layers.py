import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from utils.model_utils import ActivateModule, NormModule


CONV_MAP = {
    "gcn": GCNConv,
    "sage": SAGEConv,
}


class GNNLayer(nn.Module):
    def __init__(self, conv_name: str, in_dim: int, out_dim: int,
                 normalize: bool = True, bias: bool = True,
                 norm_str: str = "ln", act_str: str = "relu", drop=0.1):
        super().__init__()
        self.conv = CONV_MAP[conv_name](in_channels=in_dim, out_channels=out_dim,
                                             normalize=normalize, bias=bias)
        self.norm = NormModule(norm_str, out_dim)
        self.fc = FeedForwardLayer(out_dim, out_dim, out_dim, bias, act_str, drop)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = self.norm(x)
        x = self.fc(x)
        return x


class FeedForwardLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, bias, act_str='gelu', drop=0.3):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=bias),
            nn.Dropout(drop),
            ActivateModule(act_str),
            nn.Linear(hid_dim, out_dim, bias=bias),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = self.layer(x)
        return x