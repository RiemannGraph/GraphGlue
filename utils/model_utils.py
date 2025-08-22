import torch
import torch.nn.functional as F
import torch.nn as nn


class ActivateModule(nn.Module):
    ACTIVATION_MAP = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "none": nn.Identity,
    }
    def __init__(self, act_str: str):
        super().__init__()
        self.act = self.ACTIVATION_MAP[act_str]()

    def forward(self, x):
        return self.act(x)


class NormModule(nn.Module):
    NORM_MAP = {
        "layer_norm": nn.LayerNorm,
        "batch_norm": nn.BatchNorm1d,
        "none": nn.Identity,
    }
    def __init__(self, norm_str: str, dim: int):
        super().__init__()
        self.norm = self.NORM_MAP[norm_str](dim)

    def forward(self, x):
        return self.norm(x)