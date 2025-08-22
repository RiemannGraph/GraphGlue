import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import global_mean_pool
from modules.layers import GNNLayer, FeedForwardLayer
from torch_geometric.data import Data, Batch
from utils.model_utils import ActivateModule, NormModule


class PTGB(nn.Module):
    def __init__(self, M, d, hid_dim):
        super(PTGB, self).__init__()
        self.M = M
        self.generators = nn.Parameter(torch.empty(M, d))
        nn.init.kaiming_normal_(self.generators.data)
        self.W_q = nn.Linear(d, hid_dim)
        self.W_k = nn.Linear(d, hid_dim)

    def forward(self, x, n_id, sub_edge_index, n_sub_batch):
        N = x.shape[0]
        weights = torch.sigmoid(self.W_q(self.generators) @ self.W_k(x).t())   # [M, N]
        aug_sub_graphs = []

        add_n_sub_batch = torch.arange(x.shape[0]).to(x.device)
        add_n_id = torch.tensor([N] * N).long()
        add_map_id = torch.arange(N, N + N)
        counts = torch.bincount(n_sub_batch)
        add_sub_edge_src = torch.repeat_interleave(add_map_id[add_n_sub_batch], counts)
        add_edge_index = torch.stack([torch.arange(n_sub_batch.shape[0]), add_sub_edge_src], dim=0)
        # add_edge_index = torch.concat([add_edge_index, add_edge_index[[1, 0]]], dim=-1)
        new_sub_edge_index = torch.cat([sub_edge_index, add_edge_index], dim=-1)
        new_n_id = torch.concat([n_id, add_n_id], dim=-1)
        new_n_sub_batch = torch.concat([n_sub_batch, add_n_sub_batch], dim=-1)

        x_expand = x.unsqueeze(0).repeat(self.M, 1, 1)     # [M, N, d]
        p_expand = self.generators.unsqueeze(1)     # [M, 1, d]
        xp = torch.concat([x_expand, p_expand], dim=1)   # [M, N+1, d]
        for i in range(self.M):
            add_sub_edge_weight = weights[i][n_id]
            new_sub_edge_weight = torch.concat([torch.ones_like(sub_edge_index[0]), add_sub_edge_weight], dim=-1)
            aug_sub_graph = Data(x=xp[i], sub_edge_index=new_sub_edge_index, sub_edge_weight=new_sub_edge_weight,
                                 n_id=new_n_id, n_sub_batch=new_n_sub_batch)
            aug_sub_graphs.append(aug_sub_graph)

        # Additional memory cost
        # aug_sub_graph_batch = Batch.from_data_list(aug_sub_graphs)
        # return aug_sub_graph_batch

        return aug_sub_graphs


class PooLedGNN(nn.Module):
    def __init__(self, conv_name: str, n_layers: int,
                 in_dim: int, hid_dim: int,
                 normalize: bool = True, bias: bool = True,
                 norm_str: str = "ln", act_str: str = "relu", drop=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            GNNLayer(conv_name, in_dim, hid_dim,
                normalize, bias, norm_str, act_str, drop)
        ])
        for _ in range(n_layers - 1):
            self.convs.append(
                GNNLayer(conv_name, hid_dim, hid_dim,
                    normalize, bias, norm_str, act_str, drop))
        self.out_norm = NormModule(norm_str, hid_dim)
        self.out_fc = FeedForwardLayer(hid_dim, hid_dim, hid_dim, bias, act_str, drop)

    def forward(self, x, edge_index, edge_weight, pool_batch=None):
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
        x = self.out_norm(x)
        x = global_mean_pool(x, pool_batch)
        return x


if __name__ == '__main__':
    import torch_geometric.transforms as T
    x = torch.randn(5, 16)
    edge_index = torch.tensor([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 3], [2, 4], [3, 2], [3, 4], [4, 2], [4, 3]]).t()
    graph = Data(x, edge_index)
    graph = T.RootedEgoNets(2)(graph)
    ptgb = PTGB(2, 16, 32)
    x, n_id, sub_edge_index, n_sub_batch = graph.x, graph.n_id, graph.sub_edge_index, graph.n_sub_batch
    sub_graphs = ptgb(x, n_id, sub_edge_index, n_sub_batch)
    gnn = PooLedGNN("gcn", 2, 16, 16, bias=True, norm_str="none", act_str="relu", drop=0.1)
    zs = []
    for sub_graph in sub_graphs:
        z = gnn(sub_graph.x[sub_graph.n_id], sub_graph.sub_edge_index, sub_graph.sub_edge_weight, sub_graph.n_sub_batch)
        zs.append(z)
    zs = torch.stack(zs, dim=0)