import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.data import Data
from cores.layers import ActivateModule, NormModule, FeedForwardLayer, GNNLayer


class PTGB(nn.Module):
    def __init__(self, M, d, hid_dim):
        super(PTGB, self).__init__()
        self.M = M
        self.generators = nn.Parameter(torch.empty(M, d))
        nn.init.kaiming_normal_(self.generators.data)
        self.W_q = nn.Linear(d, hid_dim)
        self.W_k = nn.Linear(d, hid_dim)

    def forward(self, x, edge_index, edge_weight, batch, batch_size):
        N = x.shape[0]

        weights = torch.sigmoid(self.W_q(self.generators) @ self.W_k(x).t())  # [M, N]

        add_batch = torch.arange(batch_size, device=x.device)
        new_batch = torch.concat([batch, add_batch], dim=0)

        counts = torch.bincount(batch)
        add_edge_src = torch.arange(N, N + batch_size, device=x.device).repeat_interleave(counts)
        add_edge_dst = torch.arange(N, device=x.device)
        add_edge_index = torch.stack([add_edge_src, add_edge_dst], dim=0)
        new_edge_index = torch.concat([edge_index, add_edge_index], dim=-1)
        aug_graphs = []
        for i in range(self.M):
            xp = torch.concat([x, self.generators[i: i+1].repeat(batch_size, 1)], dim=0)
            new_edge_weight = torch.concat([edge_weight, weights[i]], dim=-1)
            aug_graph = Data(x=xp, edge_index=new_edge_index, edge_weight=new_edge_weight, batch=new_batch)
            aug_graphs.append(aug_graph)
        return aug_graphs


class PooLedSubgraphGNN(nn.Module):
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

    def forward(self, x, edge_index, edge_weight=None, pool_batch=None):
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
        x = self.out_norm(x)
        x = global_mean_pool(x, pool_batch)
        return x


class RPGraphFM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.input_lin = FeedForwardLayer(configs.in_dim, configs.hid_dim, configs.hid_dim,
                                          configs.bias, configs.act_str, configs.drop)
        self.ptg_bank = PTGB(configs.M, configs.d, configs.hid_dim)
        self.encoder = PooLedSubgraphGNN(configs.conv_name, configs.n_layers,
                                         configs.hid_dim, configs.hid_dim,
                                         configs.normalize, configs.bias,
                                         configs.norm_str, configs.act_str, configs.drop)
        self.M = configs.M

    def forward(self, graph: Data):
        """

        :param graph: 1. Feature dimension is unified. 2. BatchData
        :return: node/graph embedding, tangent vectors [torch.Tensor, torch.Tensor]
        """
        x, edge_index, edge_weight, batch, batch_size = graph.x, graph.edge_index, graph.edge_weight, graph.batch, graph.batch_size
        x = self.input_lin(x)
        z = self.encoder(x, edge_index, edge_weight, batch)

        aug_graphs = self.ptg_bank(graph.x, graph.edge_index, graph.edge_weight, graph.batch, graph.batch_size)
        z_tan = []
        for aug_graph in aug_graphs:
            tan = self.encoder(aug_graph.x, aug_graph.edge_index, aug_graph.edge_weight, aug_graph.batch)
            z_tan.append(tan)
        z_tan = torch.stack(z_tan, dim=0)
        return z, z_tan

    @staticmethod
    def knn_graph(h: torch.Tensor, top_k, return_weight: bool = False):
        """
        Construct KNN graph for graph-level datasets.

        :param h: All the graph representations for a graph-level dataset.
        :param top_k: the number of K nearest neighbors.
        :param return_weight: If True, return edge_weight, otherwise, return None.
        :return: edge_index, edge_weight [Torch.Tensor, torch.Tensor]
        """
        assert top_k < h.shape[0], f"top_k={top_k} must be smaller than f{h.shape[0]}"
        similarity = h @ h.t()
        _, indices = similarity.topk(k=top_k, dim=-1)
        edge_index = indices.t()
        if return_weight:
            edge_weight = similarity[edge_index[0], edge_index[1]]
        else:
            edge_weight = None
        return edge_index, edge_weight

    @staticmethod
    def parallel_translation(basis_src, basis_dst):
        """
        Estimation of parallel translation between two tangent spaces.
        :param basis_src: [M, d]
        :param basis_dst: [M, d]
        :return: PT matrix: torch.Tensor
        """
        U, _, VT = torch.svd(basis_dst @ basis_src.t())
        P = U @ VT
        return P

    @staticmethod
    def volume_ratio(basis_src, basis_dst):
        """
        Volume ratio between two tangent spaces to estimate Ricci Curvature.
        :param basis_src: [M, d]
        :param basis_dst: [M, d]
        :return: ratio: torch.Tensor
        """
        vol_src, vol_dst = torch.det(basis_src), torch.det(basis_dst)
        vol_src_stable = torch.sqrt(vol_src ** 2 + 1e-6)
        vol_dst_stable = torch.sqrt(vol_dst ** 2 + 1e-6)
        r = vol_dst_stable / vol_src_stable
        return r