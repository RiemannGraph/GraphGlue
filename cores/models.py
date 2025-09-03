import torch
import torch.nn as nn
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.data import Data
from cores.layers import ActivateModule, NormModule, FeedForwardLayer, GNNLayer
from cores.loss_funcs import PTGBLoss, ContrastiveLoss, GeometricPersistLoss
from data.data_process import search_adjacent_edges
from typing import List, Optional

EPS = 1e-6


class PTGB(nn.Module):
    def __init__(self, num_generators, hid_dim, att_dim):
        super(PTGB, self).__init__()
        self.num_generators = num_generators
        self.generators = nn.Parameter(torch.empty(num_generators, hid_dim))
        nn.init.kaiming_normal_(self.generators.data)
        self.W_q = nn.Linear(hid_dim, att_dim)
        self.W_k = nn.Linear(hid_dim, att_dim)

    def forward(self, x, edge_index, edge_weight, batch, batch_graph_nums):
        """

        :param x: [N, d]
        :param edge_index: [2, E]
        :param edge_weight: [E,]
        :param batch: [N]
        :param batch_graph_nums: If graph-batch, batch_nums=batch_size.
                                If neighbor-batch, batch_nums=node_nums of sampled graph.

        :return: List[Data]
        """
        N = x.shape[0]

        weights = torch.sigmoid(self.W_q(self.generators) @ self.W_k(x).t())  # [M, N]

        add_batch = torch.arange(batch_graph_nums, device=x.device)
        new_batch = torch.concat([batch, add_batch], dim=0)

        counts = torch.bincount(batch)
        add_edge_src = torch.arange(N, N + batch_graph_nums, device=x.device).repeat_interleave(counts)
        add_edge_dst = torch.arange(N, device=x.device)
        add_edge_index = torch.stack([add_edge_src, add_edge_dst], dim=0)
        new_edge_index = torch.concat([edge_index, add_edge_index], dim=-1)
        aug_graphs = []
        for i in range(self.num_generators):
            xp = torch.concat([x, self.generators[i: i + 1].repeat(batch_graph_nums, 1)], dim=0)
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
        self.num_samples = configs.num_samples
        self.num_generators = configs.num_generators
        self.input_lin = FeedForwardLayer(configs.in_dim, configs.hid_dim, configs.hid_dim,
                                          configs.bias, configs.act_str, configs.drop)
        self.ptg_bank = PTGB(configs.num_generators, configs.hid_dim, configs.att_dim)
        self.encoder = PooLedSubgraphGNN(configs.conv_name, configs.n_layers,
                                         configs.hid_dim, configs.hid_dim,
                                         configs.normalize, configs.bias,
                                         configs.norm_str, configs.act_str, configs.drop)
        self.ptg_loss = PTGBLoss(configs.num_generators, configs.temperature)
        self.contra_loss = ContrastiveLoss(configs.temperature)
        self.geo_loss = GeometricPersistLoss(configs.regular_coef_pt, configs.regular_coef_curv)

    def forward(self, graph: Data, batch_graph_nums: int = None):
        """

        :param graph: 1) Feature dimension is unified. 2) BatchData
        :param batch_graph_nums: If graph-batch, batch_nums=batch_size.
                                If neighbor-batch, batch_nums=node_nums of sampled subgraph.

        :return: node/graph embedding, tangent vectors [torch.Tensor, torch.Tensor] with shape [N, d] [N, M, d]
        """
        if batch_graph_nums is None:
            if hasattr(graph, "batch_graph_nums"):
                batch_graph_nums = graph.batch_graph_nums
            else:
                batch_graph_nums = graph.batch_size

        x, edge_index, edge_weight, batch = graph.x, graph.edge_index, graph.edge_weight, graph.batch
        x = self.input_lin(x)
        z = self.encoder(x, edge_index, edge_weight, batch)

        aug_graphs = self.ptg_bank(x, graph.edge_index, graph.edge_weight, graph.batch, batch_graph_nums)
        z_aug = []
        for aug_graph in aug_graphs:
            tan = self.encoder(aug_graph.x, aug_graph.edge_index, aug_graph.edge_weight, aug_graph.batch)
            z_aug.append(tan)
        z_aug = torch.stack(z_aug, dim=1)
        z_tan = z_aug - z.unsqueeze(1)
        return z, z_tan

    def loss(self, z, z_tan, edge_index, batch_size: Optional[int] = None):
        """

        :param z: [N, d]
        :param z_tan: [N, M, d]
        :param edge_index: [2, E]
        :param batch_size: If not None, meaning that we only consider center nodes in contrastive loss

        :return: loss for each graph batch or all datasets
        """
        triple = search_adjacent_edges(edge_index, self.num_samples)
        if len(triple):
            vi, vj, vk = triple[0], triple[1], triple[2]
            z_tan_i, z_tan_j, z_tan_k = z_tan[vi], z_tan[vj], z_tan[vk]
            pt_matrix_ij = self.parallel_translation(z_tan_i, z_tan_j)    # [T, d, d]
            pt_matrix_jk = self.parallel_translation(z_tan_j, z_tan_k)
            pt_matrix_ik = self.parallel_translation(z_tan_i, z_tan_k)
            pt_matrix = torch.stack([pt_matrix_ij, pt_matrix_jk, pt_matrix_ik], dim=0)    # [3, T, d, d]

            log_r_matrix_ij = self.log_volume_ratio(z_tan_i, z_tan_j)  # [T]
            log_r_matrix_jk = self.log_volume_ratio(z_tan_j, z_tan_k)
            log_r_matrix = torch.stack([log_r_matrix_ij, log_r_matrix_jk], dim=0)  # [2, T]

            geo_loss = self.geo_loss(pt_matrix, log_r_matrix)
        else:
            geo_loss = 0

        if batch_size is not None:
            z = z[:batch_size]
            z_tan = z_tan[:batch_size]
        ptg_loss = self.ptg_loss(z_tan)
        cl_loss = self.contra_loss(z, z.unsqueeze(1) + z_tan)

        return ptg_loss + cl_loss + geo_loss

    def register_prototypes(self, proto_z: torch.Tensor, proto_z_tan: torch.Tensor):
        """
        :param proto_z: [num_datasets, d]
        :param proto_z_tan: [num_datasets, M, d]
        """
        self.register_buffer('proto_z', proto_z)
        self.register_buffer('proto_z_tan', proto_z_tan)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        full_prefix = prefix if prefix else ""
        keys_to_register = []

        if (full_prefix + 'proto_z') in state_dict:
            keys_to_register.append('proto_z')
        if (full_prefix + 'proto_z_tan') in state_dict:
            keys_to_register.append('proto_z_tan')
        if (full_prefix + 'is_global_prototypes_registered') in state_dict:
            keys_to_register.append('is_global_prototypes_registered')

        for key in keys_to_register:
            if not hasattr(self, key):
                self.register_buffer(key, torch.empty_like(state_dict[full_prefix + key], device='cpu'))

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def frozen(self):
        for param in self.parameters():
            param.requires_grad_(False)

    def unfrozen(self):
        for param in self.parameters():
            param.requires_grad_(True)

    @staticmethod
    def knn_graph(h: torch.Tensor, top_k, return_weight: bool = False):
        """
        Construct KNN graph for graph-level datasets.

        :param h: All the graph representations for a graph-level dataset.
        :param top_k: the number of K nearest neighbors.
        :param return_weight: If True, return edge_weight, otherwise, return None.

        :return: edge_index, edge_weight [Torch.Tensor, torch.Tensor]
        """
        if top_k > h.shape[0]:
            top_k = h.shape[0]
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
        :param basis_src: [*, M, d]
        :param basis_dst: [*, M, d]

        :return: PT matrix: torch.Tensor
        """
        g = basis_src @ basis_src.transpose(-1, -2)
        x = basis_src @ basis_dst.transpose(-1, -2)
        P = torch.linalg.solve(g, x)
        return P

    @staticmethod
    def log_volume_ratio(basis_src, basis_dst):
        """
        Volume ratio between two tangent spaces to estimate Ricci Curvature.
        :param basis_src: [*, M, d]
        :param basis_dst: [*, M, d]

        :return: log ratio: torch.Tensor
        """
        vol_src, vol_dst = torch.det(basis_src.transpose(-1, -2) @ basis_src), torch.det(basis_dst.transpose(-1, -2) @ basis_dst)
        abs_vol_src_stable = torch.sqrt(vol_src ** 2 + EPS)
        abs_vol_dst_stable = torch.sqrt(vol_dst ** 2 + EPS)
        log_r = torch.log(abs_vol_dst_stable) - torch.log(abs_vol_src_stable)
        return log_r