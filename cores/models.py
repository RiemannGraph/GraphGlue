import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_mean
from cores.layers import NormModule, FeedForwardLayer, GNNLayer
from cores.loss_funcs import PTGBLoss, ContrastiveLoss, GeometricPersistLoss
from typing import List, Optional, Dict, Tuple, Any, Mapping
import re

EPS = 1e-6


class PTGB(nn.Module):
    def __init__(self, num_generators, hid_dim, att_dim):
        super(PTGB, self).__init__()
        self.num_generators = num_generators
        self.generators = nn.Parameter(torch.empty(num_generators, hid_dim))
        nn.init.orthogonal_(self.generators.data)
        self.att_proj = nn.Linear(hid_dim, att_dim)

    def forward(self, x, edge_index, edge_weight, batch, batch_size):
        """

        :param x: [N, d]
        :param edge_index: [2, E]
        :param edge_weight: [E,]
        :param batch: [N]
        :param batch_size: mini-batch size

        :return: List[Data]
        """
        N = x.shape[0]

        weights = torch.sigmoid(self.att_proj(self.generators) @ self.att_proj(x).t())  # [M, N]

        add_batch = torch.arange(batch_size, device=x.device)
        new_batch = torch.concat([batch, add_batch], dim=0)

        counts = torch.bincount(batch)
        add_edge_src = torch.arange(N, N + batch_size, device=x.device).repeat_interleave(counts)
        add_edge_dst = torch.arange(N, device=x.device)
        add_edge_index = torch.stack([add_edge_src, add_edge_dst], dim=0)
        new_edge_index = torch.concat([edge_index, add_edge_index], dim=-1)
        aug_graphs = []
        for i in range(self.num_generators):
            xp = torch.concat([x, self.generators[i: i + 1].repeat(batch_size, 1)], dim=0)
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
        self.num_generators = configs.num_generators
        self.input_lin = nn.Linear(configs.in_dim, configs.hid_dim)
        self.ptg_bank = PTGB(configs.num_generators, configs.hid_dim, configs.att_dim)
        self.encoder = PooLedSubgraphGNN(configs.conv_name, configs.n_layers,
                                         configs.hid_dim, configs.hid_dim,
                                         configs.normalize, configs.bias,
                                         configs.norm_str, configs.act_str, configs.drop)
        datasets_list = configs.pretrain_single_graph_data + configs.pretrain_multi_graph_data
        self.prototype_manager = RiemannianPrototypeManager(datasets_list, configs.hid_dim, configs.num_generators,
                                                            configs.ema_alpha, configs.temperature)
        self.ptg_loss = PTGBLoss(configs.num_generators, configs.temperature)
        self.contra_loss = ContrastiveLoss(configs.temperature)
        self.geo_loss = GeometricPersistLoss(configs.geo_regular_coef)

    def forward(self, graph: Data):
        """

        :param graph: 1) Feature dimension is unified. 2) BatchData

        :return: node/graph embedding, tangent vectors [torch.Tensor, torch.Tensor] with shape [N, d] [N, M, d]
        """

        x, edge_index, edge_weight, batch = graph.x, graph.edge_index, graph.edge_weight, graph.batch
        x = self.input_lin(x)
        z = self.encoder(x, edge_index, edge_weight, batch)

        aug_graphs = self.ptg_bank(x, graph.edge_index, graph.edge_weight, graph.batch, graph.batch_size)
        z_aug = []
        for aug_graph in aug_graphs:
            tan = self.encoder(aug_graph.x, aug_graph.edge_index, aug_graph.edge_weight, aug_graph.batch)
            z_aug.append(tan)
        z_aug = torch.stack(z_aug, dim=1)
        z_tan = z_aug - z.unsqueeze(1)
        return z, z_tan

    def local_struct_loss(self, z, z_tan):
        ptg_loss = self.ptg_loss(z_tan)
        cl_loss = self.contra_loss(z, z.unsqueeze(1) + z_tan)
        return ptg_loss + cl_loss

    def refine_struct_loss(self, z_tan, triple_paths):
        """

        :param z_tan: [N, M, d]
        :param triple_paths: [3, num_paths]

        :return: loss for each graph batch or all datasets
        """
        if triple_paths.numel() > 0:
            vi, vj, vk = triple_paths[0], triple_paths[1], triple_paths[2]
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
            geo_loss = torch.zeros(1, device=z_tan.device, dtype=z_tan.dtype, requires_grad=True).squeeze()

        return geo_loss

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        proto_z_pattern = re.compile(r'^prototype_manager\.proto_z_(?!tan_)([a-zA-Z0-9_]+)$')
        proto_z_tan_pattern = re.compile(r'^prototype_manager\.proto_z_tan_([a-zA-Z0-9_]+)$')
        datasets_to_register = set()

        for key in state_dict.keys():
            match_z = proto_z_pattern.match(key)
            match_tan = proto_z_tan_pattern.match(key)

            if match_z:
                datasets_to_register.add(match_z.group(1))  # e.g., 'Computers'
            elif match_tan:
                datasets_to_register.add(match_tan.group(1))  # e.g., 'Computers'

        for safe_name in datasets_to_register:
            z_key = f'prototype_manager.proto_z_{safe_name}'
            tan_key = f'prototype_manager.proto_z_tan_{safe_name}'

            if not hasattr(self.prototype_manager, f'proto_z_{safe_name}'):
                z_tensor = state_dict[z_key].clone()
                self.prototype_manager.register_buffer(f'proto_z_{safe_name}', z_tensor)

            if not hasattr(self.prototype_manager, f'proto_z_tan_{safe_name}'):
                tan_tensor = state_dict[tan_key].clone()
                self.prototype_manager.register_buffer(f'proto_z_tan_{safe_name}', tan_tensor)

        super().load_state_dict(state_dict, strict=strict)
        self.prototype_manager.rebuild_cache_from_buffers()

    @torch.no_grad()
    def update_prototype(self, z: torch.Tensor, z_tan: torch.Tensor, data_name_map: torch.Tensor):
        z = z.detach()
        z_tan = z_tan.detach()
        self.prototype_manager.update_prototype(z, z_tan, data_name_map)

    def get_all_prototypes(self):
        return self.prototype_manager.get_all_prototypes()

    def prototype_loss(self, z: torch.Tensor, data_name_map: torch.Tensor):
        return self.prototype_manager.loss(z, data_name_map)

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


class RiemannianPrototypeManager(nn.Module):
    """
    EMA Riemannian prototype manager and updater.

    Manages per-dataset prototypes (z and z_tan) with EMA updates.
    Supports contrastive loss between node embeddings and prototypes.
    """
    def __init__(self, datasets_list: List[str], hid_dim: int, num_generators: int, ema_alpha: float = 0.99, temperature: float = 1.0):
        super().__init__()
        self.datasets_list = datasets_list
        self.hid_dim = hid_dim
        self.num_generators = num_generators
        self.ema_alpha = ema_alpha
        self.temperature = temperature

        # Runtime caches (not saved in state_dict)
        self._proto_z_dict: Dict[str, torch.Tensor] = {}        # dataset_name -> tensor (on device)
        self._proto_z_tan_dict: Dict[str, torch.Tensor] = {}    # dataset_name -> tensor (on device)
        self.prototype_keys: List[str] = []  # ordered list of dataset names

        # For safety: keep a mapping from sanitized name to original
        self._sanitized_to_original: Dict[str, str] = {}

    @torch.no_grad()
    def update_prototype(self, z: torch.Tensor, z_tan: torch.Tensor, data_name_map: torch.Tensor):
        """
        Update or initialize prototype for a dataset using EMA.
        """
        dataset_idx = torch.unique(data_name_map).cpu().numpy()
        z_mean = scatter_mean(z, data_name_map, dim=0)
        z_tan_mean = scatter_mean(z_tan, data_name_map, dim=0)
        for i, dataset_name in enumerate([self.datasets_list[idx] for idx in dataset_idx]):
            if dataset_name not in self._proto_z_dict:
                self._register_new_prototype(dataset_name, z_mean, z_tan_mean)
            else:
                alpha = self.ema_alpha
                proto_z = self._proto_z_dict[dataset_name]
                proto_z_tan = self._proto_z_tan_dict[dataset_name]

                # In-place EMA update
                proto_z.copy_(alpha * proto_z + (1 - alpha) * z_mean[i: i+1])
                proto_z_tan.copy_(alpha * proto_z_tan + (1 - alpha) * z_tan_mean[i: i+1])

    def _register_new_prototype(self, dataset_name: str, z_mean: torch.Tensor, z_tan_mean: torch.Tensor):
        """
        Register a new prototype as buffer and update caches.
        """
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', dataset_name)

        # Clone and detach
        p_z = z_mean.detach().clone()
        p_z_tan = z_tan_mean.detach().clone()

        # Register as persistent buffers
        self.register_buffer(f'proto_z_{safe_name}', p_z)
        self.register_buffer(f'proto_z_tan_{safe_name}', p_z_tan)

        # Cache original name -> tensor
        # Note: getattr is safe here because we just registered it
        self._proto_z_dict[dataset_name] = getattr(self, f'proto_z_{safe_name}')
        self._proto_z_tan_dict[dataset_name] = getattr(self, f'proto_z_tan_{safe_name}')

        if dataset_name not in self.prototype_keys:
            self.prototype_keys.append(dataset_name)
            self._sanitized_to_original[safe_name] = dataset_name

    def get_prototype(self, dataset_name: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get prototype for a specific dataset.
        """
        return (
            self._proto_z_dict.get(dataset_name),
            self._proto_z_tan_dict.get(dataset_name)
        )

    def get_all_prototypes(self) -> Tuple[List[str], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get all prototypes as stacked tensors.
        Returns:
            (names, all_z: [N, d], all_z_tan: [N, d])
        """
        names = [name for name in self.datasets_list if name in self.prototype_keys]
        all_z = torch.stack([self._proto_z_dict[name] for name in names], dim=0)
        all_z_tan = torch.stack([self._proto_z_tan_dict[name] for name in names], dim=0)
        return names, all_z, all_z_tan

    def loss(self, z: torch.Tensor, data_name_map: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between node embeddings and all prototypes.
        """
        names, all_proto_z, _ = self.get_all_prototypes()

        all_proto_z = all_proto_z.to(z.device)
        sim = torch.mm(z, all_proto_z.t()) / self.temperature

        loss = F.cross_entropy(sim, data_name_map)
        return loss

    @torch.no_grad()
    def rebuild_cache_from_buffers(self):
        """
        Rebuild _proto_z_dict and prototype_keys from registered buffers.
        Called by parent module after loading state_dict.
        """
        self._proto_z_dict.clear()
        self._proto_z_tan_dict.clear()
        self.prototype_keys.clear()
        self._sanitized_to_original.clear()

        prefix = 'proto_z_'
        tan_prefix = 'proto_z_tan_'

        for name, buffer in self.named_buffers():
            if name.startswith(prefix) and not name.startswith(tan_prefix):
                safe_name = name[len(prefix):]
                try:
                    # Try to recover original name from previous mapping
                    original_name = self._sanitized_to_original.get(safe_name, safe_name)
                except:
                    original_name = safe_name  # fallback

                # Get both buffers
                z_buf = getattr(self, name)
                tan_name = f'{tan_prefix}{safe_name}'
                if hasattr(self, tan_name):
                    z_tan_buf = getattr(self, tan_name)
                else:
                    raise RuntimeError(f"Missing tangent prototype buffer: {tan_name}")

                self._proto_z_dict[original_name] = z_buf
                self._proto_z_tan_dict[original_name] = z_tan_buf
                if original_name not in self.prototype_keys:
                    self.prototype_keys.append(original_name)
                    self._sanitized_to_original[safe_name] = original_name

    def extra_repr(self) -> str:
        return (f"datasets_list={self.datasets_list}, hid_dim={self.hid_dim}, num_generators={self.num_generators}, "
                f"ema_alpha={self.ema_alpha}, temperature={self.temperature}")