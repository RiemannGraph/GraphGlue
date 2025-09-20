import torch
import torch.nn as nn
import torch.nn.functional as F
from cores.layers import ActivateModule
from cores.loss_funcs import PTGBLoss
from cores.models import RPGraphFM
from torch_geometric.data import Data

from utils.math import parallel_translation, diagonal_metric, matrix_log_diag


class RPGPrompt(nn.Module):
    def __init__(self, configs, feature_dim, pretrained_model: RPGraphFM, task_type: str, num_cls: int):
        """

        :param configs: PretrainConfig
        :param feature_dim:
        :param pretrained_model:
        :param task_type: [node_cls, graph_cls, edge_cls]
        :param num_cls: classes number
        """
        super(RPGPrompt, self).__init__()
        assert task_type in ["node_cls", "graph_cls", "link_cls"], "the task type must be one of [node_cls, graph_cls, link_cls]"
        self.configs = configs
        self.input_lin = nn.Linear(feature_dim, configs.in_dim)
        self.pretrained_model = pretrained_model
        self.pretrained_model.frozen()
        self.prompt_z = nn.Parameter(torch.empty(configs.hid_dim, configs.hid_dim))
        nn.init.orthogonal_(self.prompt_z.data)

        self.align_coef = configs.align_coef
        num_datasets = len(configs.pretrain_single_graph_data) + len(configs.pretrain_multi_graph_data)
        self.gated_func = nn.Sequential(
            nn.Linear(configs.hid_dim + configs.num_generators, configs.hid_dim, bias=configs.bias),
            nn.Dropout(configs.drop),
            ActivateModule(configs.act_str),
            nn.Linear(configs.hid_dim, num_datasets, bias=configs.bias),
        )
        self.head = ADAPTERS[task_type](configs.hid_dim + configs.num_generators, num_cls, configs.drop)

    def forward(self, graph: Data):
        graph.x = self.input_lin(graph.x)
        z, z_tan = self.pretrained_model(graph)

        z_adapt = z @ self.prompt_z
        z_tan_adapt = z_tan @ self.prompt_z
        log_metric_adapt = matrix_log_diag(diagonal_metric(z_tan_adapt))
        z_log_metric_adapt = torch.concat([z_adapt, log_metric_adapt], dim=-1)

        weights = self.gated_func(z_log_metric_adapt).softmax(-1)    # [*, K]

        _, proto_z, proto_metric = self.pretrained_model.get_all_prototypes() # [K, M]
        log_metric_align = weights @ matrix_log_diag(proto_metric)   # [*, M]

        dist = torch.sum((z_adapt.unsqueeze(1) - proto_z.unsqueeze(0)) ** 2, dim=-1).mean()   # [N, K]
        align_loss =  (torch.norm(log_metric_align - log_metric_adapt, dim=-1, p=2)**2).mean() + dist
        loss = align_loss * self.align_coef
        pred = self.head(z_log_metric_adapt, graph)
        return pred, loss

    @torch.no_grad()
    def transfer_metric(self):
        Q = self.prompt_z.data
        d = Q.shape[-1]
        U, S, Vt = torch.svd(Q)
        RS = torch.frobenius_norm(U @ Vt - torch.eye(d, device=Q.device), dim=[-1, -2])
        SS = torch.norm(S - torch.ones_like(S), dim=-1)
        return RS.item(), SS.item()


class NodeClassificationAdapter(nn.Module):
    def __init__(self, hid_dim: int, num_classes: int, drop: float = 0.2):
        super(NodeClassificationAdapter, self).__init__()
        self.head = nn.Linear(hid_dim, num_classes)
        self.drop = nn.Dropout(drop)

    def forward(self, z: torch.Tensor, graph: Data):
        z = self.drop(z)
        return self.head(z)  # Only use labeled nodes in few-shot


class GraphClassificationAdapter(nn.Module):
    def __init__(self, hid_dim: int, num_classes: int, drop: float = 0.2):
        super(GraphClassificationAdapter, self).__init__()
        self.head = nn.Linear(hid_dim, num_classes)
        self.drop = nn.Dropout(drop)

    def forward(self, z: torch.Tensor, graph: Data):
        z = self.drop(z)
        return self.head(z)


class LinkClassificationAdapter(nn.Module):
    """
    For knowledge graph link prediction (edge classification / triple scoring)
    Using dot product or bilinear scoring.
    """
    def __init__(self, hid_dim: int, num_classes: int, drop: float = 0.2):
        super(LinkClassificationAdapter, self).__init__()
        self.score_fn = nn.Bilinear(hid_dim, hid_dim, num_classes)
        self.drop = nn.Dropout(drop)

    def forward(self, z: torch.Tensor, graph: Data):
        z = self.drop(z)
        src_emb = z[::2]
        dst_emb = z[1::2]
        src_emb = F.normalize(src_emb, p=2, dim=1)
        dst_emb = F.normalize(dst_emb, p=2, dim=1)
        return self.score_fn(src_emb, dst_emb)


ADAPTERS = {
    'node_cls': NodeClassificationAdapter,
    'graph_cls': GraphClassificationAdapter,
    'link_cls': LinkClassificationAdapter,
}