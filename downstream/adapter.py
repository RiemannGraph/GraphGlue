import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

from cores.models import RPGraphFM
from torch_geometric.data import Data

from utils import search_triangles
from utils.math import diagonal_metric, matrix_log_diag, knn_graphs


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
        self.head = ADAPTERS[task_type](configs.hid_dim + configs.num_generators, num_cls, configs.drop)

    def forward(self, graph: Data):
        graph.x = self.input_lin(graph.x)
        z, z_tan = self.pretrained_model(graph)

        z_adapt = z @ self.prompt_z
        z_tan_adapt = z_tan @ self.prompt_z
        metric_adapt = diagonal_metric(z_tan_adapt)
        log_metric_adapt = matrix_log_diag(metric_adapt)
        z_log_metric_adapt = torch.concat([z_adapt, log_metric_adapt], dim=-1)

        _, proto_z, proto_metric = self.pretrained_model.get_all_prototypes() # [K, M]

        loss = self.align_coef * self.transfer_metric(z_adapt, metric_adapt, proto_z, proto_metric)
        pred = self.head(z_log_metric_adapt, graph)
        return pred, loss

    def transfer_metric(self, z, metric, z_proto, proto_metric):
        N = z.shape[0]
        K = z_proto.shape[0]
        weights = z @ z_proto.t()   # [N, K]
        knn_edge_index, _ = knn_graphs(weights, 3, return_weight=True, is_to_undirected=False)
        src, dst = knn_edge_index[0], knn_edge_index[1]
        dst += N
        knn_edge_index = to_undirected(torch.stack([src, dst], dim=0), num_nodes=N + K)
        proto_idx = torch.arange(K).to(z.device) + N
        proto_src = proto_idx.unsqueeze(1).expand(-1, K).reshape(-1)
        proto_dst = proto_idx.unsqueeze(0).expand(K, -1).reshape(-1)
        mask = proto_src != proto_dst
        proto_edge_index = torch.stack([proto_src[mask], proto_dst[mask]], dim=0)  # shape: [2, K*(K-1)]
        edge_index = torch.concat([knn_edge_index, proto_edge_index], dim=-1)
        paths = search_triangles(edge_index, num_path_samples=1000)
        geo_loss = self.pretrained_model.geo_loss_from_metric(torch.concat([metric, proto_metric], dim=0), paths[0])
        return geo_loss


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