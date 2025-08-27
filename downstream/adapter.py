import torch
import torch.nn as nn
from torch_geometric.data import Data
from cores.models import RPGraphFM
from torch_geometric.data import Data, Batch


class RPGPrompt(nn.Module):
    def __init__(self, configs, feature_dim, pretrained_model: RPGraphFM):
        super(RPGPrompt, self).__init__()
        assert pretrained_model.is_global_representation_registered, \
            "the global representation must be stored in pretraining phase."
        self.configs = configs
        self.input_lin = nn.Linear(feature_dim, self.configs.in_dim)
        self.pretrained_model = pretrained_model
        self.prompt = nn.Parameter(torch.empty(configs.hid_dim, configs.hid_dim))
        self.align_coef = configs.align_coef

    def forward(self, graph: Data, batch_graph_nums: int = None):
        if batch_graph_nums is None:
            if hasattr(graph, "batch_graph_nums"):
                batch_graph_nums = graph.batch_graph_nums
            else:
                batch_graph_nums = graph.batch_size
        z, z_tan = self.pretrained_model(graph, batch_graph_nums)
        z_tan = z_tan @ self.prompt
        return z, z_tan

    def loss(self, z_tan):
        return self.prompt_loss(z_tan, self.pretrained_model.global_tan, self.align_coef)

    @staticmethod
    def prompt_loss(z_tan_tgt, z_tan_src, align_coef: float):
        return align_coef * torch.frobenius_norm(z_tan_src - z_tan_tgt, dim=[1, 2]).mean()


class NodeClassificationAdapter(nn.Module):
    def __init__(self, hid_dim: int, num_classes: int):
        super(NodeClassificationAdapter, self).__init__()
        self.head = nn.Linear(hid_dim, num_classes)

    def forward(self, z: torch.Tensor, data: Data):
        return self.head(z[data.train_mask])  # Only use labeled nodes in few-shot


class GraphClassificationAdapter(nn.Module):
    def __init__(self, hid_dim: int, num_classes: int):
        super(GraphClassificationAdapter, self).__init__()
        self.head = nn.Linear(hid_dim, num_classes)

    def forward(self, z: torch.Tensor, batch_graph_nums: int):
        # z: [B, D] or [N, D] -> mean pooling over nodes to get graph-level
        z_graph = z.view(batch_graph_nums, -1, z.size(-1)).mean(dim=1)  # [B, D]
        return self.head(z_graph)


class LinkClassificationAdapter(nn.Module):
    """
    For knowledge graph link prediction (edge classification / triple scoring)
    Using dot product or bilinear scoring.
    """
    def __init__(self, hid_dim: int, num_classes: int):
        super(LinkClassificationAdapter, self).__init__()
        self.score_fn = nn.Bilinear(hid_dim, hid_dim, num_classes)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor):
        src_emb = z[edge_index[0]]
        dst_emb = z[edge_index[1]]
        return self.score_fn(src_emb, dst_emb)


ADAPTERS = {
    'node_cls': NodeClassificationAdapter,
    'graph_cls': GraphClassificationAdapter,
    'link_pred': LinkClassificationAdapter,
}