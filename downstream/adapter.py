import torch
import torch.nn as nn
from torch_geometric.data import Data
from cores.models import RPGraphFM
from torch_geometric.data import Data, Batch


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
        assert pretrained_model.is_global_representation_registered, \
            "the global representation must be stored in pretraining phase."
        assert task_type in configs.task_types, "the task type must be stored in configs.task_types."
        self.configs = configs
        self.input_lin = nn.Linear(feature_dim, self.configs.in_dim)
        self.pretrained_model = pretrained_model
        self.prompt = nn.Parameter(torch.empty(configs.hid_dim, configs.hid_dim))
        self.align_coef = configs.align_coef
        if task_type == "node_cls":
            self.head = NodeClassificationAdapter(configs.hid_dim, num_cls)
        elif task_type == "graph_cls":
            self.head = GraphClassificationAdapter(configs.hid_dim, num_cls)
        elif task_type == "edge_cls":
            self.head = LinkClassificationAdapter(configs.hid_dim, num_cls)

    def forward(self, graph: Data, batch_graph_nums: int = None):
        if batch_graph_nums is None:
            if hasattr(graph, "batch_graph_nums"):
                batch_graph_nums = graph.batch_graph_nums
            else:
                batch_graph_nums = graph.batch_size
        z, z_tan = self.pretrained_model(graph, batch_graph_nums)
        z_tan = z_tan @ self.prompt
        return z, z_tan

    def predict(self, z: torch.Tensor, graph: Data):
        z = self.head(z, graph)
        return z

    def loss(self, z_tan):
        return self.prompt_loss(z_tan, self.pretrained_model.global_tan, self.align_coef)

    @staticmethod
    def prompt_loss(z_tan_tgt, z_tan_src, align_coef: float):
        return align_coef * torch.frobenius_norm(z_tan_src.unsqueeze(0) - z_tan_tgt, dim=[1, 2]).mean()


class NodeClassificationAdapter(nn.Module):
    def __init__(self, hid_dim: int, num_classes: int):
        super(NodeClassificationAdapter, self).__init__()
        self.head = nn.Linear(hid_dim, num_classes)

    def forward(self, z: torch.Tensor, graph: Data):
        return self.head(z[: graph.batch_size])  # Only use labeled nodes in few-shot


class GraphClassificationAdapter(nn.Module):
    def __init__(self, hid_dim: int, num_classes: int):
        super(GraphClassificationAdapter, self).__init__()
        self.head = nn.Linear(hid_dim, num_classes)

    def forward(self, z: torch.Tensor, graph: Data):
        return self.head(z)


class LinkClassificationAdapter(nn.Module):
    """
    For knowledge graph link prediction (edge classification / triple scoring)
    Using dot product or bilinear scoring.
    """
    def __init__(self, hid_dim: int, num_classes: int):
        super(LinkClassificationAdapter, self).__init__()
        self.score_fn = nn.Bilinear(hid_dim, hid_dim, num_classes)

    def forward(self, z: torch.Tensor, graph: Data):
        edge_index = graph.edge_index
        src_emb = z[edge_index[0]]
        dst_emb = z[edge_index[1]]
        return self.score_fn(src_emb, dst_emb)


ADAPTERS = {
    'node_cls': NodeClassificationAdapter,
    'graph_cls': GraphClassificationAdapter,
    'link_pred': LinkClassificationAdapter,
}