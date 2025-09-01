import torch
import torch.nn as nn
from cores.loss_funcs import PTGBLoss
from cores.models import RPGraphFM, FeedForwardLayer
from torch_geometric.data import Data


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
        assert 'proto_z' in pretrained_model._buffers and 'proto_z_tan' in pretrained_model._buffers, \
            "the global prototype must be stored in pretraining phase."
        assert task_type in ["node_cls", "graph_cls", "link_cls"], "the task type must be one of [node_cls, graph_cls, link_cls]"
        self.configs = configs
        self.input_lin = nn.Linear(feature_dim, self.configs.in_dim)
        self.pretrained_model = pretrained_model
        self.prompt = nn.Parameter(torch.empty(configs.hid_dim, configs.hid_dim))
        nn.init.kaiming_normal_(self.prompt.data)
        self.align_coef = configs.align_coef
        num_datasets = len(configs.pretrain_single_graph_data) + len(configs.pretrain_multi_graph_data)
        self.gated_func = FeedForwardLayer(configs.hid_dim, configs.hid_dim, num_datasets,
                                           configs.bias, configs.act_str, configs.drop)
        self.ptg_loss = PTGBLoss(configs.num_generators, configs.temperature)
        self.head = ADAPTERS[task_type](configs.hid_dim, num_cls)

    def forward(self, graph: Data, batch_graph_nums: int = None):
        if batch_graph_nums is None:
            if hasattr(graph, "batch_graph_nums"):
                batch_graph_nums = graph.batch_graph_nums
            else:
                batch_graph_nums = graph.batch_size
        graph = graph.clone()
        graph.x = self.input_lin(graph.x)
        z, z_tan = self.pretrained_model(graph, batch_graph_nums)
        weights = self.gated_func(z)    # [*, K]
        z_tan_align = torch.einsum('ij,jkl->ikl', weights, self.pretrained_model.proto_z_tan)   # [*, M, d]
        z_tan_adapt = z_tan @ self.prompt
        align_loss = self.align_coef * torch.frobenius_norm(z_tan_adapt - z_tan_align, dim=[1, 2]).mean()
        ptgb_loss = self.ptg_loss(z_tan_align)
        return z, z_tan_adapt, align_loss + ptgb_loss

    def predict(self, z: torch.Tensor, graph: Data):
        z = self.head(z, graph)
        return z


class NodeClassificationAdapter(nn.Module):
    def __init__(self, hid_dim: int, num_classes: int):
        super(NodeClassificationAdapter, self).__init__()
        self.head = nn.Linear(hid_dim, num_classes)

    def forward(self, z: torch.Tensor, graph: Data):
        return self.head(z)  # Only use labeled nodes in few-shot


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