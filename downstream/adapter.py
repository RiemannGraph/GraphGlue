import torch
import torch.nn as nn

from cores.layers import ActivateModule
from cores.loss_funcs import PTGBLoss
from cores.models import RPGraphFM
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
        assert task_type in ["node_cls", "graph_cls", "link_cls"], "the task type must be one of [node_cls, graph_cls, link_cls]"
        self.configs = configs
        self.input_lin = nn.Linear(feature_dim, configs.in_dim)
        self.pretrained_model = pretrained_model
        self.pretrained_model.frozen()
        self.prompt_z = nn.Parameter(torch.eye(configs.hid_dim) + 0.01 * torch.randn(configs.hid_dim, configs.hid_dim))
        self.prompt_z_tan = nn.Parameter(torch.eye(configs.hid_dim) + 0.01 * torch.randn(configs.hid_dim, configs.hid_dim))

        self.align_coef = configs.align_coef
        num_datasets = len(configs.pretrain_single_graph_data) + len(configs.pretrain_multi_graph_data)
        self.gated_func = nn.Sequential(
            nn.Linear(configs.hid_dim, configs.hid_dim, bias=configs.bias),
            nn.Dropout(configs.drop),
            ActivateModule(configs.act_str),
            nn.Linear(configs.hid_dim, num_datasets, bias=configs.bias),
        )
        self.ptg_loss = PTGBLoss(configs.num_generators, configs.temperature)
        self.head = ADAPTERS[task_type](configs.hid_dim, num_cls)

    def forward(self, graph: Data):
        graph.x = self.input_lin(graph.x)
        z, z_tan = self.pretrained_model(graph)
        z = z @ self.prompt_z
        weights = self.gated_func(z).softmax(-1)    # [*, K]
        _, _, proto_z_tan = self.pretrained_model.get_all_prototypes()
        z_tan_align = torch.einsum('ij,jkl->ikl', weights, proto_z_tan.squeeze(1))   # [*, M, d]
        z_tan_adapt = z_tan @ self.prompt_z_tan
        align_loss =  torch.frobenius_norm(z_tan_adapt - z_tan_align, dim=[1, 2]).mean()
        ptgb_loss = self.ptg_loss(z_tan_align)
        loss = align_loss * self.align_coef + ptgb_loss
        return z, z_tan_adapt, loss

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
        src_emb = z[::2]
        dst_emb = z[1::2]
        return self.score_fn(src_emb, dst_emb)


ADAPTERS = {
    'node_cls': NodeClassificationAdapter,
    'graph_cls': GraphClassificationAdapter,
    'link_cls': LinkClassificationAdapter,
}