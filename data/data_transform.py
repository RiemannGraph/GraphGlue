from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, Dataset
import torch
from data.data_process import unify_feature_dimension


class UnifyFeatureDims(BaseTransform):
    def __init__(self, uni_dim: int):
        self.uni_dim = uni_dim

    def forward(self, data: Data):
        data.x = unify_feature_dimension(data.x, self.uni_dim)
        return data


class RenameFromRootedEgoNets(BaseTransform):
    """
    Rename the attribute of neighbor-sampled graph from RootedEgoNets.
    """
    def __init__(self):
        super().__init__()

    def forward(self, ego_net_data):
        data = Data()
        batch_graph_nums = ego_net_data.x.shape[0]
        data.batch_graph_nums = batch_graph_nums
        data.x = ego_net_data.x[ego_net_data.n_id]
        data.y = ego_net_data.y[ego_net_data.n_id]
        data.edge_index = ego_net_data.sub_edge_index
        data.edge_weight = ego_net_data.edge_weight[ego_net_data.e_id]
        data.batch = ego_net_data.n_sub_batch
        data.origin_edge_index = ego_net_data.edge_index
        return data