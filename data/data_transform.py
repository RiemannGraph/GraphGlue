from typing import Optional

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, Dataset
import torch
from data.data_process import unify_feature_dimension, KGNodeInitializer, link_k_shot_split


class UnifyFeatureDims(BaseTransform):
    def __init__(self, uni_dim: int):
        super().__init__()
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
        if ego_net_data.y is not None:
            data.y = ego_net_data.y[ego_net_data.n_id]
        if ego_net_data.edge_type is not None:
            data.edge_type = ego_net_data.edge_type[ego_net_data.e_id]
        data.edge_index = ego_net_data.sub_edge_index
        data.edge_weight = ego_net_data.edge_weight[ego_net_data.e_id]
        data.batch = ego_net_data.n_sub_batch
        data.origin_edge_index = ego_net_data.edge_index
        data.n_id = ego_net_data.n_id
        return data


class FewShotLinkSplit(BaseTransform):
    def __init__(self, k_shot, num_splits, num_val=0.1, num_test=0.2):
        super().__init__()
        self.k_shot = k_shot
        self.num_splits = num_splits
        self.num_val = num_val
        self.num_test = num_test

    def forward(self, data):
        train_mask, val_mask, test_mask = link_k_shot_split(data,
                                                            self.k_shot, self.num_splits,
                                                            self.num_val, self.num_test)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data


class InitKGNodeFeatures(BaseTransform):
    def __init__(self, kg_model: str, embed_dim: int,
                 batch_size: int, epochs: int, device: str = "cuda",
                 train_data: Optional[Data] = None,
                 val_data: Optional[Data] = None,
                 test_data: Optional[Data] = None,):
        super().__init__()
        self.kg_model = kg_model
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def forward(self, data: Data):
        model = KGNodeInitializer(self.kg_model, device=self.device)
        train_data = Data(
            edge_index=data.edge_index[:, data.train_mask],
            edge_type=data.edge_type[data.train_mask],
            num_nodes=data.num_nodes
        ) if self.train_data is None else self.train_data
        valid_data = Data(
            edge_index=data.edge_index[:, data.val_mask],
            edge_type=data.edge_type[data.val_mask],
            num_nodes=data.num_nodes
        ) if self.val_data is None else self.val_data
        test_data = Data(
            edge_index=data.edge_index[:, data.test_mask],
            edge_type=data.edge_type[data.test_mask],
            num_nodes=data.num_nodes
        ) if self.test_data is None else self.test_data
        results = model.fit(train_data, valid_data, test_data, self.embed_dim, self.batch_size, self.epochs,
                            verbose=True)
        data.x = results["node_embeddings"].detach().cpu()
        return data