import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import (
    Reddit, AttributedGraphDataset,
    Planetoid, Amazon, FacebookPagePage,
    WordNet18RR, TUDataset, MoleculeNet
)
from data.data_custom import FB15k_237
from ogb.nodeproppred import PygNodePropPredDataset
from data.data_transform import FlattenLabels, UnifyFeatureDims, FewShotLinkSplit, Node2VecEmbedding
from data.data_process import graph_few_shot_splits, link_k_shot_split
from torch_geometric.data import Dataset
from torch_geometric.utils import to_undirected


def load_pretrain_single_graph_data(configs, data_name):
    root = configs.root
    if data_name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(root=root, name=data_name, transform=T.Compose([T.ToUndirected()]))
    elif data_name == 'Computers':
        dataset = Amazon(root, data_name)
    elif data_name == 'Reddit':
        dataset = Reddit(f"{root}/{data_name}")
    elif data_name == 'FB15k_237':
        transform = Node2VecEmbedding(configs.nv_dim, configs.nv_batch_size,
                                      configs.nv_walk_length, configs.nv_context_size,
                                      configs.nv_lr, configs.nv_walks_per_node,
                                      configs.nv_p, configs.nv_q, configs.nv_num_epochs)
        dataset = FB15k_237(f"{root}/{data_name}", split='train', pre_transform=transform)
    elif data_name == 'PPI':
        dataset = AttributedGraphDataset(root, name=data_name.lower())
    else:
        raise ValueError('Invalid data_name')
    data = dataset[0]
    data = UnifyFeatureDims(configs.in_dim)(data)
    if data.edge_weight is None:
        data.edge_weight = torch.ones_like(data.edge_index[0]).float()
    data.edge_index, data.edge_weight = to_undirected(data.edge_index, data.edge_weight, num_nodes=data.num_nodes)
    return data


def load_pretrain_multi_graph_data(configs, data_name):
    root = configs.root
    if data_name in ["PCBA", "HIV"]:
        dataset = MoleculeNet(root, name=data_name, transform=UnifyFeatureDims(configs.in_dim))
    elif data_name in ["PROTEINS", "MUTAG", "ENZYMES"]:
        dataset = TUDataset(root, data_name, transform=UnifyFeatureDims(configs.in_dim))
    else:
        raise ValueError('Invalid data_name')
    dataset = GraphDataset(dataset)
    return dataset


def load_few_shot_single_graph_data(configs, data_name, k_shot, num_splits, num_val=0.1, num_test=0.2):
    root = configs.root
    transform = T.RandomNodeSplit(split='test_rest', num_splits=num_splits,
                                  num_train_per_class=k_shot, num_val=num_val, num_test=num_test)
    if data_name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(root=root, name=data_name, transform=T.Compose([T.ToUndirected(), FlattenLabels(), transform]))
    elif data_name in ["Cora", "CiteSeers", "PubMed"]:
        dataset = Planetoid(root, data_name, transform=transform)
    elif data_name in ["Computers", "Photo"]:
        dataset = Amazon(root, data_name, transform=transform)
    elif data_name == 'Reddit':
        dataset = Reddit(f"{root}/{data_name}", transform=transform)
    elif data_name == "FacebookPagePage":
        dataset = FacebookPagePage(f"{root}/{data_name}", transform=transform)
    elif data_name == 'PPI':
        dataset = AttributedGraphDataset(root, name=data_name.lower(), transform=transform)
    else:
        raise ValueError('Invalid data_name')
    data = dataset[0]
    if data.edge_weight is None:
        data.edge_weight = torch.ones_like(data.edge_index[0]).float()
    data.edge_index, data.edge_weight = to_undirected(data.edge_index, data.edge_weight, num_nodes=data.num_nodes)
    return dataset, data


def load_few_shot_multi_graph_data(configs, data_name, k_shot, num_splits, num_val=0.5, num_test=0.5):
    """Just for single class classification"""
    root = configs.root
    if data_name in ["PROTEINS", "MUTAG", "ENZYMES"]:
        dataset = TUDataset(root, data_name)
    elif data_name in ["PCBA", "HIV"]:
        dataset=  MoleculeNet(root, data_name)
    else:
        raise ValueError('Invalid data_name')
    dataset = GraphDataset(dataset)
    train_mask, val_mask, test_mask = graph_few_shot_splits(dataset, k_shot, num_val, num_splits)
    return dataset, train_mask, val_mask, test_mask


def load_few_shot_link_graph_data(configs, data_name, k_shot, num_splits, num_val=0.1, num_test=0.2):
    root = configs.root
    if data_name == "WordNet18RR":
        transform_split = FewShotLinkSplit(k_shot, num_splits, num_val, num_test)
        transform_x = Node2VecEmbedding(configs.nv_dim, configs.nv_batch_size,
                                      configs.nv_walk_length, configs.nv_context_size,
                                      configs.nv_lr, configs.nv_walks_per_node,
                                      configs.nv_p, configs.nv_q, configs.nv_num_epochs)
        dataset = WordNet18RR(f"{root}/{data_name}", pre_transform=T.Compose([transform_split, transform_x]))
    elif data_name == 'FB15k_237':
        transform_split = FewShotLinkSplit(k_shot, num_splits, num_val, num_test)
        transform_x = Node2VecEmbedding(configs.in_dim, configs.nv_batch_size,
                                      configs.nv_walk_length, configs.nv_context_size,
                                      configs.nv_lr, configs.nv_walks_per_node,
                                      configs.nv_p, configs.nv_q, configs.nv_num_epochs)
        dataset = FB15k_237(f"{root}/{data_name}", split='train', pre_transform=T.Compose([transform_split, transform_x]))
    else:
        raise ValueError('Invalid data_name')
    data = dataset[0]
    if data.edge_weight is None:
        data.edge_weight = torch.ones_like(data.edge_index[0]).float()
    train_mask, val_mask, test_mask = link_k_shot_split(data, k_shot, num_splits, num_val, num_test)
    return dataset, data, (train_mask, val_mask, test_mask)


class GraphDataset(Dataset):
    def __init__(self, dataset: Dataset):
        """

        :param dataset: Graph-level dataset
        """
        super(GraphDataset, self).__init__()
        self.dataset = dataset

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data = self.dataset[idx]
        data.x = data.x.float()
        data.y = data.y.long().reshape(-1)
        if data.edge_weight is None:
            data.edge_weight = torch.ones_like(data.edge_index[0]).float()
        return data