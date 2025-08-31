import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import (
    AmazonProducts, Reddit, FB15k_237, AttributedGraphDataset,
    Planetoid, Amazon, FacebookPagePage,
    WordNet18RR, TUDataset, MoleculeNet
)
from ogb.nodeproppred import PygNodePropPredDataset
from data.data_process import KGNodeInitializer, UnifyFeatureDims
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import coalesce, to_undirected
import numpy as np


def load_pretrain_single_graph_data(configs, data_name):
    root = configs.root
    if data_name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(root=root, name=data_name, transform=T.Compose([T.ToUndirected()]))
        data = dataset[0]
    elif data_name == 'AmazonProducts':
        dataset = AmazonProducts(f"{root}/{data_name}")
        data = dataset[0]
    elif data_name == 'Reddit':
        dataset = Reddit(root)
        data = dataset[0]
    elif data_name == 'FB15k_237':
        device = torch.device('cuda')
        train_data = FB15k_237(f"{root}/{data_name}", split='train')[0]
        valid_data = FB15k_237(f"{root}/{data_name}", split='val')[0]
        test_data = FB15k_237(f"{root}/{data_name}", split='test')[0]
        model = KGNodeInitializer(configs.kg_model, device=device)
        results = model.fit(train_data, valid_data, test_data, configs.in_dim, configs.kg_batch_size, configs.kg_epochs, verbose=True)
        data = FB15k_237(root, split='train')[0]
        data.x = results["node_embeddings"]
    elif data_name == 'PPI':
        dataset = AttributedGraphDataset(root, name=data_name.lower())
        data = dataset[0]
    else:
        raise ValueError('Invalid data_name')
    data = UnifyFeatureDims(configs.in_dim)(data)
    if data.edge_weight is None:
        data.edge_weight = torch.ones_like(data.edge_index[0]).float()
    data.edge_index, data.edge_weight = to_undirected(data.edge_index, data.edge_weight, num_nodes=data.num_nodes)
    return data


def load_pretrain_multi_graph_data(configs, data_name):
    root = configs.root
    if data_name in ["PCBA"]:
        dataset = MoleculeNet(root, name=data_name, transform=UnifyFeatureDims(configs.in_dim))
    else:
        raise ValueError('Invalid data_name')
    dataset = GraphDataset(dataset)
    return dataset


def load_few_shot_single_graph_data(configs, data_name, k_shot, num_splits, num_val=0.1, num_test=0.2):
    root = configs.root
    transform = T.RandomNodeSplit(split='test_rest', num_splits=num_splits,
                                  num_train_per_class=k_shot, num_val=num_val, num_test=num_test)
    if data_name in ["Cora", "CiteSeers", "PubMed"]:
        dataset = Planetoid(root, data_name, transform=transform)
    elif data_name == "Computers":
        dataset = Amazon(root, data_name, transform=transform)
    elif data_name == "FacebookPagePage":
        dataset = FacebookPagePage(f"{root}/{data_name}", transform=transform)
    else:
        raise ValueError('Invalid data_name')
    return dataset


def load_few_shot_multi_graph_data(configs, data_name, k_shot, num_splits, num_val=0.1, num_test=0.2):
    root = configs.root
    if data_name == "Tox21":
        dataset = MoleculeNet(root, data_name)
    elif data_name == "PROTEINS":
        dataset = TUDataset(root, data_name)
    else:
        raise ValueError('Invalid data_name')
    return dataset


def load_few_shot_edge_data(configs, data_name, k_shot, num_splits, num_val=0.1, num_test=0.2):
    """
    :return list of (train_data, val_data, test_data) for each split
    """
    root = configs.root

    if data_name == "WordNet18RR":
        dataset = WordNet18RR(f"{root}/{data_name}")
        data = dataset[0]
    else:
        raise ValueError('Invalid data_name')

    edge_index = data.edge_index  # [2, num_edges]
    edge_type = data.edge_type  # [num_edges,]
    num_relations = int(edge_type.max().item() + 1)

    all_splits = []
    for _ in range(num_splits):
        train_edges = []
        train_edge_types = []
        val_edges = []
        val_edge_types = []
        test_edges = []
        test_edge_types = []

        for rel in range(num_relations):
            mask = (edge_type == rel)
            rel_edges = edge_index[:, mask]
            rel_edge_types = edge_type[mask]

            rel_edges = rel_edges.t().cpu().numpy()
            unique_edges, unique_indices = np.unique(rel_edges, axis=0, return_index=True)
            rel_edges = torch.tensor(rel_edges).t()  # back to tensor
            rel_edge_types = rel_edge_types[unique_indices]

            num_rel_edges = rel_edges.shape[1]
            k = min(k_shot, num_rel_edges)

            remaining_edges = rel_edges
            remaining_types = rel_edge_types

            if k > 0:
                perm = torch.randperm(num_rel_edges)
                train_idx = perm[:k]
                train_edges.append(rel_edges[:, train_idx])
                train_edge_types.append(rel_edge_types[train_idx])

                if num_rel_edges > k:
                    remaining_edges = rel_edges[:, perm[k:]]
                    remaining_types = rel_edge_types[perm[k:]]
                else:
                    remaining_edges = torch.empty((2, 0), dtype=torch.long)
                    remaining_types = torch.empty((0,), dtype=torch.long)

            num_remaining = remaining_edges.shape[1]
            if num_remaining == 0:
                val_edges_rel, test_edges_rel = torch.empty((2, 0)), torch.empty((2, 0))
                val_types_rel, test_types_rel = torch.empty(0), torch.empty(0)
            else:
                val_ratio = num_val / (num_val + num_test)
                val_size = int(num_remaining * val_ratio)

                perm_remaining = torch.randperm(num_remaining)
                val_idx = perm_remaining[:val_size]
                test_idx = perm_remaining[val_size:]

                val_edges_rel = remaining_edges[:, val_idx]
                test_edges_rel = remaining_edges[:, test_idx]
                val_types_rel = remaining_types[val_idx]
                test_types_rel = remaining_types[test_idx]

            val_edges.append(val_edges_rel)
            val_edge_types.append(val_types_rel)
            test_edges.append(test_edges_rel)
            test_edge_types.append(test_types_rel)

        def safe_cat(tensors, dim=1):
            tensors = [t for t in tensors if t.shape[dim] > 0]
            return torch.cat(tensors, dim=dim) if len(tensors) > 0 else torch.empty((2, 0), dtype=torch.long)

        train_edge_index = safe_cat(train_edges, dim=1)
        val_edge_index = safe_cat(val_edges, dim=1)
        test_edge_index = safe_cat(test_edges, dim=1)

        train_edge_type = safe_cat(train_edge_types, dim=0)
        val_edge_type = safe_cat(val_edge_types, dim=0)
        test_edge_type = safe_cat(test_edge_types, dim=0)

        train_data = Data(
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            edge_label_index=train_edge_index,
            edge_label=train_edge_type,
        )

        val_data = Data(
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            edge_label_index=val_edge_index,
            edge_label=val_edge_type,
        )

        test_data = Data(
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            edge_label_index=test_edge_index,
            edge_label=test_edge_type,
        )
        all_splits.append((train_data, val_data, test_data))
    return all_splits


class GraphDataset(Dataset):
    def __init__(self, dataset: Dataset):
        """

        :param dataset: Graph-level dataset
        :param in_dim: the unified dimension of features as inputs
        """
        super(GraphDataset, self).__init__()
        self.dataset = dataset

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data = self.dataset[idx]
        data.x = data.x.float()
        if data.edge_weight is None:
            data.edge_weight = torch.ones_like(data.edge_index[0]).float()
        return data