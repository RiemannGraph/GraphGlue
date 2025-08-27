import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import (
    AmazonProducts, Reddit, FB15k_237, AttributedGraphDataset,
    Planetoid, Amazon, FacebookPagePage,
    WordNet18RR, TUDataset, MoleculeNet
)
from ogb.nodeproppred import PygNodePropPredDataset
from data.data_process import KGNodeInitializer, unify_feature_dimension
from torch_geometric.data import Dataset
from torch_geometric.utils import to_undirected


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
    data.x = unify_feature_dimension(data.x, configs.in_dim)
    if not hasattr(data, "edge_weight"):
        data.edge_weight = torch.ones_like(data.edge_index[0])
    data.edge_index, data.edge_weight = to_undirected(data.edge_index, data.edge_weight, num_nodes=data.num_nodes)
    return data


def load_pretrain_multi_graph_data(configs, data_name):
    root = configs.root
    if data_name in ["PCBA"]:
        dataset = MoleculeNet(root, name=data_name)
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
    elif data_name == "WordNet18RR":
        dataset = WordNet18RR(root, transform=transform)
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


class GraphDataset(Dataset):
    def __init__(self, dataset: Dataset, in_dim=128):
        """

        :param dataset: Graph-level dataset
        :param in_dim: the unified dimension of features as inputs
        """
        super(GraphDataset, self).__init__()
        self.in_dim = in_dim
        self.dataset = dataset

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data = self.dataset[idx]
        data.x = unify_feature_dimension(data.x, self.in_dim)
        if not hasattr(data, "edge_weight"):
            data.edge_weight = torch.ones_like(data.edge_index[0])
        return data