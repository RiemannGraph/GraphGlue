import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import AmazonProducts, Reddit, FB15k_237, PPI, MoleculeNet
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
        train_data = FB15k_237(root, split='train')[0]
        valid_data = FB15k_237(root, split='val')[0]
        test_data = FB15k_237(root, split='test')[0]
        model = KGNodeInitializer(configs.kg_model, device=device)
        results = model.fit(train_data, valid_data, test_data, configs.in_dim, configs.kg_batch_size, configs.kg_epochs, verbose=True)
        data = FB15k_237(root, split='train')[0]
        data.x = results["node_embeddings"]
    else:
        raise ValueError('Invalid data_name')
    data.x = unify_feature_dimension(data.x, configs.in_dim)
    if not hasattr(data, "edge_weight"):
        data.edge_weight = torch.ones_like(data.edge_index[0])
    data.edge_index, data.edge_weight = to_undirected(data.edge_index, data.edge_weight, num_nodes=data.num_nodes)
    return data


def load_pretrain_multi_graph_data(configs, data_name):
    root = configs.root
    if data_name == 'PPI':
        dataset = PPI(root)
    elif data_name in ["PCBA"]:
        dataset = MoleculeNet(root, name=data_name)
    else:
        raise ValueError('Invalid data_name')
    dataset = GraphDataset(dataset)
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