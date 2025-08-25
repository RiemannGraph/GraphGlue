import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import AmazonProducts, Reddit, FB15k_237, PPI, MoleculeNet
from ogb.nodeproppred import PygNodePropPredDataset
from data.data_process import KGNodeInitializer
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import k_hop_subgraph


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
        results = model.fit(train_data, valid_data, test_data, configs.kg_batch_size, configs.kg_epochs, verbose=True)
        data = FB15k_237(root, split='train')[0]
        data.x = results["node_embeddings"]
    else:
        raise ValueError('Invalid data_name')
    dataset = EgoGraphDataset(data, configs.k_hops, configs.in_dim)
    return dataset


def load_pretrain_multi_graph_data(configs, data_name):
    root = configs.root
    if data_name == 'PPI':
        dataset = PPI(root)
    elif data_name in ["PCBA"]:
        dataset = MoleculeNet(root, name=data_name)
    else:
        raise ValueError('Invalid data_name')
    return dataset


class EgoGraphDataset(Dataset):
    def __init__(self, data: Data, k_hops, in_dim=128):
        super(EgoGraphDataset, self).__init__()
        assert data.num_features >= in_dim, f"hid_dim={in_dim} is too large!"
        self.k_hops = k_hops
        self.in_dim = in_dim
        self.num_nodes = data.num_nodes
        data = self.dim_reduce(data)
        self.data = data

    def dim_reduce(self, data):
        U, S, VT = torch.svd(data.x)
        x_reduced = S.unsqueeze(-1) * VT[: self.in_dim].t()
        data.x = x_reduced
        return data

    def __len__(self):
        return self.num_nodes

    def __getitem__(self, idx):
        subset, edge_index, mapping, _ = k_hop_subgraph(
            idx, self.k_hops, self.data.edge_index, relabel_nodes=True
        )

        return Data(
            x=self.data.x[subset],
            edge_index=edge_index,
            edge_weight=torch.ones_like(edge_index[0]),
            center_node_idx=subset[mapping]
        )