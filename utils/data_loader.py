import torch
import numpy as np
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.datasets import AmazonProducts, Reddit, FB15k_237, PPI, MoleculeNet
from ogb.nodeproppred import PygNodePropPredDataset
from utils.data_process import KGNodeInitializer


def load_pretrain_single_graph_data(configs):
    root, data_name = configs.root, configs.data_name
    if data_name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(root=root, name=data_name, transform = T.Compose([T.ToUndirected()]))
        data = dataset[0]
    elif data_name == 'AmazonProducts':
        dataset = AmazonProducts(root)
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
    return data


def load_pretrain_multi_graph_data(configs):
    root, data_name = configs.root, configs.data_name
    if data_name == 'PPI':
        dataset = PPI(root)
    elif data_name in ["PCBA"]:
        dataset = MoleculeNet(root, name=data_name)
    else:
        raise ValueError('Invalid data_name')
    return dataset

if __name__ == '__main__':
    from utils.tools import DotDict
    configs = DotDict({})
    configs.data_name = "PPI"
    configs.root = "../datasets"
    configs.kg_model = "transe"
    configs.kg_batch_size = 1000
    configs.kg_epochs = 500
    data = load_pretrain_single_graph_data(configs)