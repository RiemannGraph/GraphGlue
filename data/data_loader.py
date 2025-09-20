from typing import Optional, List, Dict
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import (
    Reddit, AttributedGraphDataset,
    Planetoid, Amazon, FacebookPagePage,
    WordNet18RR, TUDataset, MoleculeNet
)
from torch_geometric.loader.dataloader import Collater
from torch_geometric.loader import NeighborSampler
from data.data_custom import FB15k_237
from ogb.nodeproppred import PygNodePropPredDataset
from data.data_transform import FlattenLabels, UnifyFeatureDims, FewShotLinkSplit, Node2VecEmbedding
from data.data_process import graph_few_shot_splits, link_k_shot_split
from torch_geometric.data import Dataset, Data, Batch


def load_pretrain_single_graph_data(configs, data_name: str):
    root = configs.root
    transform = T.Compose([
        T.ToUndirected(),
        UnifyFeatureDims(configs.in_dim)
    ])
    if data_name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(root=root, name=data_name, transform=transform)
    elif data_name in ["Computers", "Photo"]:
        dataset = Amazon(root, data_name, transform=transform)
    elif data_name == 'Reddit':
        dataset = Reddit(f"{root}/{data_name}", transform=transform)
    elif data_name == "FacebookPagePage":
        dataset = FacebookPagePage(f"{root}/{data_name}", transform=transform)
    elif data_name == 'FB15k_237':
        kg_transform = Node2VecEmbedding(configs.nv_dim, configs.nv_batch_size,
                                      configs.nv_walk_length, configs.nv_context_size,
                                      configs.nv_lr, configs.nv_walks_per_node,
                                      configs.nv_p, configs.nv_q, configs.nv_num_epochs)
        dataset = FB15k_237(f"{root}/{data_name}", split='train', pre_transform=kg_transform, transform=transform)
    elif data_name == "WordNet18RR":
        kg_transform = Node2VecEmbedding(configs.nv_dim, configs.nv_batch_size,
                                      configs.nv_walk_length, configs.nv_context_size,
                                      configs.nv_lr, configs.nv_walks_per_node,
                                      configs.nv_p, configs.nv_q, configs.nv_num_epochs)
        dataset = WordNet18RR(f"{root}/{data_name}", pre_transform=kg_transform, transform=transform)
    elif data_name == 'PPI':
        dataset = AttributedGraphDataset(root, name=data_name.lower(), transform=transform)
    elif data_name in ["Cora", "CiteSeers", "PubMed"]:
        dataset = Planetoid(root, data_name, transform=transform)
    else:
        raise ValueError('Invalid data_name')
    data = dataset[0]
    return data


def load_pretrain_multi_graph_data(configs, data_name: str, data_name_map: int):
    root = configs.root
    if data_name in ["PCBA", "HIV", "Lipophilicity"]:
        if data_name == "Lipophilicity":
            data_name = "lipo"
        dataset = MoleculeNet(root, name=data_name, transform=UnifyFeatureDims(configs.in_dim))
    elif data_name in ["PROTEINS", "MUTAG"]:
        dataset = TUDataset(root, data_name, transform=UnifyFeatureDims(configs.in_dim))
    else:
        raise ValueError('Invalid data_name')
    dataset = GraphDataset(dataset, data_name_map=data_name_map)
    return dataset


def load_few_shot_single_graph_data(configs, data_name, k_shot, num_splits, num_val=0.1):
    root = configs.root
    transform = T.Compose([
        FlattenLabels(),
        T.RandomNodeSplit(split='test_rest', num_splits=num_splits,
                          num_train_per_class=k_shot, num_val=num_val)
    ])
    if data_name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(root=root, name=data_name, transform=T.Compose([T.ToUndirected(), transform]))
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
    dataset = Node2GraphDataset(data, configs.k_hops, configs.num_neighbors, labeled=True)
    train_mask, val_mask, test_mask = graph_few_shot_splits(dataset, k_shot, num_val, num_splits)
    return dataset, train_mask, val_mask, test_mask


def load_few_shot_multi_graph_data(configs, data_name, k_shot, num_splits, num_val=0.5):
    """Just for single class classification"""
    root = configs.root
    if data_name in ["PROTEINS", "MUTAG", "ENZYMES"]:
        dataset = TUDataset(root, data_name)
    elif data_name in ["PCBA", "HIV"]:
        dataset=  MoleculeNet(root, data_name)
    else:
        raise ValueError('Invalid data_name')
    dataset = GraphDataset(dataset, labeled=True)
    train_mask, val_mask, test_mask = graph_few_shot_splits(dataset, k_shot, num_val, num_splits)
    return dataset, train_mask, val_mask, test_mask


def load_few_shot_link_graph_data(configs, data_name, k_shot, num_splits, num_val=0.1):
    root = configs.root
    if data_name == "WordNet18RR":
        transform_x = Node2VecEmbedding(configs.nv_dim, configs.nv_batch_size,
                                      configs.nv_walk_length, configs.nv_context_size,
                                      configs.nv_lr, configs.nv_walks_per_node,
                                      configs.nv_p, configs.nv_q, configs.nv_num_epochs)
        dataset = WordNet18RR(f"{root}/{data_name}", pre_transform=T.Compose([transform_x]))
        data = dataset[0]
    elif data_name == 'FB15k_237':
        transform_x = Node2VecEmbedding(configs.in_dim, configs.nv_batch_size,
                                      configs.nv_walk_length, configs.nv_context_size,
                                      configs.nv_lr, configs.nv_walks_per_node,
                                      configs.nv_p, configs.nv_q, configs.nv_num_epochs)
        train_dataset = FB15k_237(f"{root}/{data_name}", split='train', pre_transform=T.Compose([transform_x]))
        valid_dataset = FB15k_237(f"{root}/{data_name}", split='val', pre_transform=T.Compose([transform_x]))
        test_dataset = FB15k_237(f"{root}/{data_name}", split='test', pre_transform=T.Compose([transform_x]))

        train_data = train_dataset[0]
        valid_data = valid_dataset[0]
        test_data = test_dataset[0]

        all_edge_index = torch.cat([
            train_data.edge_index,
            valid_data.edge_index,
            test_data.edge_index
        ], dim=1)

        all_edge_type = torch.cat([
            train_data.edge_type,
            valid_data.edge_type,
            test_data.edge_type
        ])

        num_nodes = train_data.num_nodes
        data = Data(
            x=train_data.x if hasattr(train_data, 'x') else None,
            edge_index=all_edge_index,
            edge_type=all_edge_type,
            num_nodes=num_nodes
        )
    else:
        raise ValueError('Invalid data_name')
    if data.edge_weight is None:
        data.edge_weight = torch.ones_like(data.edge_index[0]).float()
    train_mask, val_mask, test_mask, selected_relations_list = link_k_shot_split(data, k_shot,
                                                                                 num_splits, num_val,
                                                                                 num_way=configs.num_way_link)

    def mask2dataset(mask, split_relations_list):
        d_list = []
        for t in range(num_splits):
            # global_rel → local_idx (0~9)
            rel_to_idx = {rel.item(): i for i, rel in enumerate(split_relations_list[t])}
            print(f"Split {t} relation mapping: {rel_to_idx}")

            ds = Link2GraphDataset(
                data, configs.k_hops, configs.num_neighbors,
                input_edge_idx=mask[:, t].nonzero().squeeze(),
                labeled=True,
                relation_mapping=rel_to_idx
            )

            d_list.append(ds)
        return d_list

    train_sets = mask2dataset(train_mask, selected_relations_list)
    val_sets = mask2dataset(val_mask, selected_relations_list)
    test_sets = mask2dataset(test_mask, selected_relations_list)
    return data, train_sets, val_sets, test_sets


class GraphDataset(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 data_name_map: Optional[int] = None,
                 labeled: bool = False):
        """

        :param dataset: Graph-level dataset
        :param data_name_map: e.g. if dataset is PROTEINS in {"PROTEINS": 0, "PubMed": 1}, them data_name_map = 0
        """
        super(GraphDataset, self).__init__()
        self.dataset = dataset
        self.data_name_map = data_name_map
        self._labeled = labeled

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes

    @property
    def num_features(self) -> int:
        return self.dataset.num_features

    @property
    def dataset_type(self):
        return "graph"

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data = self.dataset[idx]
        return Data(
            x=data.x.float(),
            y=data.y.long().reshape(-1) if hasattr(data, 'y') and self._labeled else None,
            edge_index=data.edge_index,
            edge_weight=data.edge_weight \
            if hasattr(data, 'edge_weight') and data.edge_weight is not None \
            else torch.ones_like(data.edge_index[0]).float(),
            data_name_map=self.data_name_map,
            data_type="graph"
        )


class Node2GraphDataset(Dataset):
    def __init__(
            self,
            data: Data,
            k_hops: int = 2,
            num_neighbors: Optional[List[int]] = None,
            data_name_map: int = None,
            input_node_idx: torch.Tensor = None,
            labeled: bool = False
    ):
        """

        :param data: Original Data object
        :param k_hops: number of hops
        :param data_name_map: e.g. if dataset is Cora in {"Cora": 0, "PubMed": 1}, them data_name_map = 0
        :param input_node_idx: nodes to extract subgraph. if None, extract all nodes.
        """
        super(Node2GraphDataset, self).__init__()
        assert len(num_neighbors) == k_hops, "sampling neighbor hops should be equal to k_hops"
        self.data = data
        self.k_hops = k_hops
        self.input_node_idx = input_node_idx if input_node_idx is not None else torch.arange(data.num_nodes)
        self.data_name_map = data_name_map
        self.sampler = NeighborSampler(
                        data.edge_index,
                        sizes=num_neighbors,
                        node_idx=self.input_node_idx,
                        num_nodes=data.num_nodes
                        )
        self._labeled = labeled
        if labeled and hasattr(data, 'y'):
            self.labels = data.y

    @property
    def num_classes(self) -> int:
        return torch.unique(self.data.y).numel()

    @property
    def num_features(self) -> int:
        return self.data.x.shape[1]

    @property
    def dataset_type(self):
        return "node"

    def len(self):
        return len(self.input_node_idx)

    def get(self, idx):
        target_node = self.input_node_idx[idx].reshape(-1)
        batch_size, n_id, adjs = self.sampler.sample(target_node)

        edge_index_list = []
        for adj in adjs:
            edge_index_list.append(adj.edge_index)
        if len(edge_index_list) > 0:
            edge_index = torch.cat(edge_index_list, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # mapping = (n_id == target_node).nonzero(as_tuple=True)[0].item()

        data = Data(
            x=self.data.x[n_id].clone(),
            edge_index=edge_index,
            # original_node_ids=n_id,
            # center_node_idx=mapping, # target node index in subset
            edge_weight=self.data.edge_weight[torch.cat([adj.e_id for adj in adjs])] \
            if hasattr(self.data, 'edge_weight') and self.data.edge_weight is not None \
            else torch.ones_like(edge_index[0]).float(),
            data_name_map=self.data_name_map,
            data_type="node"
        )
        if self._labeled:
            data.y = self.data.y[target_node]
        return data


class Link2GraphDataset(Dataset):
    def __init__(
            self,
            data: Data,
            k_hops: int = 2,
            num_neighbors: Optional[List[int]] = None,
            data_name_map: int = None,
            input_edge_idx: torch.Tensor = None,
            labeled: bool = False,
            relation_mapping: Optional[Dict[int, int]] = None,
    ):
        """
        Dataset that samples a k-hop subgraph around each link (edge), with edge_type as label.

        :param data: Original Data object
        :param k_hops: number of hops for neighbor sampling
        :param num_neighbors: list of number of neighbors to sample at each hop
        :param data_name_map: dataset identifier (e.g., 0 for Cora)
        :param input_edge_idx: edge indices to extract subgraphs for. If None, use all edges.
        :param labeled: whether to include edge_type as label
        """
        super(Link2GraphDataset, self).__init__()
        assert len(num_neighbors) == k_hops, "sampling neighbor hops should be equal to k_hops"
        self.data = data
        self.k_hops = k_hops
        self.input_edge_idx = input_edge_idx.reshape(-1) if input_edge_idx is not None else torch.arange(data.edge_index.size(1))
        self.data_name_map = data_name_map

        edge_nodes = data.edge_index[:, self.input_edge_idx].flatten().unique()
        self.sampler = NeighborSampler(
            data.edge_index,
            sizes=num_neighbors,
            node_idx=edge_nodes,
            num_nodes=data.num_nodes
        )

        self._labeled = labeled
        if labeled:
            if hasattr(data, 'edge_type'):
                self.edge_labels_global = data.edge_type[self.input_edge_idx]
                if self.edge_labels_global.dim() == 0:
                    self.edge_labels_global = self.edge_labels_global.unsqueeze(0)

                if relation_mapping is not None:
                    self.edge_labels = torch.tensor([
                        relation_mapping[gl.item()] for gl in self.edge_labels_global
                    ], dtype=torch.long)
                else:
                    raise ValueError("relation_mapping is None")
            else:
                raise ValueError("No edge labels found. Please provide 'edge_type' or 'edge_attr'.")

    @property
    def num_classes(self) -> int:
        if self._labeled:
            return torch.unique(self.edge_labels).numel()
        return 0

    @property
    def num_features(self) -> int:
        return self.data.x.shape[1]

    @property
    def dataset_type(self):
        return "link"

    def len(self):
        return len(self.input_edge_idx)

    def get(self, idx):
        edge_idx = self.input_edge_idx[idx]
        u, v = self.data.edge_index[:, edge_idx]  # 标量或向量

        _, n_id_u, adjs_u = self.sampler.sample([u])  # 注意：传入 list
        _, n_id_v, adjs_v = self.sampler.sample([v])

        edge_index_list_u = []
        for adj in adjs_u:
            edge_index_list_u.append(adj.edge_index)
        edge_index_u = torch.cat(edge_index_list_u, dim=1) \
            if edge_index_list_u else torch.empty((2, 0), dtype=torch.long)
        edge_weight_u = self.data.edge_weight[torch.cat([adj.e_id for adj in adjs_u])] \
            if hasattr(self.data, 'edge_weight') and self.data.edge_weight is not None \
            else torch.ones_like(edge_index_u[0]).float()

        edge_index_list_v = []
        for adj in adjs_v:
            edge_index_list_v.append(adj.edge_index)
        edge_index_v = torch.cat(edge_index_list_v, dim=1)\
            if edge_index_list_v else torch.empty((2, 0), dtype=torch.long)
        edge_weight_v = self.data.edge_weight[torch.cat([adj.e_id for adj in adjs_v])] \
            if hasattr(self.data, 'edge_weight') and self.data.edge_weight is not None \
            else torch.ones_like(edge_index_v[0]).float()

        u_local = (n_id_u == u).nonzero(as_tuple=True)[0].item()
        v_local = (n_id_v == v).nonzero(as_tuple=True)[0].item()

        edge_label = self.edge_labels[idx].reshape(-1)

        data_u = Data(
            x=self.data.x[n_id_u],
            edge_index=edge_index_u,
            edge_weight=edge_weight_u,
            root_n_id=u_local,
            data_name_map=self.data_name_map,
            data_type="node"
        )

        data_v = Data(
            x=self.data.x[n_id_v],
            edge_index=edge_index_v,
            edge_weight=edge_weight_v,
            root_n_id=v_local,
            data_name_map=self.data_name_map,
            data_type="node"
        )
        return [data_u, data_v], edge_label


class LinkCollater(Collater):
    def __init__(self, dataset, follow_batch=None, exclude_keys=None):
        super().__init__(dataset, follow_batch, exclude_keys)

    def __call__(self, batch):
        pairs, labels = zip(*batch)
        flattened = []
        for i, pair in enumerate(pairs):
            for j, data_obj in enumerate(pair):
                for attr_name in ['x', 'edge_index', 'edge_weight']:
                    if hasattr(data_obj, attr_name) and isinstance(getattr(data_obj, attr_name), torch.Tensor):
                        tensor_attr = getattr(data_obj, attr_name)
                        if tensor_attr.dim() == 0:
                            print(f"Find 0-d tensor: batch[{i}][{j}].{attr_name}, value: {tensor_attr}")
                            if attr_name == 'edge_weight':
                                setattr(data_obj, attr_name, torch.tensor([], dtype=torch.float))
                            elif attr_name == 'x':
                                setattr(data_obj, attr_name,
                                        torch.empty((0, data_obj.x.size(1)) if data_obj.x.dim() == 1 else data_obj.x))
                flattened.append(data_obj)

        for i, lbl in enumerate(labels):
            if isinstance(lbl, torch.Tensor) and lbl.dim() == 0:
                print(f"label[{i}] 是 0-d tensor: {lbl}")

        batch_obj = super().__call__(flattened)

        batch_obj.num_edges = len(batch)

        batch_obj.edge_label = torch.tensor(labels, device=flattened[0].x.device)  # [B]

        if hasattr(self.dataset, 'relation_mapping'):
            idx_to_rel = {v: k for k, v in self.dataset.relation_mapping.items()}
            candidate_rels = [idx_to_rel[i] for i in range(len(idx_to_rel))]
            batch_obj.candidate_relations = torch.tensor(candidate_rels)

        return batch_obj


class LinkDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=LinkCollater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )