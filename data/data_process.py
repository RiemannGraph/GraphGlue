import numpy as np
import torch
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional
from torch_geometric.data import Data, Dataset
from torch_geometric.nn.kge import TransE, ComplEx, DistMult, RotatE
from torch_geometric.utils import degree, to_undirected, is_undirected


class KGNodeInitializer:

    MODEL_CONFIG = {
        'transe': {
            'model_class': TransE,
            'optimizer': lambda params: optim.Adam(params, lr=0.01),
            'kwargs': {}
        },
        'complex': {
            'model_class': ComplEx,
            'optimizer': lambda params: optim.Adagrad(params, lr=0.001, weight_decay=1e-6),
            'kwargs': {}
        },
        'distmult': {
            'model_class': DistMult,
            'optimizer': lambda params: optim.Adam(params, lr=0.0001, weight_decay=1e-6),
            'kwargs': {}
        },
        'rotate': {
            'model_class': RotatE,
            'optimizer': lambda params: optim.Adam(params, lr=1e-3),
            'kwargs': {'margin': 9.0}
        }
    }

    def __init__(self, model_name: str, device: torch.device):
        """

        Args:
            model_name: 'transe', 'complex', 'distmult', 'rotate'
            device: cpu or cuda
        """
        assert model_name in self.MODEL_CONFIG, f"Unsupported model: {model_name}"

        self.model_name = model_name
        self.device = device
        self.model = None
        self.optimizer = None

    def setup_model(self, num_nodes: int, num_relations: int, hidden_channels: int = 50):
        config = self.MODEL_CONFIG[self.model_name]
        self.model = config['model_class'](
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_channels=hidden_channels,
            **config['kwargs']
        ).to(self.device)

        self.optimizer = config['optimizer'](self.model.parameters())

    def create_loader(self, data: Data, batch_size: int, shuffle: bool = True):
        if self.model is None:
            raise RuntimeError("Using setup_model() to initialize the model at first.")

        return self.model.loader(
            head_index=data.edge_index[0],
            rel_type=data.edge_type,
            tail_index=data.edge_index[1],
            batch_size=batch_size,
            shuffle=shuffle
        )

    def train_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = total_examples = 0

        for head_index, rel_type, tail_index in train_loader:
            self.optimizer.zero_grad()
            loss = self.model.loss(head_index, rel_type, tail_index)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss) * head_index.numel()
            total_examples += head_index.numel()

        return total_loss / total_examples if total_examples > 0 else 0.0

    @torch.no_grad()
    def evaluate(self, data: Data, batch_size: int = 1000, k: int = 10) -> Tuple[float, float, float]:
        self.model.eval()
        rank, mrr, hits = self.model.test(
            head_index=data.edge_index[0],
            rel_type=data.edge_type,
            tail_index=data.edge_index[1],
            batch_size=batch_size,
            k=k,
        )
        return rank, mrr, hits

    def get_node_embeddings(self) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        return self.model.node_emb.weight.data.clone()

    def get_relation_embeddings(self) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        return self.model.rel_emb.weight.data.clone()

    def fit(
            self,
            train_data: Data,
            valid_data: Data,
            test_data: Data,
            hid_channels: int,
            batch_size: int = 1024,
            epochs: int = 500,
            eval_interval: int = 25,
            verbose: bool = True
    ) -> Dict[str, Any]:
        self.setup_model(
            num_nodes=train_data.num_nodes,
            num_relations=train_data.num_edge_types if hasattr(train_data, 'num_edge_types')
            else train_data.edge_type.max().item() + 1,
            hidden_channels=hid_channels
        )

        train_data = train_data.to(self.device)
        valid_data = valid_data.to(self.device)
        test_data = test_data.to(self.device)

        train_loader = self.create_loader(train_data, batch_size)

        results = {'train_loss': [], 'val_metrics': [], 'test_metrics': None}

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(train_loader)
            results['train_loss'].append(loss)

            if verbose and epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

            if epoch % eval_interval == 0 and valid_data is not None:
                rank, mrr, hits = self.evaluate(valid_data)
                results['val_metrics'].append((epoch, rank, mrr, hits))

                if verbose:
                    print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, '
                          f'Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}')

        if test_data is not None:
            rank, mrr, hits = self.evaluate(test_data)
            results['test_metrics'] = (rank, mrr, hits)

            if verbose:
                print(f'Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, '
                      f'Test Hits@10: {hits:.4f}')

        results['node_embeddings'] = self.get_node_embeddings()
        results['relation_embeddings'] = self.get_relation_embeddings()

        return results


def search_adjacent_edges(edge_index, num_samples=None):
    """
    :param edge_index: [2, E]
    :param num_samples: If None, return all paths. Else, return given number of paths.
    :return paths: torch.Tensor (i, j) (j, k) [N, 3]
    """
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)
    device = edge_index.device

    src = edge_index[0]  # i
    dst = edge_index[1]  # j

    sorted_dst, dst_perm = torch.sort(dst)
    sorted_src, src_perm = torch.sort(src)

    left_idx = torch.searchsorted(sorted_src, sorted_dst, side='left')
    right_idx = torch.searchsorted(sorted_src, sorted_dst, side='right')
    match_counts = right_idx - left_idx
    total_matches = match_counts.sum()

    if total_matches == 0:
        return torch.empty((0, 3), dtype=torch.long, device=device)

    cum_match_counts = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                                  torch.cumsum(match_counts, 0)])
    dst_indices_repeated = torch.arange(len(sorted_dst), device=device).repeat_interleave(match_counts)

    base_indices = torch.arange(total_matches, device=device)
    group_id = torch.searchsorted(cum_match_counts[1:], base_indices, right=True)
    group_start = cum_match_counts[group_id]
    offset = base_indices - group_start
    src_indices = left_idx[group_id] + offset

    i_j_edge_idx = dst_perm[dst_indices_repeated]  # i->j
    j_k_edge_idx = src_perm[src_indices]  # j->k

    paths = torch.stack([
        src[i_j_edge_idx],  # i (from i->j)
        dst[i_j_edge_idx],  # j (from i->j)
        dst[j_k_edge_idx]  # k (from j->k)
    ], dim=1)
    paths = paths[paths[:, 0] != paths[:, 2]]

    if num_samples is not None:
        node_degree = degree(edge_index)
        j_deg = node_degree[paths[:, 1]]
        i_deg = node_degree[paths[:, 0]]
        k_deg = node_degree[paths[:, 2]]
        scores = (i_deg + j_deg + k_deg)
        prob = scores / scores.sum()
        sampled_idx = torch.multinomial(prob, num_samples, replacement=False)
        paths = paths[sampled_idx]

    return paths.t().contiguous()


def unify_feature_dimension(
        x,
        uni_dim: int,
        center: bool = True
):
    if x.dim() == 1:
        x = x.unsqueeze(-1)  # [n] -> [n, 1]

    num_nodes, original_dim = x.shape
    device = x.device

    if num_nodes == 0 or original_dim == 0:
        return torch.zeros((num_nodes, uni_dim), device=device, dtype=torch.float)

    x = x.float()

    if center:
        x = x - x.mean(dim=0, keepdim=True)

    if original_dim >= uni_dim:
        U, S, Vt = torch.svd(x, some=True)  # U: [n, min(n,d)], S: [min(n,d)]
        k = min(U.shape[1], uni_dim)
        U_k = U[:, :k]  # [n, k]
        S_k = S[:k]  # [k]
        x_reduced = U_k * S_k  # [n, k]

        if k < uni_dim:
            padding = torch.zeros((num_nodes, uni_dim - k), device=device, dtype=torch.float)
            x_reduced = torch.cat([x_reduced, padding], dim=1)  # [n, uni_dim]

    else:
        U, S, Vt = torch.svd(x, some=True)
        k = U.shape[1]  # min(n, original_dim)
        x_reduced = U * S  # [n, k]

        if k < uni_dim:
            padding = torch.zeros((num_nodes, uni_dim - k), device=device, dtype=torch.float)
            x_reduced = torch.cat([x_reduced, padding], dim=1)  # [n, uni_dim]
        else:
            x_reduced = x_reduced[:, :uni_dim]

    x_reduced = torch.nan_to_num(x_reduced, nan=0.0, posinf=0.0, neginf=0.0)

    return x_reduced


def graph_few_shot_splits(dataset, k_shot, num_val, num_splits):
    train_masks, val_masks, test_masks = [], [], []
    for _ in range(num_splits):
        train_mask, val_mask, test_mask = _graph_few_shot_one_split(dataset, k_shot, num_val)
        train_masks.append(train_mask)
        val_masks.append(val_mask)
        test_masks.append(test_mask)
    train_mask = torch.stack(train_masks, dim=1)
    val_mask = torch.stack(val_masks, dim=1)
    test_mask = torch.stack(test_masks, dim=1)
    return train_mask, val_mask, test_mask


def _graph_few_shot_one_split(dataset, k_shot=5, num_val=0.5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        dataset_train, dataset_val, dataset_test
    """

    labels = [data.y.item() for data in dataset]
    num_classes = len(set(labels))
    num_graphs = len(dataset)

    label_to_indices = [[] for _ in range(num_classes)]
    for idx, y in enumerate(labels):
        label_to_indices[y].append(idx)

    for y in range(num_classes):
        if len(label_to_indices[y]) < k_shot:
            raise ValueError(f"Class {y} has only {len(label_to_indices[y])} graphs, but k_shot={k_shot}")

    train_indices = []
    remaining_indices = []

    for y in range(num_classes):
        indices = np.array(label_to_indices[y])
        np.random.shuffle(indices)
        train_indices.extend(indices[:k_shot].tolist())
        remaining_indices.extend(indices[k_shot:].tolist())

    val_size = int(len(remaining_indices) * num_val)
    np.random.shuffle(remaining_indices)

    val_indices = remaining_indices[:val_size]
    test_indices = remaining_indices[val_size:]

    print(f"Total graphs: {num_graphs}")
    print(f"Train (support): {len(train_indices)} graphs ({k_shot} per class)")
    print(f"Val: {len(val_indices)} graphs")
    print(f"Test: {len(test_indices)} graphs")
    print(f"Val ratio in remaining: {len(val_indices) / (len(val_indices) + len(test_indices)):.2f}")

    train_mask = np.zeros(len(dataset), dtype=bool)
    val_mask = np.zeros(len(dataset), dtype=bool)
    test_mask = np.zeros(len(dataset), dtype=bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)

    return train_mask, val_mask, test_mask


def link_k_shot_split(data, k_shot, num_splits, num_val=0.1, num_test=0.2):
    """
    :return list of (train_data, val_data, test_data) for each split
    """

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