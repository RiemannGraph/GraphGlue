import numpy as np
import torch
from torch_geometric.utils import degree, to_undirected, is_undirected


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

    if num_samples is not None and num_samples < paths.shape[0]:
        node_degree = degree(edge_index[0])
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


def link_k_shot_split(data, k_shot, num_splits, num_val=0.1,):
    """
    :return list of (train_data, val_data, test_data) for each split
    """

    edge_index = data.edge_index  # [2, num_edges]
    edge_type = data.edge_type    # [num_edges,]
    num_edges = edge_index.size(1)
    num_relations = int(edge_type.max().item() + 1)

    train_masks = []
    val_masks = []
    test_masks = []

    for _ in range(num_splits):
        train_indices = []
        val_indices = []
        test_indices = []

        for rel in range(num_relations):
            rel_mask = (edge_type == rel)
            rel_indices = rel_mask.nonzero(as_tuple=False).view(-1)  # [num_rel_edges]
            num_rel_edges = rel_indices.size(0)

            if num_rel_edges == 0:
                continue

            perm = torch.randperm(num_rel_edges)
            rel_indices_shuffled = rel_indices[perm]

            k = min(k_shot, num_rel_edges)
            train_indices.append(rel_indices_shuffled[:k])

            remaining_indices = rel_indices_shuffled[k:]
            num_remaining = remaining_indices.size(0)

            if num_remaining == 0:
                val_split = torch.empty(0, dtype=torch.long)
                test_split = torch.empty(0, dtype=torch.long)
            else:
                val_ratio = num_val
                val_size = int(num_remaining * val_ratio)
                val_split = remaining_indices[:val_size]
                test_split = remaining_indices[val_size:]

            val_indices.append(val_split)
            test_indices.append(test_split)

        train_idx = torch.cat(train_indices) if len(train_indices) > 0 else torch.empty(0, dtype=torch.long)
        val_idx = torch.cat(val_indices) if len(val_indices) > 0 else torch.empty(0, dtype=torch.long)
        test_idx = torch.cat(test_indices) if len(test_indices) > 0 else torch.empty(0, dtype=torch.long)

        train_mask = torch.zeros(num_edges).bool()
        val_mask = torch.zeros(num_edges).bool()
        test_mask = torch.zeros(num_edges).bool()

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        train_masks.append(train_mask)
        val_masks.append(val_mask)
        test_masks.append(test_mask)

    train_mask = torch.stack(train_masks, dim=1)
    val_mask = torch.stack(val_masks, dim=1)
    test_mask = torch.stack(test_masks, dim=1)

    return train_mask, val_mask, test_mask