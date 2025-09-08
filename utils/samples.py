import torch
import time
from torch_geometric.utils import is_undirected, to_undirected, degree
from torch_geometric.utils.num_nodes import maybe_num_nodes


def search_triangles(
    edge_index,
    num_path_samples: int = None,
    path_sample_times: int = 1,
    return_relabel_mapping: bool = False,
    threshold: int = 5000,  # edge threshold
    verbose: bool = False
):
    """
    For small graph, we just search all paths then samples.
    For large graph, we perform samples while searching.
    """
    E = edge_index.size(1)
    N = maybe_num_nodes(edge_index)
    device = edge_index.device

    if verbose:
        print(f"[INFO] Graph Scale: {N} nodes, {E} edges")
        start_time = time.time()

    if E <= threshold and N <= 10000:
        if verbose:
            print("Search then Sample")
        result = _search_triangles_small_scale(
            edge_index,
            num_path_samples,
            path_sample_times,
            return_relabel_mapping
        )
    else:
        if verbose:
            print("Sample while Search")
        result = _search_triangles_large_scale(
            edge_index,
            num_path_samples,
            path_sample_times,
            return_relabel_mapping,
            sampling_strategy='weighted'
        )

    if verbose:
        end_time = time.time()
        print(f"Time Cost: {end_time - start_time:.4f} 秒")

    return result


def _search_triangles_small_scale(
    edge_index,
    num_path_samples: int = None,
    path_sample_times: int = 1,
    return_relabel_mapping: bool = False,
):
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)
    device = edge_index.device

    row, col = edge_index
    num_nodes = maybe_num_nodes(edge_index)

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    adj[row, col] = True
    adj = adj | adj.t()

    triu_mask = torch.triu(torch.ones(num_nodes, num_nodes, device=device, dtype=torch.bool), diagonal=1)
    edge_mask = adj & triu_mask
    edges = torch.nonzero(edge_mask, as_tuple=False)  # [E, 2]

    if edges.size(0) == 0:
        empty_shape = (path_sample_times, 3, 0) if num_path_samples is not None else (0, 3)
        result = torch.empty(empty_shape, dtype=torch.long, device=device)
        if return_relabel_mapping:
            return result, torch.empty(0, dtype=torch.long, device=device), {}
        return result

    i_nodes = edges[:, 0]  # [E]
    j_nodes = edges[:, 1]  # [E]

    i_exp = i_nodes.unsqueeze(1)  # [E, 1]
    j_exp = j_nodes.unsqueeze(1)  # [E, 1]
    k_candidate = torch.arange(num_nodes, device=device).unsqueeze(0)  # [1, N]

    is_ik_edge = adj[i_exp, k_candidate]  # [E, N]
    is_jk_edge = adj[j_exp, k_candidate]  # [E, N]
    is_triangle = is_ik_edge & is_jk_edge

    valid_k = (k_candidate != i_exp) & (k_candidate != j_exp)
    is_triangle = is_triangle & valid_k

    e_indices, k_indices = torch.nonzero(is_triangle, as_tuple=True)
    if len(e_indices) == 0:
        empty_shape = (path_sample_times, 3, 0) if num_path_samples is not None else (0, 3)
        result = torch.empty(empty_shape, dtype=torch.long, device=device)
        if return_relabel_mapping:
            return result, torch.empty(0, dtype=torch.long, device=device), {}
        return result

    i_tri = i_nodes[e_indices]
    j_tri = j_nodes[e_indices]
    k_tri = k_indices
    triangles = torch.stack([i_tri, j_tri, k_tri], dim=1)  # [T, 3]

    return _postprocess_triangles_optimized(
        triangles,
        num_path_samples,
        path_sample_times,
        return_relabel_mapping,
        device,
        edge_index
    )


def _search_triangles_large_scale(
    edge_index,
    num_path_samples: int = 1000,
    path_sample_times: int = 1,
    return_relabel_mapping: bool = False,
    sampling_strategy: str = 'weighted',
):
    assert path_sample_times >= 1
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)
    device = edge_index.device

    row, col = edge_index
    num_nodes = maybe_num_nodes(edge_index)

    MAX_NODE = num_nodes
    if MAX_NODE >= 2**31:
        raise ValueError("Node index too large for edge hashing (>=2^31)")

    edges_minmax = torch.stack([torch.min(row, col), torch.max(row, col)], dim=0)
    edges_unique = torch.unique(edges_minmax, dim=1)
    edge_hash = edges_unique[0] * MAX_NODE + edges_unique[1]

    deg = degree(row, num_nodes=num_nodes, dtype=torch.long)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=device), deg.cumsum(0)])
    idx = torch.argsort(row)
    col_sorted = col[idx]

    candidate_mask = deg >= 2
    candidate_nodes = torch.where(candidate_mask)[0]
    candidate_deg = deg[candidate_mask]

    if len(candidate_nodes) == 0:
        empty_shape = (path_sample_times, 3, 0)
        result = torch.empty(empty_shape, dtype=torch.long, device=device)
        if return_relabel_mapping:
            return result, torch.empty(0, dtype=torch.long, device=device), {}
        return result

    if sampling_strategy == 'weighted':
        sample_weights = candidate_deg.float()
        sample_weights = sample_weights / sample_weights.sum()
    else:
        sample_weights = None

    all_sampled_triangles = []

    for _ in range(path_sample_times):
        sampled_triangles = []
        attempts = 0
        max_attempts = num_path_samples * 20

        while len(sampled_triangles) < num_path_samples and attempts < max_attempts:
            attempts += 1

            batch_size = min(1024, num_path_samples * 2)
            j_indices = torch.multinomial(
                sample_weights if sample_weights is not None else torch.ones(len(candidate_nodes), device=device),
                batch_size,
                replacement=True
            )
            j_nodes = candidate_nodes[j_indices]

            j_ptr_start = ptr[j_nodes]
            j_ptr_end = ptr[j_nodes + 1]
            j_deg = j_ptr_end - j_ptr_start

            rand1 = torch.floor(torch.rand(batch_size, device=device) * j_deg).long()
            rand2 = torch.floor(torch.rand(batch_size, device=device) * j_deg).long()

            # ensure i ≠ k
            same_mask = rand1 == rand2
            while same_mask.any():
                rand2[same_mask] = torch.floor(torch.rand(same_mask.sum(), device=device) * j_deg[same_mask]).long()
                same_mask = rand1 == rand2

            max_deg_in_batch = j_deg.max().item()   # [B, max_deg]
            if max_deg_in_batch == 0:
                continue

            neighbor_mat = torch.zeros(batch_size, max_deg_in_batch, dtype=torch.long, device=device)
            for i in range(batch_size):
                start, end = j_ptr_start[i], j_ptr_end[i]
                if end > start:
                    neighbor_mat[i, :end - start] = col_sorted[start:end]

            i_nodes = torch.gather(neighbor_mat, 1, rand1.unsqueeze(1)).squeeze(1)
            k_nodes = torch.gather(neighbor_mat, 1, rand2.unsqueeze(1)).squeeze(1)

            i_min_k = torch.min(i_nodes, k_nodes)
            i_max_k = torch.max(i_nodes, k_nodes)
            query_hash = i_min_k * MAX_NODE + i_max_k

            is_edge = torch.isin(query_hash, edge_hash)
            valid_mask = is_edge & (i_nodes != j_nodes) & (k_nodes != j_nodes) & (i_nodes != k_nodes)

            if valid_mask.any():
                valid_i = i_nodes[valid_mask]
                valid_j = j_nodes[valid_mask]
                valid_k = k_nodes[valid_mask]
                new_tris = torch.stack([valid_i, valid_j, valid_k], dim=1)
                sampled_triangles.append(new_tris)

                if torch.cat(sampled_triangles, dim=0).size(0) >= num_path_samples:
                    collected = torch.cat(sampled_triangles, dim=0)[:num_path_samples]
                    all_sampled_triangles.append(collected.t())
                    break

        if len(sampled_triangles) == 0:
            all_sampled_triangles.append(torch.empty((3, 0), dtype=torch.long, device=device))
        else:
            collected = torch.cat(sampled_triangles, dim=0)
            if collected.size(0) < num_path_samples:
                if collected.size(0) == 0:
                    all_sampled_triangles.append(torch.empty((3, 0), dtype=torch.long, device=device))
                else:
                    repeat_times = (num_path_samples // collected.size(0)) + 1
                    padded = collected.repeat(repeat_times, 1)[:num_path_samples]
                    all_sampled_triangles.append(padded.t())

    if path_sample_times == 1:
        result = all_sampled_triangles[0].unsqueeze(0)
    else:
        result = torch.stack(all_sampled_triangles, dim=0)

    if not return_relabel_mapping:
        return result

    # relabel
    relabeled_list = []
    mappings = []
    inverse_mappings = []

    for t in range(path_sample_times):
        tri_flat = result[t].t()
        if tri_flat.size(0) == 0:
            relabeled_list.append(result[t])
            mappings.append(torch.empty(0, dtype=torch.long, device=device))
            inverse_mappings.append(torch.tensor([], dtype=torch.long, device=device))
            continue

        unique_nodes, inverse_flat = torch.unique(tri_flat, return_inverse=True)
        relabeled_tri = inverse_flat.view_as(tri_flat).t().contiguous()
        relabeled_list.append(relabeled_tri)

        inv_map = torch.full((unique_nodes.max().item() + 1,), -1, dtype=torch.long, device=device)
        inv_map[unique_nodes] = torch.arange(len(unique_nodes), device=device)

        mappings.append(unique_nodes)
        inverse_mappings.append(inv_map)

    relabeled_triangles = torch.stack(relabeled_list, dim=0)

    if path_sample_times == 1:
        return relabeled_triangles.squeeze(0), mappings[0], inverse_mappings[0]
    else:
        return relabeled_triangles, mappings, inverse_mappings


def _postprocess_triangles_optimized(
    triangles,
    num_path_samples,
    path_sample_times,
    return_relabel_mapping,
    device,
    edge_index
):
    if num_path_samples is None:
        if not return_relabel_mapping:
            return triangles.t().contiguous()
        else:
            unique_nodes, inverse_flat = torch.unique(triangles, return_inverse=True)
            relabeled = inverse_flat.view_as(triangles).t().contiguous()
            inv_map = torch.full((unique_nodes.max().item() + 1,), -1, dtype=torch.long, device=device)
            inv_map[unique_nodes] = torch.arange(len(unique_nodes), device=device)
            return relabeled, unique_nodes, inv_map

    N = triangles.size(0)
    if N == 0:
        empty_shape = (path_sample_times, 3, 0)
        result = torch.empty(empty_shape, dtype=torch.long, device=device)
        if return_relabel_mapping:
            return result, torch.empty(0, dtype=torch.long, device=device), {}
        return result

    num_path_samples = min(num_path_samples, N)

    node_degree = degree(edge_index[0], num_nodes=edge_index.max()+1, dtype=torch.float)
    j_deg = node_degree[triangles[:, 1]]
    i_deg = node_degree[triangles[:, 0]]
    k_deg = node_degree[triangles[:, 2]]
    scores = (i_deg + j_deg + k_deg).clamp(min=1e-12)
    prob = scores / scores.sum()

    if path_sample_times == 1:
        sampled_idx = torch.multinomial(prob, num_path_samples, replacement=False)
        sampled_triangles = triangles[sampled_idx].t().unsqueeze(0)
    else:
        sampled_list = []
        for _ in range(path_sample_times):
            idx = torch.multinomial(prob, num_path_samples, replacement=False)
            sampled_list.append(triangles[idx].t())
        sampled_triangles = torch.stack(sampled_list, dim=0)

    if not return_relabel_mapping:
        return sampled_triangles

    relabeled_list = []
    mappings = []
    inverse_mappings = []

    for t in range(path_sample_times):
        tri_flat = sampled_triangles[t].t()
        unique_nodes, inverse_flat = torch.unique(tri_flat, return_inverse=True)
        relabeled_tri = inverse_flat.view_as(tri_flat).t().contiguous()
        relabeled_list.append(relabeled_tri)

        inv_map = torch.full((unique_nodes.max().item() + 1,), -1, dtype=torch.long, device=device)
        inv_map[unique_nodes] = torch.arange(len(unique_nodes), device=device)

        mappings.append(unique_nodes)
        inverse_mappings.append(inv_map)

    relabeled_triangles = torch.stack(relabeled_list, dim=0)

    if path_sample_times == 1:
        return relabeled_triangles.squeeze(0), mappings[0], inverse_mappings[0]
    else:
        return relabeled_triangles, mappings, inverse_mappings