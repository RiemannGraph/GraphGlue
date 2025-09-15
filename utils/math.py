import torch

EPS = 1e-6


def matrix_log_sym(G: torch.Tensor) -> torch.Tensor:
    """
    Compute the matrix logarithm for a batch of symmetric positive definite matrices.
    Uses eigenvalue decomposition for numerical stability.

    Args:
        G: Tensor of shape [*, M, M], symmetric positive definite.

    Returns:
        Log(G): Tensor of same shape as G.
    """
    eigvals, eigvecs = torch.linalg.eigh(G)
    eigvals = torch.clamp(eigvals, min=EPS)
    log_eigvals = torch.log(eigvals)
    log_G = torch.matmul(eigvecs, torch.matmul(torch.diag_embed(log_eigvals), eigvecs.transpose(-2, -1)))
    return log_G


def metric(basis: torch.Tensor) -> torch.Tensor:
    return basis @ basis.transpose(-1, -2)


def log_volume(basis):
    """
    Log volume of metric tensor w.r.t. the standard basis.
    :param basis: [*, M, d]
    :return:
    """
    log_vol_stable = torch.logdet(metric(basis).clamp(min=EPS))
    return log_vol_stable


def log_volume_ratio(basis_src, basis_dst):
    """
    Volume ratio between two tangent spaces to estimate Ricci Curvature.
    :param basis_src: [*, M, d]
    :param basis_dst: [*, M, d]

    :return: log ratio: torch.Tensor
    """
    log_vol_src, log_vol_dst = log_volume(basis_src), log_volume(basis_dst)
    log_ratio = log_vol_src - log_vol_dst
    return log_ratio


def parallel_translation(G_i: torch.Tensor, G_j: torch.Tensor) -> torch.Tensor:
    """
    Compute the optimal isometric parallel transport map P such that:
        P^T @ Gj @ P = Gi
    using SVD for improved numerical stability.

    Args:
        G_i (torch.Tensor): Metric tensor at node i, shape (..., M, M)
        G_j (torch.Tensor): Metric tensor at node j, shape (..., M, M)

    Returns:
        P (torch.Tensor): Optimal parallel transport map, shape (..., M, M)
    """
    S_j, U_j = torch.linalg.eigh(G_j)
    S_j = torch.clamp(S_j, min=EPS)
    G_j_inv_sqrt = U_j @ torch.diag_embed(1.0 / torch.sqrt(S_j)) @ U_j.transpose(-2, -1)

    G_j_sqrt = U_j @ torch.diag_embed(torch.sqrt(S_j)) @ U_j.transpose(-2, -1)
    A = G_j_sqrt @ G_i @ G_j_sqrt

    S_A, U_A = torch.linalg.eigh(A)
    S_A = torch.clamp(S_A, min=EPS)
    A_sqrt = U_A @ torch.diag_embed(torch.sqrt(S_A)) @ U_A.transpose(-2, -1)

    P = G_j_inv_sqrt @ A_sqrt @ G_j_inv_sqrt

    return P