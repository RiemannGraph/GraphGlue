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


def log_volume(basis):
    """
    Log volume of metric tensor w.r.t. the standard basis.
    :param basis: [*, M, d]
    :return:
    """
    metric = basis @ basis.transpose(-1, -2)  # [M, M]
    log_vol_stable = torch.logdet(metric)
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


def parallel_translation(basis_src: torch.Tensor, basis_dst: torch.Tensor) -> torch.Tensor:
    """
    Compute the optimal isometric parallel transport map P such that:
        P^T @ Gj @ P = Gi
    using the closed-form solution from matrix geometry.

    Args:
        basis_src (torch.Tensor): Tangent basis at node i, shape (N, M, d)
        basis_dst (torch.Tensor): Tangent basis at node j, shape (N, M, d)

    Returns:
        P (torch.Tensor): Optimal parallel transport map, shape (N, M, M)
                         such that P^T @ G_j @ P = G_i
    """
    G_i = basis_src @ basis_src.transpose(-1, -2)
    G_j = basis_dst @ basis_dst.transpose(-1, -2)

    def safe_sqrt_inv(M):
        eigvals, eigvecs = torch.linalg.eigh(M)
        eigvals_clamped = torch.clamp(eigvals, min=EPS)
        sqrt_M = eigvecs @ torch.diag_embed(torch.sqrt(eigvals_clamped)) @ eigvecs.transpose(-1, -2)
        inv_sqrt_M = eigvecs @ torch.diag_embed(1.0 / torch.sqrt(eigvals_clamped)) @ eigvecs.transpose(-1, -2)
        return sqrt_M, inv_sqrt_M

    G_j_sqrt, G_j_inv_sqrt = safe_sqrt_inv(G_j)

    A = G_j_sqrt @ G_i @ G_j_sqrt

    A_sqrt, _ = safe_sqrt_inv(A)

    P = G_j_inv_sqrt @ A_sqrt @ G_j_inv_sqrt

    return P