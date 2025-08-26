import torch
import torch.nn.functional as F
import torch.nn as nn


class PTGBLoss(nn.Module):
    def __init__(self, num_generators, temperature: int = 1.0):
        super(PTGBLoss, self).__init__()
        self.alpha = nn.Parameter(torch.ones(num_generators))
        self.temperature = temperature

    def forward(self, z, z_tan):
        """

        :param z: [N, d]
        :param z_tan: [N, M, d]
        :return: loss
        """
        z = F.normalize(z, dim=-1).unsqueeze(1)
        z_tan = F.normalize(z_tan, dim=-1)
        sim = self.alpha.unsqueeze(1) * -torch.sum((z_tan - z)**2, dim=-1)  # [N, M]
        sim = torch.exp(sim / self.temperature)
        div = torch.sum(sim, dim=-1, keepdim=True)
        loss = torch.mean(-torch.log(sim / div))
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z, z_tan):
        """

        :param z: [N, d]
        :param z_tan: [N, M, d]
        :return: loss
        """
        z = F.normalize(z, dim=-1)
        z_tan = F.normalize(z_tan, dim=-1)
        pos_sim = torch.exp(-torch.sum((z_tan - z.unsqueeze(1))**2, dim=-1) / self.temperature)  # [N, M]
        neg_sim = torch.exp(-torch.sum((z.unsqueeze(0) - z.unsqueeze(1))**2, dim=-1) / self.temperature)   # [N, N]
        div = (neg_sim.sum(1) - neg_sim.diag()).unsqueeze(-1) + pos_sim
        loss = -torch.log(pos_sim / div)
        return loss.mean()


class GeometricPersistLoss(nn.Module):
    def __init__(self, regular_coef: int):
        super(GeometricPersistLoss, self).__init__()
        self.regular_coef = regular_coef

    def forward(self, pt_matrix, log_r_matrix):
        """
        In order of (i,j) (j,k) (i,k)
        :param pt_matrix: Parallel Translation matrix with shape [3, T, d, d]
        :param log_r_matrix: Log Volume Ratio matrix with shape [3, T]
        :return: geometric persistent loss
        """
        log_r_matrix = log_r_matrix.unsqueeze(-1).unsqueeze(-1)   # [3, T, 1, 1]
        term_ij = log_r_matrix[0] * pt_matrix[0]    # [T, d, d]
        term_jk = log_r_matrix[1] * pt_matrix[1]
        term_ik = log_r_matrix[2] * pt_matrix[2]
        loss = torch.frobenius_norm(term_jk @ term_ij - term_ik, dim=[1, 2])
        loss = self.regular_coef * torch.mean(loss)
        return loss

