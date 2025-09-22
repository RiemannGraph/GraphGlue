import torch
import torch.nn.functional as F
import torch.nn as nn

EPS = 1e-6


class PTGBLoss(nn.Module):
    def __init__(self, num_generators, temperature: int = 1.0):
        super(PTGBLoss, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(num_generators))
        self.temperature = temperature

    def forward(self, z_tan):
        """
        :param z_tan: [N, M, d]
        :return: loss
        """
        sim = torch.exp(self.alpha) * -torch.norm(z_tan, dim=-1, p=2) ** 2  # [N, M]
        sim = torch.exp(sim / self.temperature) + EPS
        div = torch.sum(sim, dim=-1, keepdim=True) + EPS
        loss = torch.mean(-torch.log(sim / div))
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z, z_aug):
        """

        :param z: [N, d]
        :param z_aug: [N, M, d]
        :return: loss
        """
        z = F.normalize(z, dim=-1)
        z_aug = F.normalize(z_aug, dim=-1)
        pos_sim = torch.exp(-torch.sum((z_aug - z.unsqueeze(1)) ** 2, dim=-1) / self.temperature)  # [N, M]
        neg_sim = torch.exp(-torch.sum((z.unsqueeze(0) - z.unsqueeze(1))**2, dim=-1) / self.temperature)   # [N, N]
        div = (neg_sim.sum(1) - neg_sim.diag()).unsqueeze(-1) + pos_sim
        loss = -torch.log(pos_sim / (div + 1e-6))
        return loss.mean()


class GeometricPersistLoss(nn.Module):
    def __init__(self, geo_regular_coef: float):
        super(GeometricPersistLoss, self).__init__()
        self.geo_regular_coef = geo_regular_coef

    def forward(self, pt_matrix, log_r_matrix):
        """
        In order of (i,j) (j,k) (i,k)
        :param pt_matrix: Parallel Translation matrix with shape [3, T, M]
        :param log_r_matrix: Log Volume Ratio matrix with shape [2, T]
        :return: geometric persistent loss
        """
        holo_loss = torch.mean(torch.norm(pt_matrix[1] * pt_matrix[0] - pt_matrix[2], p=2, dim=-1) ** 2)
        curv_loss = torch.mean((log_r_matrix[0] - log_r_matrix[1]) ** 2)
        return self.geo_regular_coef * holo_loss, self.geo_regular_coef * curv_loss

