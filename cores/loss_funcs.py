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
        :param z_tan: [M, N, d]
        :return: loss
        """
        z = F.normalize(z, dim=-1).unsqueeze(0)
        z_tan = F.normalize(z_tan, dim=-1)
        sim = self.alpha.unsqueeze(1) * -torch.sum((z_tan - z)**2, dim=-1)  # [M, N]
        sim = torch.exp(sim / self.temperature)
        div = torch.sum(sim, dim=0, keepdim=True)
        loss = torch.mean(-torch.log(sim / div))
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    def forward(self, z, z_tan):
        z = F.normalize(z, dim=-1)
        z_tan = F.normalize(z_tan, dim=-1)
        pos_sim = torch.exp(-torch.sum((z_tan - z.unsqueeze(0))**2, dim=-1) / self.temperature)  # [M, N]
        neg_sim = torch.exp(-torch.sum((z.unsqueeze(0) - z.unsqueeze(1))**2, dim=-1) / self.temperature)   # [N, N]
        div = (neg_sim.sum(1) - neg_sim.diag()).unsqueeze(0) + pos_sim
        loss = -torch.log(pos_sim / div)
        return loss.mean()


def PT_loss(z, edge_index):
    pass


def Curv_loss(z, edge_index):
    pass
