# task.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Dict, Tuple
import numpy as np


def create_prototypes(
    z: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Compute class prototypes from embeddings of labeled nodes (few-shot or full).
    :param z: [N, D] embeddings
    :param y: [N] labels
    :param mask: [N] bool, which nodes are labeled (support set)
    :param num_classes: int
    :return: [C, D] prototypes for each class
    """
    device = z.device
    prototypes = torch.zeros(num_classes, z.size(-1), device=device)

    masked_z = z[mask]
    masked_y = y[mask]

    for c in range(num_classes):
        class_mask = (masked_y == c)
        if class_mask.sum() > 0:
            prototypes[c] = masked_z[class_mask].mean(dim=0)
        else:
            prototypes[c] = torch.randn(z.size(-1), device=device)

    return prototypes


def prototypical_contrastive_loss(
    z_query: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 1.0
):
    """
    Compute contrastive loss using prototypes as anchors.
    Similar to InfoNCE, but class-level.

    :param z_query: [Q, D] query embeddings
    :param prototypes: [C, D] class prototypes
    :param temperature: scalar, controls sharpness
    :return: loss scalar
    """
    # Normalize
    z_query = F.normalize(z_query, p=2, dim=-1)
    prototypes = F.normalize(prototypes, p=2, dim=-1)

    # Similarity: [Q, C]
    sim = torch.mm(z_query, prototypes.t()) / temperature

    log_prob = F.log_softmax(sim, dim=-1)
    loss = -log_prob.mean()

    return loss


def evaluate_with_prototype(
    model: nn.Module,
    data: Data,
    mask: torch.Tensor,
    query_mask: torch.Tensor,
    num_classes: int,
    device: torch.device
) -> torch.Tensor:
    """
    Evaluate accuracy using prototype method.
    """
    model.eval()
    with torch.no_grad():
        z, z_tan = model(data.to(device))

        prototypes = create_prototypes(z, data.y, mask, num_classes)
        logits = torch.mm(F.normalize(z[query_mask]), F.normalize(prototypes).t())
        pred = logits.argmax(dim=-1)

    return pred