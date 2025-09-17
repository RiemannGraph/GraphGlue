import torch
import torch.nn.functional as F
import numpy as np
from downstream.adapter import RPGPrompt
from sklearn.metrics import roc_auc_score


def _forward_pass(model, data):
    z_log_metric_adapt, align_loss = model(data)
    pred = model.predict(z_log_metric_adapt, data)
    return pred, align_loss


def _compute_metrics(preds_list, trues_list, metric: str = "acc"):
    preds = np.concatenate(preds_list, axis=-1)
    trues = np.concatenate(trues_list, axis=-1)
    if metric == "acc":
        metric = np.sum(preds == trues) / len(preds)
    elif metric == "auc":
        metric = roc_auc_score(trues, preds)
    return metric


def train_step(loader, optimizer, model: RPGPrompt, device, label_attr='y', metric="acc"):
    model.train()
    total_loss = 0.0
    preds_list = []
    trues_list = []

    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)

        pred, align_loss = _forward_pass(model, data)

        label = getattr(data, label_attr)

        loss = F.cross_entropy(pred, label) + align_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds_list.append(pred.detach().cpu().numpy().argmax(-1))
        trues_list.append(label.detach().cpu().numpy())

    acc = _compute_metrics(preds_list, trues_list, metric)
    avg_loss = total_loss / len(loader)
    return avg_loss, acc


def eval_step(loader, model: RPGPrompt, device, label_attr='y', metric="acc"):
    model.eval()
    total_loss = 0.0
    preds_list = []
    trues_list = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred, align_loss = _forward_pass(model, data)

            label = getattr(data, label_attr)

            loss = F.cross_entropy(pred, label) + align_loss
            total_loss += loss.item()
            preds_list.append(pred.detach().cpu().numpy().argmax(-1))
            trues_list.append(label.detach().cpu().numpy())

    acc = _compute_metrics(preds_list, trues_list, metric)
    avg_loss = total_loss / len(loader)
    return avg_loss, acc