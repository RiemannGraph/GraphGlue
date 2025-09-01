import torch
import torch.nn.functional as F
import numpy as np
from downstream.adapter import RPGPrompt


def train_node_cls(train_loader, optimizer, model: RPGPrompt, device):
    total_loss = 0.
    batch_size = train_loader.batch_size
    trues = []
    preds = []
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        z, z_tan, align_loss = model(data, data.batch_graph_nums)
        pred = model.predict(z, data)
        loss = F.cross_entropy(pred[: batch_size], data.y[: batch_size]) + align_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds.append(pred[: batch_size].detach().cpu().numpy().argmax(-1))
        trues.append(data.y[: batch_size].detach().cpu().numpy())
    preds = np.concatenate(preds, axis=-1)
    trues = np.concatenate(trues, axis=-1)
    acc = np.sum(preds == trues) / len(preds)
    loss = total_loss / len(train_loader)
    return loss, acc


def evaluate_node_cls(loader, model: RPGPrompt, device):
    model.eval()
    total_loss = 0.
    batch_size = loader.batch_size
    trues = []
    preds = []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            z, z_tan, align_loss = model(data, data.batch_graph_nums)
            pred = model.predict(z, data)
            loss = F.cross_entropy(pred[: batch_size], data.y[: batch_size]) + align_loss
            total_loss += loss.item()
            preds.append(pred[: batch_size].detach().cpu().numpy().argmax(-1))
            trues.append(data.y[: batch_size].detach().cpu().numpy())
    preds = np.concatenate(preds, axis=-1)
    trues = np.concatenate(trues, axis=-1)
    acc = np.sum(preds == trues) / len(preds)
    loss = total_loss / len(loader)
    return loss, acc


def train_graph_cls(train_loader, optimizer, model: RPGPrompt, device):
    total_loss = 0.
    trues = []
    preds = []
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        z, z_tan, align_loss = model(data, data.batch_size)
        pred = model.predict(z, data)
        loss = F.cross_entropy(pred, data.y) + align_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds.append(pred.detach().cpu().numpy().argmax(-1))
        trues.append(data.y.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=-1)
    trues = np.concatenate(trues, axis=-1)
    acc = np.sum(preds == trues) / len(preds)
    loss = total_loss / len(train_loader)
    return loss, acc


def evaluate_graph_cls(loader, model: RPGPrompt, device):
    total_loss = 0.
    trues = []
    preds = []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            z, z_tan, align_loss = model(data, data.batch_size)
            pred = model.predict(z, data)
            loss = F.cross_entropy(pred, data.y) + align_loss
            total_loss += loss.item()
            preds.append(pred.detach().cpu().numpy().argmax(-1))
            trues.append(data.y.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=-1)
    trues = np.concatenate(trues, axis=-1)
    acc = np.sum(preds == trues) / len(preds)
    loss = total_loss / len(loader)
    return loss, acc


def train_link_cls(train_loader, optimizer, model: RPGPrompt, device):
    total_loss = 0.
    trues = []
    preds = []
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        z, z_tan, align_loss = model(data, data.batch_graph_nums)
        pred = model.predict(z, data)
        loss = F.cross_entropy(pred, data.edge_type) + align_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds.append(pred.detach().cpu().numpy().argmax(-1))
        trues.append(data.edge_type.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=-1)
    trues = np.concatenate(trues, axis=-1)
    acc = np.sum(preds == trues) / len(preds)
    loss = total_loss / len(train_loader)
    return loss, acc


def evaluate_link_cls(loader, model: RPGPrompt, device):
    total_loss = 0.
    trues = []
    preds = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            z, z_tan, align_loss = model(data, data.batch_graph_nums)
            pred = model.predict(z, data)
            loss = F.cross_entropy(pred, data.edge_type) + align_loss
            total_loss += loss.item()
            preds.append(pred.detach().cpu().numpy().argmax(-1))
            trues.append(data.edge_type.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=-1)
    trues = np.concatenate(trues, axis=-1)
    acc = np.sum(preds == trues) / len(preds)
    loss = total_loss / len(loader)
    return loss, acc