import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Dict, Tuple
import numpy as np
from downstream.adapter import RPGPrompt


def train_node_cls(train_loader, optimizer, model: RPGPrompt, device):
    total_loss = 0.
    trues = []
    preds = []
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        z, z_tan = model(data, data.batch_graph_nums)
        pred = model.predict(z, data)
        loss = F.cross_entropy(pred[: data.batch_size], data.y[: data.batch_size]) + model.loss(z_tan)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds.append(pred[: data.batch_size].detach().cpu().numpy().argmax(-1))
        trues.append(data.y[: data.batch_size].detach().cpu().numpy())
    preds = np.concatenate(preds, axis=-1)
    trues = np.concatenate(trues, axis=-1)
    acc = np.sum(preds == trues) / len(preds)
    loss = total_loss / len(train_loader)
    return loss, acc


def evaluate_node_cls(loader, model: RPGPrompt, device):
    total_loss = 0.
    trues = []
    preds = []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            z, z_tan = model(data, data.batch_graph_nums)
            pred = model.predict(z, data)
            loss = F.cross_entropy(pred[: data.batch_size], data.y[: data.batch_size]) + model.loss(z_tan)
            total_loss += loss.item()
            preds.append(pred[: data.batch_size].detach().cpu().numpy().argmax(-1))
            trues.append(data.y[: data.batch_size].detach().cpu().numpy())
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
        z, z_tan = model(data, data.batch_size)
        pred = model.predict(z, data)
        loss = F.cross_entropy(pred, data.y) + model.loss(z_tan)
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
            z, z_tan = model(data, data.batch_size)
            pred = model.predict(z, data)
            loss = F.cross_entropy(pred, data.y) + model.loss(z_tan)
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
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        z, z_tan = model(data, data.batch_graph_nums)
        pred = model.predict(z, data)
        loss = F.cross_entropy(pred, data.edge_type) + model.loss(z_tan)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss = total_loss / len(train_loader)
    return loss