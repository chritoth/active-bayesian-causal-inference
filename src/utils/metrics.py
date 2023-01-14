import networkx as nx
import torch
from cdt.metrics import SHD
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def auroc(posterior_edge_probs: torch.Tensor, true_adj_mat: torch.Tensor):
    assert posterior_edge_probs.squeeze().shape == true_adj_mat.squeeze().shape
    edge_probs = posterior_edge_probs.detach().view(-1).numpy()
    targets = true_adj_mat.int().view(-1).numpy()

    fpr, tpr, _ = roc_curve(targets, edge_probs)
    return auc(fpr, tpr)


def auprc(posterior_edge_probs: torch.Tensor, true_adj_mat: torch.Tensor):
    assert posterior_edge_probs.squeeze().shape == true_adj_mat.squeeze().shape
    edge_probs = posterior_edge_probs.detach().view(-1).numpy()
    targets = true_adj_mat.int().view(-1).numpy()

    precision, recall, _ = precision_recall_curve(targets, edge_probs)
    return auc(recall, precision)


def shd(target_graph: nx.DiGraph, predicted_graph: nx.DiGraph):
    return torch.tensor(SHD(target_graph, predicted_graph)).float()
