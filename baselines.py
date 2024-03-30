import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch_geometric.utils import degree, to_undirected, softmax, to_scipy_sparse_matrix
from scipy.special import logsumexp
from numpy.linalg import norm, pinv
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.covariance import EmpiricalCovariance
from torch_sparse import SparseTensor, matmul
import numpy as np
import faiss


def propagation(e, edge_index, alpha=0, K=8):
    e = e.unsqueeze(1)
    N = e.shape[0]
    row, col = edge_index
    d = degree(col, N).float()
    d_norm = 1. / d[col]
    value = torch.ones_like(row) * d_norm
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    adj = adj.to_device(e.device)
    for _ in range(K):
        e = e * alpha + matmul(adj, e) * (1 - alpha)
    
    return e.squeeze(1)

class MSP():
    def __init__(self, args):
        self.dataset = args.dataset

    def inference(self, logits):
        sp = torch.softmax(logits, dim=-1)
        score, pred = sp.max(dim=-1)
        return pred, score

    def detect(self, logits):
        if self.dataset in ('proteins', 'ppi'): # for multi-label binary classification
            pass
        else: # for single-label multi-class classification
            pred, score = self.inference(logits)
        return score
        

class Energy():
    def __init__(self, args):
        self.T = args.T
        self.dataset = args.dataset

    def inference(self, logits):
        _, pred = torch.max(logits, dim=1)
        conf = self.T * torch.logsumexp(logits / self.T, dim=-1)
        return pred, conf

    def detect(self, logits):
        if self.dataset in ('proteins', 'ppi'): # for multi-label binary classification
            pass
        else: # for single-label multi-class classification
            _, neg_energy = self.inference(logits)
        return neg_energy
    
class ODIN():
    def __init__(self, args) -> None:
        super().__init__()
        self.temperature = 1000
        self.noise = args.noise #0.0014
    
    def inference(self, logits):
        sp = torch.softmax(logits / self.temperature, dim=-1)
        score, pred = sp.max(dim=-1)
        return pred, score
    
    def detect(self, logits):
        _, neg_energy = self.inference(logits)
        return neg_energy

    
class KNN():
    def __init__(self, args) -> None:
        super().__init__()
        self.K = args.neighbors
        self.activation_log = None
        self.normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

    def setup(self, net: nn.Module, dataset_ind, train_idx, device):
        net.eval()
        x, edge_index = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        
        with torch.no_grad():
            feature = net(x, edge_index)
            self.train_feature = feature

        self.activation_log = self.normalizer(feature.data.cpu().numpy())
        self.index = faiss.IndexFlatL2(feature.shape[1])
        self.index.add(self.activation_log)

    @torch.no_grad()
    def detect(self, logit, train_idx):
        feature = logit
        # setup index
        feature_normed = self.normalizer(feature.cpu().numpy())
        self.index = faiss.IndexFlatL2(feature.shape[1])
        self.index.add(feature_normed[train_idx])
        D, _ = self.index.search(
            feature_normed,
            self.K,
        )
        kth_dist = -D[:, -1]
        kth_dist = torch.from_numpy(kth_dist)
        return kth_dist
    
class GNNSafe():
    def __init__(self, args):
        self.T = args.T
        self.dataset = args.dataset

    def inference(self, logits):
        _, pred = torch.max(logits, dim=1)
        conf = self.T * torch.logsumexp(logits / self.T, dim=-1)
        return pred, conf

    def detect(self, logits, edge_index, args):
        '''return negative energy, a vector for all input nodes'''
        if self.dataset in ('proteins', 'ppi'): # for multi-label binary classification
            pass
        else: # for single-label multi-class classification
            _, scores = self.inference(logits)
        scores = propagation(scores, edge_index, alpha=args.alpha, K=args.K)
        return scores
    
class Mahalanobis(nn.Module):
    def __init__(self, args):
        super(Mahalanobis, self).__init__()

    def detect(self, logit, train_idx, test_idx, y):
        logit = logit.cpu().numpy()
        num_nodes = logit.shape[0]
        num_classes = logit.shape[1]
        scores = np.zeros(num_nodes)
        train_labels = y[train_idx]
        train_features = logit[train_idx]
        mean_cls = [ np.mean(train_features[train_labels==i], axis=0) for i in range(num_classes)]
        cov = lambda x: np.cov(x.T, bias=True)*x.shape[0]
        sigma = np.sum([cov(train_features[train_labels==i]) for i in range(num_classes)], axis=0)/len(train_idx)
        inv_sigma = np.linalg.pinv(sigma)
        def maha_score(X):
            score_cls = np.zeros((num_classes, len(X)))
            for cls in range(num_classes):
                mean = mean_cls[cls]
                z = X - mean
                score_cls[cls] = -np.sum(z * ((inv_sigma.dot(z.T)).T), axis=-1)
            return score_cls.max(0)

        scores[test_idx] = maha_score(logit[test_idx])
        return torch.as_tensor(scores)