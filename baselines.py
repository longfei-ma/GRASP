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

def propagation2(e, train_idx, edge_index, alpha=0, K=8):
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
        e[train_idx] = 1
    
    return e.squeeze(1)

def propagation_grasp(scores, dataset_ind, train_idx, test_id, test_ood, device, args):
    scores = scores.unsqueeze(1)
    test_nodes = torch.concat([test_id, test_ood])
    row, col = dataset_ind.edge_index
    if args.col: row, col = col, row
    N = dataset_ind.num_nodes
    value = torch.ones_like(row)
    adj1 = SparseTensor(row=row, col=col, value=value, sparse_sizes=(N, N))
    adj1 = adj1.to_device(device)
    add_nodes = select_G(scores, train_idx, test_nodes, adj1, args)
    scores[train_idx] = torch.where(scores[train_idx]<1, 1., scores[train_idx])

    edge_index = to_undirected(dataset_ind.edge_index)
    row, col = edge_index
    d = degree(col, N).float()
    d_add = torch.zeros(N, dtype=d.dtype)
    d_add[add_nodes] = len(add_nodes)
    d += d_add
    d_inv = 1. / d.unsqueeze(1)
    d_inv = torch.nan_to_num(d_inv, nan=0.0, posinf=0.0, neginf=0.0)
    d_norm = 1. / d[col]
    value = torch.ones_like(row) * d_norm
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    adj = adj.to_device(device)
    d_inv = d_inv.type(scores.dtype)
    d_inv = d_inv.to(device)
    e_add = torch.zeros(N, 1, dtype=scores.dtype, device=device)

    for k in range(args.K):
        e_add[add_nodes] = scores[add_nodes].sum()*d_inv[add_nodes]
        scores = scores * args.alpha + (matmul(adj, scores)+args.delta*e_add) * (1 - args.alpha)
        scores[train_idx] = torch.where(scores[train_idx]<1, 1., scores[train_idx])
        if args.adj1 and k < args.K - 1:
            add_nodes = select_G2(scores, train_idx, test_nodes, adj1, k, args)

    return scores.squeeze(1) 

def select_G2(scores, train_idx, test_nodes, adj, k, args):
    if args.test:
        nodes_use = test_nodes
    else:
        nodes_use = train_idx
    if args.tau2 == 100:
        return nodes_use.tolist()
    K = int(args.tau2/100 * len(nodes_use))
    if args.st == 'random':
        return np.random.choice(nodes_use, K, replace=False).tolist()

    scores = scores.squeeze(1)
    values = scores[test_nodes].cpu()
    if args.st == 'test':
        K = int(args.tau2/100 * len(test_nodes))
        return test_nodes[np.argpartition(values, kth=-K)[-K:]].tolist()

    #get Sid and Sood
    p = args.tau1
    thresholds1 = np.percentile(values, p)
    mask = values < thresholds1
    sood = test_nodes[mask]
    thresholds2 = np.percentile(values, 100-p)
    mask = values > thresholds2
    sid = test_nodes[mask]

    #calculate metric to select G
    N = scores.size(0)
    id_count = torch.zeros(N)
    ood_count = torch.zeros(N)
    id_count[sid] = 1
    ood_count[sood] = 1
    device = scores.device
    id_count = id_count.unsqueeze(1).to(device)
    ood_count = ood_count.unsqueeze(1).to(device)
    id_add = torch.zeros(N, 1, dtype=scores.dtype, device=device)
    ood_add = torch.zeros(N, 1, dtype=scores.dtype, device=device)

    for _ in range(k+1):
        id_count = matmul(adj, id_count) + id_add
        ood_count = matmul(adj, ood_count) + ood_add

    id_count = id_count.squeeze(1).cpu()
    ood_count = ood_count.squeeze(1).cpu()

    metrics = id_count[nodes_use]/(ood_count[nodes_use]+1)
    
    #select the top big K
    if args.st == 'top':
        return nodes_use[np.argpartition(metrics, kth=-K)[-K:]].tolist()
    #select the top small K
    elif args.st == 'low':
        return nodes_use[np.argpartition(metrics, kth=K)[: K]].tolist()
    
def select_G(scores, train_idx, test_nodes, adj, args):
    if args.tau2 == 100:
        return train_idx.tolist()
    K = int(args.tau2/100 * len(train_idx))
    if args.st == 'random':
        return np.random.choice(train_idx, K, replace=False).tolist()

    scores = scores.squeeze(1)
    values = scores[test_nodes].cpu()
    if args.st == 'test':
        K = int(args.tau2/100 * len(test_nodes))
        return test_nodes[np.argpartition(values, kth=-K)[-K:]].tolist()

    #get Sid and Sood
    p = args.tau1
    thresholds1 = np.percentile(values, p)
    mask = values < thresholds1
    sood = test_nodes[mask]
    thresholds2 = np.percentile(values, 100-p)
    mask = values > thresholds2
    sid = test_nodes[mask]

    #calculate metric to select G
    N = scores.size(0)
    id_count = torch.zeros(N)
    ood_count = torch.zeros(N)
    id_count[sid] = 1
    ood_count[sood] = 1
    device = scores.device
    id_count = id_count.unsqueeze(1).to(device)
    ood_count = ood_count.unsqueeze(1).to(device)
    id_count = matmul(adj, id_count)
    ood_count = matmul(adj, ood_count)

    id_count = id_count.squeeze(1).cpu()
    ood_count = ood_count.squeeze(1).cpu()

    metrics = id_count[train_idx]/(ood_count[train_idx]+1)
    
    #select the top big K
    if args.st == 'top':
        return train_idx[np.argpartition(metrics, kth=-K)[-K:]].tolist()
    #select the top small K
    elif args.st == 'low':
        return train_idx[np.argpartition(metrics, kth=K)[: K]].tolist()


def gdc(e, edge_index, alpha=0.1, K=8, eps=0.0001):
    e = e.unsqueeze(1)
    N = e.shape[0]
    row, col = edge_index
    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    row, col = edge_index
    d = degree(col, N).float()
    deg_inv_sqrt = d.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    d_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    value = torch.ones_like(row) * d_norm
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    adj = adj.to_device(e.device)

    adj2 = adj
    score = matmul(adj2, e) * (1 - alpha) + e * alpha
    for _ in range(K-1):
        adj2 = matmul(adj2, adj)
        adj2 = scipy.sparse.csr_matrix(adj2.to_scipy())
        d_inv = 1/adj2.sum(1).A1
        d_invsqrt = scipy.sparse.diags(np.sqrt(d_inv))
        adj2 = d_invsqrt @ adj2 @ d_invsqrt
        adj2[adj2 < eps] = 0

        adj2 = SparseTensor.from_scipy(adj2)
        score = matmul(adj2, score) * (1 - alpha) + e * alpha
    
    return e.squeeze(1)

def appnp(e, edge_index, alpha=0.1, K=8):
    e = e.unsqueeze(1)
    N = e.shape[0]
    row, col = edge_index
    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    row, col = edge_index
    d = degree(col, N).float()
    deg_inv_sqrt = d.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    d_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    value = torch.ones_like(row) * d_norm
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    adj = adj.to_device(e.device)
    score = e
    for _ in range(K):
        score = matmul(adj, score) * (1 - alpha) + e * alpha
    return score.squeeze(1)

def graph_heat(e, edge_index, K=2, alpha=0.5, s=3.5, eps=1e-4):
    e = e.unsqueeze(1) 
    N = e.shape[0]
    adjacency_matrix = to_scipy_sparse_matrix(edge_index, num_nodes=N).tocsr()
    laplacian_matrix = sp.csgraph.laplacian(adjacency_matrix, normed=False).tocsr()
    laplacian_matrix[laplacian_matrix>-np.log(eps)] = 0
    adj =  sp.linalg.expm(-s*laplacian_matrix)
    adj[adj<eps] = 0.
    # adj.eliminate_zeros()
    row, col = adj.nonzero()
    value = adj[row, col]
    row = torch.as_tensor(row, dtype=torch.long)
    col = torch.as_tensor(col, dtype=torch.long)
    value = torch.as_tensor(value, dtype=float)

    # adj = SparseTensor.from_scipy(adj.T).to(e.device)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    adj = adj.to_device(e.device)
    e = alpha * e + (1 - alpha) * matmul(adj, e)
    
    return e.squeeze(1)

def mixhop(e, edge_index, hops=2, K=2):
    e = e.unsqueeze(1)
    N = e.shape[0]
    row, col = edge_index
    d = degree(col, N).float()
    d_inv_sqrt = d.pow(-0.5)
    col_norm = d_inv_sqrt[col]
    row_norm = d_inv_sqrt[row]
    value = col_norm * row_norm
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    adj_t = adj_t.to_device(e.device)
    for _ in range(K):
        xs = [e]
        for j in range(1, hops+1):
            for hop in range(j):
                e = matmul(adj_t, e)
            xs += [e]
        e = torch.cat(xs, dim=1).mean(dim=1)
        e = e.unsqueeze(1)
    return e.squeeze(1)

def gprgnn(e, edge_index, K=10):
    e = e.unsqueeze(1)
    N = e.shape[0]
    row, col = edge_index
    d = degree(col, N).float()
    d_inv_sqrt = d.pow(-0.5)
    col_norm = d_inv_sqrt[col]
    row_norm = d_inv_sqrt[row]
    value = col_norm * row_norm
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    adj_t = adj_t.to_device(e.device)
    xs = [e]
    for _ in range(K):
        e = matmul(adj_t, e)
        xs += [e]
    return torch.cat(xs, dim=1).mean(dim=1)

class fDBD():
    def __init__(self, distance_as_normalizer=True) -> None:
        super().__init__()
        self.distance_as_normalizer = distance_as_normalizer
        self.activation_log = None

    @torch.no_grad()
    def detect(self, logit, net, train_idx):
        num_classes = logit.shape[1]
        feature = logit
        train_mean = feature[train_idx].mean(axis=0)
        w = net.convs[-1].lin.weight
        # compute denominator matrix

        denominator_matrix = np.zeros((num_classes, num_classes))
        for p in range(num_classes):
            w_p = w - w[p, :]
            denominator = np.linalg.norm(w_p, axis=1)
            denominator[p] = 1
            denominator_matrix[p, :] = denominator

        denominator_matrix = torch.as_tensor(denominator_matrix).to(logit.device)

        values, nn_idx = logit.max(1)
        logits_sub = torch.abs(logit - values.repeat(num_classes, 1).T)
        if self.distance_as_normalizer:
            score = torch.sum(logits_sub / denominator_matrix[nn_idx],
                              axis=1) / torch.norm(feature - train_mean,
                                                   dim=1)
        else:
            score = torch.sum(logits_sub / denominator_matrix[nn_idx],
                              axis=1) / torch.norm(feature, dim=1)
        return score

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

class KLM():
    def __init__(self, K=7) -> None:
        super().__init__()
    
    def kl(self, p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def setup(self, net: nn.Module, dataset_ind, dataset_ood, device):
        net.eval()
        valid_idx = dataset_ind.splits['valid']
        test_idx = dataset_ind.splits['test']
        x, edge_index = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        num_classes = torch.unique(dataset_ind.y[valid_idx]).tolist()
        
        with torch.no_grad():
            feature = net(x, edge_index).cpu().numpy()
            print('Extracting id validation feature')
            logit_id_train = feature[valid_idx]
            softmax_id_train = softmax(logit_id_train, axis=-1)
            pred_labels_train = np.argmax(softmax_id_train, axis=-1)
            self.mean_softmax_train = [
                softmax_id_train[pred_labels_train == i].mean(axis=0)
                for i in num_classes
            ]

            """ print('Extracting id testing feature')
            logit_id_val = feature[test_idx]
            softmax_id_val = softmax(logit_id_val, axis=-1)
            self.score_id = -pairwise_distances_argmin_min(
                softmax_id_val,
                np.array(self.mean_softmax_train),
                metric=self.kl)[1] """

    def detect(self, net, dataset, node_idx, device, args):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logit_ood = net(x, edge_index).cpu()
        softmax_ood = softmax(logit_ood.numpy(), axis=-1)
        score_ood = -pairwise_distances_argmin_min(
            softmax_ood, np.array(self.mean_softmax_train), metric=self.kl)[1]
        score_ood = torch.from_numpy(score_ood)
        
        if args.use_prop: # use propagation
            score_ood = score_ood.to(device)
            score_ood = propagation(score_ood, edge_index, args.K, args.alpha, args.prop_symm)
        return score_ood[node_idx]    

class DICE():
    def __init__(self) -> None:
        super().__init__()
        self.p = 90
    
    def setup(self, net: nn.Module, dataset_ind, dataset_ood, device):
        net.eval()
        train_idx = dataset_ind.splits['train']
        x, edge_index = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        with torch.no_grad():
            _, features = net(x, edge_index, return_feature_list=True)
            
        feature = features[-2]
        self.mean_act = feature[train_idx].mean(0)

    def calculate_mask(self, w):
        contrib = self.mean_act[None, :] * w
        self.thresh = np.percentile(contrib.cpu().numpy(), self.p)
        mask = torch.Tensor((contrib > self.thresh)).to(w.device)
        self.masked_w = w * mask

    def detect(self, net, dataset, node_idx, device, args):
        self.calculate_mask(net.convs[-1].lin.weight)
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        _, features = net(x, edge_index, return_feature_list=True)
        feature = features[-2]
        """ vote = feature[:, None, :] * self.masked_w
        output = vote.sum(2) + (net.convs[-2].lin.bias if net.convs[-2].lin.bias else 0) """
        net.convs[-1].lin.weight.data = self.masked_w
        output = net.convs[-1](feature, edge_index)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        if args.use_prop: # use propagation
            energyconf = energyconf.to(device)
            energyconf = propagation(energyconf, edge_index, args.K, args.alpha, args.prop_symm)
        return energyconf[node_idx]  
    
class VIM():
    def __init__(self) -> None:
        super().__init__()
        self.dim = 32
    
    def setup(self, net: nn.Module, dataset_ind, dataset_ood, device):
        net.eval()
        train_idx = dataset_ind.splits['train']
        test_idx = dataset_ind.splits['test']
        x, edge_index = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)

        with torch.no_grad():
            print('Extracting id training and testing feature')
            _, features = net(x, edge_index, return_feature_list=True)
            
        feature, logit_id = features[-2].cpu().numpy(), features[-1].cpu().numpy() 
        logit_id_train = logit_id[train_idx]     
        logit_id_val = logit_id[test_idx]   
        feature_id_train = feature[train_idx]   
        feature_id_val = feature[test_idx]   
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        self.NS = np.ascontiguousarray(
            (eigen_vectors.T[np.argsort(eig_vals * -1)[self.dim:]]).T)

        vlogit_id_train = norm(np.matmul(feature_id_train, self.NS),
                               axis=-1)
        self.alpha = logit_id_train.max(
            axis=-1).mean() / vlogit_id_train.mean()
        print(f'{self.alpha=:.4f}')

        vlogit_id_val = norm(np.matmul(feature_id_val, self.NS),
                             axis=-1) * self.alpha
        energy_id_val = logsumexp(logit_id_val, axis=-1)
        self.score_id = -vlogit_id_val + energy_id_val

    def detect(self, net, dataset, node_idx, device, args):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        _, features = net(x, edge_index, return_feature_list=True)
        feature, logit_ood = features[-2].cpu().numpy(), features[-1].cpu().numpy()
        energy_ood = logsumexp(logit_ood, axis=-1)
        vlogit_ood = norm(np.matmul(feature, self.NS),
                          axis=-1) * self.alpha
        score_ood = -vlogit_ood + energy_ood
        score_ood = torch.from_numpy(score_ood)
        if args.use_prop: # use propagation
            score_ood = score_ood.to(device)
            score_ood = propagation(score_ood, edge_index, args.K, args.alpha, args.prop_symm)
        return score_ood[node_idx]
        
class MLS():
    def __init__(self) -> None:
        super().__init__()
    
    def detect(self, net, dataset, node_idx, device, args):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits = net(x, edge_index)
        conf, _ = torch.max(logits, dim=1)
        if args.use_prop: # use propagation
            conf = conf.to(device)
            conf = propagation(conf, edge_index, args.K, args.alpha, args.prop_symm)
        return conf[node_idx]

class React():
    def __init__(self) -> None:
        super().__init__()
        self.percentile = 90
    
    def setup(self, net: nn.Module, dataset_ind, dataset_ood, device):
        net.eval()
        valid_idx = dataset_ind.splits['valid']
        x, edge_index = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)

        with torch.no_grad():
            print('Extracting id training and testing feature')
            _, features = net(x, edge_index, return_feature_list=True)
        feature = features[-2][valid_idx].cpu()
        self.threshold = np.percentile(feature.flatten(), self.percentile)
        print('Threshold at percentile {:2d} over id data is: {}'.format(
            self.percentile, self.threshold))

    def detect(self, net, dataset, node_idx, device, args):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        _, features = net(x, edge_index, return_feature_list=True)
        feature = features[-2]
        feature = feature.clip(max=self.threshold)
        output = net.convs[-1](feature, edge_index)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        if args.use_prop: # use propagation
            energyconf = energyconf.to(device)
            energyconf = propagation(energyconf, edge_index, args.K, args.alpha, args.prop_symm)
        return energyconf[node_idx]

class GradNorm():
    def __init__(self) -> None:
        super().__init__()

    def gradnorm(self, dataset_ind, feature, edge_index, conv):
        gcnconv = copy.deepcopy(conv)
        gcnconv.zero_grad()

        logsoftmax = torch.nn.LogSoftmax(dim=-1).to(feature.device)
        num_classes = len(torch.unique(dataset_ind.y))
        
        lss = logsoftmax(gcnconv(feature, edge_index))
        targets = torch.ones((1, num_classes)).to(feature.device)
        confs = []
        for ls in lss:
            loss = torch.mean(torch.sum(-targets * ls[None], dim=-1))
            loss.backward(retain_graph=True)
            layer_grad_norm = torch.sum(torch.abs(
            gcnconv.lin.weight.grad.data)).cpu().item()
            confs.append(layer_grad_norm)
            gcnconv.zero_grad()

        return torch.tensor(confs)
    
    def setup(self, net: nn.Module, dataset_ind, dataset_ood, device):
        """ net.eval()
        x, edge_index = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)

        with torch.no_grad():
            _, features = net(x, edge_index, return_feature_list=True)
            
        feature = features[-2]
        gcnconv = net.convs[-1]
        with torch.enable_grad():
            self.score_id = self.gradnorm(dataset_ind, feature, gcnconv) """
        pass

    def detect(self, net, dataset, node_idx, device, args):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        _, features = net(x, edge_index, return_feature_list=True)
        feature = features[-2]
        gcnconv = net.convs[-1]
        with torch.enable_grad():
            score_ood = self.gradnorm(dataset, feature, edge_index, gcnconv)

        if args.use_prop: # use propagation
            score_ood = score_ood.to(device)
            score_ood = propagation(score_ood, edge_index, args.K, args.alpha, args.prop_symm)
        return score_ood[node_idx]
    
class Gram():
    def __init__(self) -> None:
        super().__init__()
        self.powers = [1, 2]

    def setup(self, net: nn.Module, dataset_ind, dataset_ood, device):
        net.eval()
        x, edge_index = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        num_classes = len(torch.unique(dataset_ind.y))

        num_layer = 2 
        num_poles_list = self.powers
        num_poles = len(num_poles_list)
        feature_class = [[[None for x in range(num_poles)]
                        for y in range(num_layer)] for z in range(num_classes)]
        label_list = []
        mins = [[[None for x in range(num_poles)] for y in range(num_layer)]
                for z in range(num_classes)]
        maxs = [[[None for x in range(num_poles)] for y in range(num_layer)]
                for z in range(num_classes)]

        # collect features and compute gram metrix
        label = dataset_ind.y
        _, feature_list = net(x, edge_index, return_feature_list=True)
        label_list = label.reshape(-1).tolist()
        for layer_idx in range(num_layer):
            for pole_idx, p in enumerate(num_poles_list):
                temp = feature_list[layer_idx].detach()
                temp = temp**p
                temp = torch.matmul(temp, temp.t())
                temp = temp.sign() * torch.abs(temp)**(1 / p)
                temp = temp.data.tolist()
                for feature, label in zip(temp, label_list):
                    if isinstance(feature_class[label][layer_idx][pole_idx],
                                type(None)):
                        feature_class[label][layer_idx][pole_idx] = feature
                    else:
                        feature_class[label][layer_idx][pole_idx].extend(
                            feature)
        # compute mins/maxs
        for label in range(num_classes):
            for layer_idx in range(num_layer):
                for poles_idx in range(num_poles):
                    feature = torch.tensor(
                        np.array(feature_class[label][layer_idx][poles_idx]))
                    current_min = feature.min(dim=0, keepdim=True)[0]
                    current_max = feature.max(dim=0, keepdim=True)[0]

                    if mins[label][layer_idx][poles_idx] is None:
                        mins[label][layer_idx][poles_idx] = current_min
                        maxs[label][layer_idx][poles_idx] = current_max
                    else:
                        mins[label][layer_idx][poles_idx] = torch.min(
                            current_min, mins[label][layer_idx][poles_idx])
                        maxs[label][layer_idx][poles_idx] = torch.max(
                            current_min, maxs[label][layer_idx][poles_idx])

        self.feature_min, self.feature_max = mins, maxs

    def get_deviations(self, model, x, edge_index, mins, maxs, num_classes, powers):
        model.eval()

        num_layer = 2
        num_poles_list = powers
        exist = 1
        pred_list = []
        dev = [0 for x in range(x.shape[0])]

        # get predictions
        logits, feature_list = model(x, edge_index, return_feature_list=True)
        confs = F.softmax(logits, dim=1).cpu().detach().numpy()
        preds = np.argmax(confs, axis=1)
        predsList = preds.tolist()
        preds = torch.tensor(preds)

        for pred in predsList:
            exist = 1
            if len(pred_list) == 0:
                pred_list.extend([pred])
            else:
                for pred_now in pred_list:
                    if pred_now == pred:
                        exist = 0
                if exist == 1:
                    pred_list.extend([pred])

        # compute sample level deviation
        for layer_idx in range(num_layer):
            for pole_idx, p in enumerate(num_poles_list):
                # get gram metirx
                temp = feature_list[layer_idx].detach()
                temp = temp**p
                temp = torch.matmul(temp, temp.t())
                temp = temp.sign() * torch.abs(temp)**(1 / p)
                temp = temp.data.tolist()

                # compute the deviations with train data
                for idx in range(len(temp)):
                    dev[idx] += (F.relu(mins[preds[idx]][layer_idx][pole_idx] -
                                        sum(temp[idx])) /
                                torch.abs(mins[preds[idx]][layer_idx][pole_idx] +
                                        10**-6)).sum()
                    dev[idx] += (F.relu(
                        sum(temp[idx]) - maxs[preds[idx]][layer_idx][pole_idx]) /
                                torch.abs(maxs[preds[idx]][layer_idx][pole_idx] +
                                        10**-6)).sum()
        conf = [i / 50 for i in dev]

        return torch.tensor(conf)

    def detect(self, net, dataset, node_idx, device, args):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        num_classes = len(torch.unique(dataset.y))
        deviations = self.get_deviations(net, x, edge_index, self.feature_min,
                                           self.feature_max, num_classes,
                                           self.powers)

        if args.use_prop: # use propagation
            deviations = deviations.to(device)
            deviations = propagation(deviations, edge_index, args.K, args.alpha, args.prop_symm)
        return deviations[node_idx]


class OE(nn.Module):
    def __init__(self, d, c, args):
        super(OE, self).__init__()
        """ if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                        out_channels=c, num_layers=args.num_layers,
                        dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads)
        else:
            raise NotImplementedError """

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def detect(self, dataset, node_idx, device, args):

        logits = self.encoder(dataset.x.to(device), dataset.edge_index.to(device))[node_idx]
        if args.dataset in ('proteins', 'ppi'):
            pred = torch.sigmoid(logits).unsqueeze(-1)
            pred = torch.cat([pred, 1- pred], dim=-1)
            max_logits = pred.max(dim=-1)[0]
            return max_logits.sum(dim=1)
        else:
            return logits.max(dim=1)[0]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):

        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx

        logits_in = self.encoder(dataset_ind.x.to(device), dataset_ind.edge_index.to(device))[train_in_idx]
        logits_out = self.encoder(dataset_ood.x.to(device), dataset_ood.edge_index.to(device))[train_ood_idx]

        train_idx = dataset_ind.splits['train']
        if args.dataset in ('proteins', 'ppi'):
            loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))
        loss += 0.5 * -(logits_out.mean(1) - torch.logsumexp(logits_out, dim=1)).mean()
        return loss


        