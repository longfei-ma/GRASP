import torch
from torch_geometric.utils import degree, to_undirected
from torch_sparse import SparseTensor, matmul
    
class GRASP():
    def __init__(self, args):
        self.T = args.T
        self.dataset = args.dataset

    def inference(self, logits, score='MSP'):
        if score == 'Energy':
            _, pred = torch.max(logits, dim=1)
            score = self.T * torch.logsumexp(logits / self.T, dim=-1)
        elif score == 'MSP':
            sp = torch.softmax(logits, dim=-1)
            score, pred = sp.max(dim=-1)
        return pred, score

    def detect(self, logits, edge_index, add_nodes, args):
        if self.dataset in ('proteins', 'ppi'): # for multi-label binary classification
            pass
        else: # for single-label multi-class classification
            _, scores = self.inference(logits)
        
        scores = self.propagation(scores, edge_index, add_nodes, alpha=args.alpha, K=args.K, delta=args.delta)
        return scores
    
    def propagation(self, e, edge_index, add_nodes, alpha=0, K=8, delta=1.):
        e = e.unsqueeze(1)
        N = e.shape[0]
        edge_index = to_undirected(edge_index)
        row, col = edge_index
        d = degree(col, N).float()
        d_add = torch.zeros(N, dtype=d.dtype, device=d.device)
        d_add[add_nodes] = len(add_nodes)
        d += d_add
        d_inv = 1. / d.unsqueeze(1)
        d_inv = torch.nan_to_num(d_inv, nan=0.0, posinf=0.0, neginf=0.0)
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        adj = adj.to_device(e.device)
        d_inv = d_inv.type(e.dtype)
        d_inv = d_inv.to(e.device)
        e_add = torch.zeros(N, 1, dtype=e.dtype, device=e.device)
        for _ in range(K):
            e_add[add_nodes] =  e[add_nodes].sum()*d_inv[add_nodes]
            e = e * alpha + (matmul(adj, e)+delta*e_add) * (1 - alpha)
        return e.squeeze(1)