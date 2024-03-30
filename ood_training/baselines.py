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

gpn_params = dict()
gpn_params["dim_hidden"] = 64
gpn_params["dropout_prob"] = 0.5
gpn_params["K"] = 10
gpn_params["add_self_loops"] = True
gpn_params["maf_layers"] = 0
gpn_params["gaussian_layers"] = 0
gpn_params["use_batched_flow"] = True
gpn_params["loss_reduction"] = 'sum'
gpn_params["approximate_reg"] = True
gpn_params["factor_flow_lr"] = None
gpn_params["flow_weight_decay"] = 0.0
gpn_params["pre_train_mode"] = 'flow'
gpn_params["alpha_evidence_scale"] = 'latent-new'
gpn_params["alpha_teleport"] = 0.1
gpn_params["entropy_reg"] = 0.0001
gpn_params["dim_latent"] = 32
gpn_params["radial_layers"] = 10
gpn_params["likelihood_type"] = None

from ood_training.layers import APPNPPropagation
from ood_training.layers import Density, Evidence
from ood_training.utils import Prediction, apply_mask
from ood_training.nn import uce_loss, entropy_reg

class GPN(nn.Module):
    def __init__(self, d, c, args):
        super(GPN, self).__init__()
        self.params = gpn_params
        self.params["dim_feature"] = d
        self.params["num_classes"] = c

        self.input_encoder = nn.Sequential(
            nn.Linear(d, self.params["dim_hidden"]),
            nn.ReLU(),
            nn.Dropout(p=self.params["dropout_prob"]))

        self.latent_encoder = nn.Linear(self.params["dim_hidden"], self.params["dim_latent"])

        use_batched = True if self.params["use_batched_flow"] else False
        self.flow = Density(
            dim_latent=self.params["dim_latent"],
            num_mixture_elements=c,
            radial_layers=self.params["radial_layers"],
            maf_layers=self.params["maf_layers"],
            gaussian_layers=self.params["gaussian_layers"],
            use_batched_flow=use_batched)

        self.evidence = Evidence(scale=self.params["alpha_evidence_scale"])

        self.propagation = APPNPPropagation(
            K=self.params["K"],
            alpha=self.params["alpha_teleport"],
            add_self_loops=self.params["add_self_loops"],
            cached=False,
            normalization='sym')

        self.detect_type = args.GPN_detect_type

        assert self.detect_type in ('Alea', 'Epist', 'Epist_wo_Net')
        assert self.params["pre_train_mode"] in ('encoder', 'flow', None)
        assert self.params['likelihood_type'] in ('UCE', 'nll_train', 'nll_train_and_val', 'nll_consistency', None)

    def reset_parameters(self):
        self.input_encoder = nn.Sequential(
            nn.Linear(self.params["dim_feature"], self.params["dim_hidden"]),
            nn.ReLU(),
            nn.Dropout(p=self.params["dropout_prob"]))

        self.latent_encoder = nn.Linear(self.params["dim_hidden"], self.params["dim_latent"])

        use_batched = True if self.params["use_batched_flow"] else False
        self.flow = Density(
            dim_latent=self.params["dim_latent"],
            num_mixture_elements=self.params["num_classes"],
            radial_layers=self.params["radial_layers"],
            maf_layers=self.params["maf_layers"],
            gaussian_layers=self.params["gaussian_layers"],
            use_batched_flow=use_batched)

        self.evidence = Evidence(scale=self.params["alpha_evidence_scale"])

        self.propagation = APPNPPropagation(
            K=self.params["K"],
            alpha=self.params["alpha_teleport"],
            add_self_loops=self.params["add_self_loops"],
            cached=False,
            normalization='sym')

    def forward(self, dataset):
        pred =  self.forward_impl(dataset, dataset.x.device)
        return pred.soft

    def forward_impl(self, dataset, device):
        #edge_index = dataset.edge_index.to(device) if dataset.edge_index is not None else dataset.adj_t.to(device)
        edge_index = dataset.edge_index
        x = dataset.x
        self.input_encoder.to(device)
        self.latent_encoder.to(device)
        self.flow.to(device)
        self.evidence.to(device)
        self.propagation.to(device)
        h = self.input_encoder(x)
        z = self.latent_encoder(h)

        # compute feature evidence (with Normalizing Flows)
        # log p(z, c) = log p(z | c) p(c)
        if self.training:
            p_c = self.get_class_probalities(dataset).to(device)
            self.p_c = p_c
        else:
            p_c = self.p_c
        log_q_ft_per_class = self.flow(z) + p_c.view(1, -1).log()

        if '-plus-classes' in self.params["alpha_evidence_scale"]:
            further_scale = self.params["num_classes"]
        else:
            further_scale = 1.0

        beta_ft = self.evidence(
            log_q_ft_per_class, dim=self.params["dim_latent"],
            further_scale=further_scale).exp()

        alpha_features = 1.0 + beta_ft

        beta = self.propagation(beta_ft, edge_index)
        alpha = 1.0 + beta

        soft = alpha / alpha.sum(-1, keepdim=True)
        logits = None
        log_soft = soft.log()

        max_soft, hard = soft.max(dim=-1)

        # ---------------------------------------------------------------------------------
        pred = Prediction(
            # predictions and intermediary scores
            alpha=alpha,
            soft=soft,
            log_soft=log_soft,
            hard=hard,

            logits=logits,
            latent=z,
            latent_features=z,

            hidden=h,
            hidden_features=h,

            evidence=beta.sum(-1),
            evidence_ft=beta_ft.sum(-1),
            log_ft_per_class=log_q_ft_per_class,

            # prediction confidence scores
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=alpha[torch.arange(hard.size(0)), hard],
            prediction_confidence_structure=None,

            # sample confidence scores
            sample_confidence_aleatoric=max_soft,
            sample_confidence_epistemic=alpha.sum(-1),
            sample_confidence_features=alpha_features.sum(-1),
            sample_confidence_structure=None
        )
        # ---------------------------------------------------------------------------------

        return pred

    def get_optimizer(self, lr: float, weight_decay: float):
        flow_lr = lr if self.params["factor_flow_lr"] is None else self.params["factor_flow_lr"] * lr
        flow_weight_decay = weight_decay if self.params["flow_weight_decay"] is None else self.params["flow_weight_decay"]

        flow_params = list(self.flow.named_parameters())
        flow_param_names = [f'flow.{p[0]}' for p in flow_params]
        flow_param_weights = [p[1] for p in flow_params]

        all_params = list(self.named_parameters())
        params = [p[1] for p in all_params if p[0] not in flow_param_names]

        # all params except for flow
        flow_optimizer = torch.optim.Adam(flow_param_weights, lr=flow_lr, weight_decay=flow_weight_decay)
        model_optimizer = torch.optim.Adam(
            [{'params': flow_param_weights, 'lr': flow_lr, 'weight_decay': flow_weight_decay},
             {'params': params}],
            lr=lr, weight_decay=weight_decay)

        return model_optimizer, flow_optimizer

    def get_warmup_optimizer(self, lr: float, weight_decay: float):
        model_optimizer, flow_optimizer = self.get_optimizer(lr, weight_decay)

        if self.params["pre_train_mode"] == 'encoder':
            warmup_optimizer = model_optimizer
        else:
            warmup_optimizer = flow_optimizer

        return warmup_optimizer

    def loss_compute(self, dataset_ind):
        device = dataset_ind.x.device
        train_in_idx = dataset_ind.splits['train']
        prediction = self.forward_impl(dataset_ind, device)
        y = dataset_ind.y[train_in_idx]
        alpha_train = prediction.alpha[train_in_idx]
        reg = self.params["entropy_reg"]
        return uce_loss(alpha_train, y, reduction=self.params["loss_reduction"]) + entropy_reg(alpha_train, reg, approximate=True, reduction=self.params["loss_reduction"])

    def valid_loss(self, dataset_ind, device):
        val_idx = dataset_ind.splits['valid']
        prediction = self.forward_impl(dataset_ind, device)
        y = dataset_ind.y[val_idx]
        alpha_train = prediction.alpha[val_idx]
        reg = self.params["entropy_reg"]
        return (uce_loss(alpha_train, y, reduction=self.params["loss_reduction"]) + entropy_reg(alpha_train, reg, approximate=True,reduction=self.params["loss_reduction"])).detach().cpu().item()

    def detect(self, dataset, device):
        pred = self.forward_impl(dataset, device)
        if self.detect_type == 'Alea':
            score = pred.sample_confidence_aleatoric
        elif self.detect_type == 'Epist':
            score = pred.sample_confidence_epistemic
        elif self.detect_type == 'Epist_wo_Net':
            score = pred.sample_confidence_features
        else:
            raise ValueError(f"Unknown detect type {self.detect_type}")

        return score

    def get_class_probalities(self, data):
        l_c = torch.zeros(self.params["num_classes"], device=data.x.device)
        train_idx = data.splits['train']
        y_train = data.y[train_idx]

        # calculate class_counts L(c)
        for c in range(self.params["num_classes"]):
            class_count = (y_train == c).int().sum()
            l_c[c] = class_count

        L = l_c.sum()
        p_c = l_c / L

        return p_c


from ood_training.layers import GCNConv
import ood_training.nn as unn
from ood_training.nn import loss_reduce
import torch.distributions as D
from ood_training.models.gdk import GDK
from ood_training.utils import RunConfiguration, ModelConfiguration, DataConfiguration
from ood_training.utils import TrainingConfiguration
from models import LINK, GCN, MLP, SGC, GAT, SGCMem, MultiLP, MixHop, GCNJK, GATJK, H2GCN, APPNP_Net, LINK_Concat, LINKX, GPRGNN, GCNII

class SGCN(nn.Module):
    def __init__(self, d, c, args, dataset=None):
        super(SGCN, self).__init__()
        self.params = dict()
        self.params = dict()
        self.params["seed"] = args.gkde_seed
        self.params["dim_hidden"] = args.gkde_dim_hidden
        self.params["dropout_prob"] = args.gkde_dropout_prob
        self.params["use_kernel"] = bool(args.gkde_use_kernel)
        self.params["lambda_1"] = args.gkde_lambda_1
        self.params["teacher_training"] = bool(args.gkde_teacher_training)
        self.params["use_bayesian_dropout"] = bool(args.gkde_use_bayesian_dropout)
        self.params["sample_method"] = args.gkde_sample_method
        self.params["num_samples_dropout"] = args.gkde_num_samples_dropout
        self.params["loss_reduction"] = args.gkde_loss_reduction

        self.params["dim_feature"] = d
        self.params["num_classes"] = c

        self.alpha_prior = None
        self.y_teacher = None

        self.conv1 = GCNConv(
            self.params["dim_feature"],
            self.params["dim_hidden"],
            cached=False,
            add_self_loops=True,
            normalization='sym')

        activation = []

        activation.append(nn.ReLU())
        activation.append(nn.Dropout(p=self.params["dropout_prob"]))

        self.activation = nn.Sequential(*activation)

        self.conv2 = GCNConv(
            self.params["dim_hidden"],
            self.params["num_classes"],
            cached=False,
            add_self_loops=True,
            normalization='sym')
        

        self.evidence_activation = torch.exp
        self.epoch = None

        self.detect_type = args.GPN_detect_type

        assert self.detect_type in ('Alea', 'Epist')

    def reset_parameters(self):
        self.alpha_prior = None
        self.y_teacher = None

        self.conv1 = GCNConv(
            self.params["dim_feature"],
            self.params["dim_hidden"],
            cached=False,
            add_self_loops=True,
            normalization='sym')

        activation = []

        activation.append(nn.ReLU())
        activation.append(nn.Dropout(p=self.params["dropout_prob"]))

        self.activation = nn.Sequential(*activation)

        self.conv2 = GCNConv(
            self.params["dim_hidden"],
            self.params["num_classes"],
            cached=False,
            add_self_loops=True,
            normalization='sym')

        self.evidence_activation = torch.exp
        self.epoch = None

    def forward(self, dataset):
        pred =  self.forward_impl(dataset)
        return pred.soft

    def forward_impl(self, dataset):
        edge_index = dataset.edge_index
        x = dataset.x
        device = x.device
        self.conv1 = self.conv1.to(device)
        self.conv2 = self.conv2.to(device)
        if self.training or (not self.params["use_bayesian_dropout"]):
            x = self.conv1(x, edge_index)
            x = self.activation(x)
            x = self.conv2(x, edge_index)
            evidence = self.evidence_activation(x)

        else:
            self_training = self.training
            self.train()
            samples = [None] * self.params["num_samples_dropout"]

            for i in range(self.params["num_samples_dropout"]):
                x = self.conv1(x, edge_index)
                x = self.activation(x)
                x = self.conv2(x, edge_index)
                samples[i] = x

            log_evidence = torch.stack(samples, dim=1)

            if self.params["sample_method"] == 'log_evidence':
                log_evidence = log_evidence.mean(dim=1)
                evidence = self.evidence_activation(log_evidence)

            elif self.params["sample_method"] == 'alpha':
                evidence = self.evidence_activation(log_evidence)
                evidence = evidence.mean(dim=1)

            else:
                raise AssertionError

            if self_training:
                self.train()
            else:
                self.eval()

        alpha = 1.0 + evidence
        soft = alpha / alpha.sum(-1, keepdim=True)
        max_soft, hard = soft.max(-1)

        # ---------------------------------------------------------------------------------
        pred = Prediction(
            # prediction and intermediary scores
            soft=soft,
            hard=hard,
            alpha=alpha,

            # prediction confidence scores
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=alpha[torch.arange(hard.size(0)), hard],
            prediction_confidence_structure=None,

            # sample confidence scores
            sample_confidence_aleatoric=max_soft,
            sample_confidence_epistemic=alpha.sum(-1),
            sample_confidence_features=None,
            sample_confidence_structure=None,
        )
        # ---------------------------------------------------------------------------------

        return pred

    def loss_compute(self, dataset_ind):
        if self.params["loss_reduction"] in ('sum', None):
            n_nodes = 1.0
            frac_train = 1.0

        else:
            n_nodes = dataset_ind.y.size(0)
            frac_train = dataset_ind.train_mask.float().mean()

        prediction = self.forward_impl(dataset_ind)

        alpha = prediction.alpha
        #n_nodes = data.y.size(0)
        #n_train = data.train_mask.sum()
        # bayesian risk of sum of squares
        alpha_train = alpha[dataset_ind.splits['train']]
        y = dataset_ind.y[dataset_ind.splits['train']]
        bay_risk = unn.bayesian_risk_sosq(alpha_train, y, reduction='sum')
        losses = {'BR': bay_risk * 1.0 / (n_nodes * frac_train)}

        # KL divergence w.r.t. alpha-prior from Gaussian Dirichlet Kernel
        if self.params["use_kernel"]:
            dirichlet = D.Dirichlet(alpha)
            alpha_prior = self.alpha_prior.to(alpha.device).detach()
            dirichlet_prior = D.Dirichlet(alpha_prior)
            KL_prior = D.kl.kl_divergence(dirichlet, dirichlet_prior)
            KL_prior = loss_reduce(KL_prior, reduction='sum')
            losses['KL_prior'] = self.params["lambda_1"] * KL_prior / n_nodes

        # KL divergence for teacher training
        if self.params["teacher_training"]:
            assert self.y_teacher is not None

            # currently only works for full-batch training
            # i.e. epochs == iterations
            if self.training:
                if self.epoch is None:
                    self.epoch = 0
                else:
                    self.epoch += 1

            y_teacher = self.y_teacher.to(prediction.soft.device).detach()
            lambda_2 = min(1.0, self.epoch * 1.0 / 200)
            categorical_pred = D.Categorical(prediction.soft)
            categorical_teacher = D.Categorical(y_teacher)

            KL_teacher = D.kl.kl_divergence(categorical_pred, categorical_teacher)
            KL_teacher = loss_reduce(KL_teacher, reduction='sum')
            KL_teacher = torch.nan_to_num(KL_teacher, posinf=0, neginf=0, nan=0)
            losses['KL_teacher'] = lambda_2 * KL_teacher / n_nodes

        return losses['BR'] + losses['KL_prior'] + losses['KL_teacher']

    def valid_loss(self, dataset_ind, device):

        if self.params["loss_reduction"] in ('sum', None):
            n_nodes = 1.0
            frac_train = 1.0

        else:
            n_nodes = dataset_ind.y.size(0)
            frac_train = dataset_ind.splits['valid'].float().mean()

        prediction = self.forward_impl(dataset_ind)

        alpha = prediction.alpha
        #n_nodes = data.y.size(0)
        #n_train = data.train_mask.sum()
        # bayesian risk of sum of squares
        alpha_train = alpha[dataset_ind.splits['valid']]
        y = dataset_ind.y[dataset_ind.splits['valid']]
        bay_risk = unn.bayesian_risk_sosq(alpha_train, y.to(device), reduction='sum')
        losses = {'BR': bay_risk * 1.0 / (n_nodes * frac_train)}

        # KL divergence w.r.t. alpha-prior from Gaussian Dirichlet Kernel
        if self.params["use_kernel"]:
            dirichlet = D.Dirichlet(alpha)
            alpha_prior = self.alpha_prior.to(alpha.device)
            dirichlet_prior = D.Dirichlet(alpha_prior)
            KL_prior = D.kl.kl_divergence(dirichlet, dirichlet_prior)
            KL_prior = loss_reduce(KL_prior, reduction='sum')
            losses['KL_prior'] = self.params["lambda_1"] * KL_prior / n_nodes

        # KL divergence for teacher training
        if self.params["teacher_training"]:
            assert self.y_teacher is not None

            # currently only works for full-batch training
            # i.e. epochs == iterations
            if self.training:
                if self.epoch is None:
                    self.epoch = 0
                else:
                    self.epoch += 1

            y_teacher = self.y_teacher.to(prediction.soft.device)
            lambda_2 = min(1.0, self.epoch * 1.0 / 200)
            categorical_pred = D.Categorical(prediction.soft)
            categorical_teacher = D.Categorical(y_teacher)
            KL_teacher = D.kl.kl_divergence(categorical_pred, categorical_teacher)
            KL_teacher = loss_reduce(KL_teacher, reduction='sum')
            losses['KL_teacher'] = lambda_2 * KL_teacher / n_nodes

        return losses['BR'] + losses['KL_prior'] + losses['KL_teacher']

    def create_storage(self, dataset_ind, pretrained_model, device):
        # create storage for model itself

        # create kernel and load alpha-prior
        gdk_config = ModelConfiguration(
                model_name='GDK',
                num_classes=self.params["num_classes"],
                dim_features=self.params["dim_feature"],
                seed=self.params["seed"],
                init_no=1 # GDK only with init_no = 1
        )
        kernel = GDK(gdk_config)
        prediction = kernel(dataset_ind)
        self.alpha_prior = prediction.alpha.to(device)

        x = pretrained_model(dataset_ind)
        log_soft = F.log_softmax(x, dim=-1)
        soft = torch.exp(log_soft)
        self.y_teacher = soft.to(device)

    def detect(self, dataset, device):
        pred = self.forward_impl(dataset)
        if self.detect_type == 'Alea':
            score = pred.sample_confidence_aleatoric
        elif self.detect_type == 'Epist':
            score = pred.sample_confidence_epistemic
        else:
            raise ValueError(f"Unknown detect type {self.detect_type}")

        return score.detach().cpu()

from torch_geometric.nn import MessagePassing
from torch.nn import Linear, Parameter
from torch_geometric.utils import remove_self_loops, add_self_loops
from scipy.stats import entropy
class OODGATConv(MessagePassing):
    def __init__(self, in_dim, out_dim, heads, adjust=True, concat=True, dropout=0.0,
                 add_self_loops= True, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(OODGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.adjust = adjust
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.bias = bias

        self.lin = glorot_init(in_dim, heads * out_dim)
        # The learnable parameters to compute attention coefficients:
        self.att_q = Parameter(glorot_init_2(heads, out_dim).unsqueeze(0))
        if adjust:
            self.att_v = Parameter(glorot_init_2(heads, out_dim).unsqueeze(0))
        if bias and concat:
            self.bias = Parameter(torch.zeros(heads * out_dim))
        elif bias and not concat:
            self.bias = Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        self.lin = glorot_init(self.in_dim, self.heads * self.out_dim)
        self.att_q = Parameter(glorot_init_2(self.heads, self.out_dim).unsqueeze(0))
        if self.bias is not None and self.concat:
            self.bias = Parameter(torch.zeros(self.heads * self.out_dim))
        elif self.bias is not None and not self.concat:
            self.bias = Parameter(torch.zeros(self.out_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index, return_attention_weights=False):
        H, C = self.heads, self.out_dim
        # We first transform the input node features.
        x = torch.matmul(x, self.lin).view(-1, H, C)  # x: [N, H, C]
        # Next, we compute node-level attention coefficients
        alpha = (x * self.att_q).sum(dim=-1) # alpha: [N, H]

        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        out = self.propagate(edge_index, x=x, alpha=alpha) # out: [N, H, C]

        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if return_attention_weights:
            return (out, alpha)
        else:
            return out

    def message(self, x_i, x_j, alpha_j, alpha_i, index):
        edge_weight_alpha = 1 - torch.abs(F.sigmoid(alpha_i) - F.sigmoid(alpha_j))
        if self.adjust:
            edge_weight_beta = (self.att_v * F.leaky_relu(x_i + x_j)).sum(-1)
            edge_weight = edge_weight_alpha * edge_weight_beta
        else:
            edge_weight = edge_weight_alpha
        edge_weight = softmax(edge_weight, index)
        edge_weight = F.dropout(edge_weight, p=self.dropout, training=self.training)

        return x_j * edge_weight.unsqueeze(-1)

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return Parameter(initial)

def glorot_init_2(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return initial

class EntropyLoss(nn.Module):
    '''
    return: mean entropy of the given batch if reduction is True, n-dim vector of entropy if reduction is False.
    '''
    def __init__(self, reduction=True):
        super(EntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        if self.reduction:
            b = -1.0 * b.sum()
            b = b / x.shape[0]
        else:
            b = -1.0 * b.sum(axis=1)

        return b

def cosine_similarity(x1, x2, reduction=True):
    '''
    compute cosine similarity between x1 and x2.
    :param x1: N * D tensor or 1d tensor.
    :param x2: N * D tensor or 1d tensor.
    :return: a scalar tensor if reduction is True, a tensor of shape N if reduction is False.
    '''
    cos_sim = nn.CosineSimilarity(dim=-1)
    if reduction:
        sim = cos_sim(x1, x2).mean()
    else:
        sim = cos_sim(x1, x2)

    return sim

def get_consistent_loss_new(x1, x2, f1=None, f2=None):
    '''
    compute consistent loss between attention scores and output entropy.
    :param x1: ood score matrix, H * N tensor. the larger, the more likely to be ood.
    :param x2: entropy vector, N-dim tensor.
    :return: scalar tensor of computed loss.
    '''
    x1 = x1.mean(axis=0)
    if f1 is not None:
        x1 = f1(x1)
    if f2 is not None:
        x2 = f2(x2)
    loss = cosine_similarity(x1, x2)

    return -1.0 * loss

class CE_uniform(nn.Module):
    '''
    return: CE of the given batch if reduction is True, n-dim vector of CE if reduction is False.
    '''
    def __init__(self, n_id_classes, reduction=True):
        super(CE_uniform, self).__init__()
        self.reduction = reduction
        self.n_id_classes = n_id_classes

    def forward(self, x):
        b = (1/self.n_id_classes) * F.log_softmax(x, dim=1)
        if self.reduction:
            b = -1.0 * b.sum()
            b = b / x.shape[0]
        else:
            b = -1.0 * b.sum(axis=1)

        return b

def local_ent_loss(logits, att, n_id_classes, m=0.5):
    att_norm = F.sigmoid(torch.hstack([att[0], att[1]]).mean(axis=1)).detach()  # n-dim
    mask = torch.ge(att_norm - m, 0)
    ce_uni = CE_uniform(n_id_classes, reduction=False)
    ce = ce_uni(logits)  # N-dim
    if mask.sum() > 0:
        loss = ce[mask].mean()
    else:
        loss = 0

    return loss

class OODGAT(nn.Module):
    def __init__(self, in_dim, out_dim, adjust=True, add_self_loop=True, bias=True, args=None):
        super(OODGAT, self).__init__()

        self.conv1 = OODGATConv(in_dim, args.hidden_dim, args.heads, adjust, True, args.drop_edge, add_self_loop, bias)
        self.conv2 = OODGATConv(args.hidden_dim * args.heads, out_dim, args.heads, adjust, False, args.drop_edge, add_self_loop, bias)
        self.drop_prob = args.drop_prob
        self.drop_input = args.drop_input
        self.OODGAT_detect_type = args.OODGAT_detect_type


    def forward(self, data, return_attention_weights=False):
        x, edge_index = data.x, data.edge_index
        if not return_attention_weights:
            x = F.dropout(x, p=self.drop_input, training=self.training)
            x = F.elu(self.conv1(x, edge_index, False))
            x = F.dropout(x, p=self.drop_prob, training=self.training)
            x = self.conv2(x, edge_index, False)
            return x
        else:
            attention = []
            x = F.dropout(x, p=self.drop_input, training=self.training)
            x, a = self.conv1(x, edge_index, True)
            attention.append(a)
            x = F.elu(x)
            x = F.dropout(x, p=self.drop_prob, training=self.training)
            x, a = self.conv2(x, edge_index, True)
            attention.append(a)
            return (x, attention)
    
    def loss_compute(self, dataset_ind, device, args, epoch):
        a = torch.tensor(0.9).to(device)
        b = torch.tensor(0.01).to(device)
        
        xent = nn.CrossEntropyLoss()
        ent_loss_func = EntropyLoss(reduction=False)
        loss = torch.zeros(1).to(device)
        logits, att = self.forward(dataset_ind, return_attention_weights=True)
        if args.w_consistent is not None and args.w_consistent > 0:
            ent_loss = ent_loss_func(logits)  # ent_loss: N-dim tensor
            cos_loss_1 = get_consistent_loss_new(att[0].T, (ent_loss - ent_loss.mean()) / ent_loss.std(),
                                                    f1=F.sigmoid, f2=F.sigmoid)
            cos_loss_2 = get_consistent_loss_new(att[1].T, (ent_loss - ent_loss.mean()) / ent_loss.std(),
                                                    f1=F.sigmoid, f2=F.sigmoid)
            consistent_loss = 0.5 * (cos_loss_1 + cos_loss_2)
            loss += torch.pow(a, b * epoch) * args.w_consistent * consistent_loss
        if args.w_discrepancy is not None and args.w_discrepancy > 0:
            loss -= torch.pow(a, b * epoch) * args.w_discrepancy * cosine_similarity(att[0].mean(axis=1), att[1].mean(axis=1))
        if args.w_ent is not None and args.w_ent > 0:
            loss += torch.pow(a, b * epoch) * args.w_ent * local_ent_loss(logits, att, dataset_ind.y.max().item() + 1, args.margin)

        sup_loss = xent(logits[dataset_ind.splits['train']], dataset_ind.y.squeeze()[dataset_ind.splits['train']])
        loss += sup_loss
        return loss

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    
    def detect(self, dataset, device):
        logits, a = self.forward(dataset, return_attention_weights=True)
        a = [a[0].detach().cpu(), a[1].detach().cpu()]
        # OOD detection
        pred_dist = F.softmax(logits, dim=1).detach().cpu()
        if self.OODGAT_detect_type == 'ATT':
            score = F.sigmoid(torch.hstack([a[0], a[1]]).mean(axis=1))
        elif self.OODGAT_detect_type == 'ENT':
            score = entropy(pred_dist, axis=1)
            score = torch.as_tensor(score)
        
        return -score

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


        