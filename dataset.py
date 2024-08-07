from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io
import pickle
import pandas as pd
from sklearn.preprocessing import label_binarize
import gdown
from os import path
import os
import torch_geometric.transforms as T

from load_data import load_twitch, load_fb100, load_twitch_gamer, DATAPATH
from data_utils import rand_splits, rand_train_test_idx, even_quantile_labels, set_random_seed, dataset_drive_url

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Reddit2, Actor, WebKB
from torch_geometric.transforms import NormalizeFeatures
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, subgraph, mask_to_index, to_undirected
from torch_geometric.utils.map import map_index
from ogb.nodeproppred import NodePropPredDataset


class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None
        self.node_idx = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_dataset(args, sub_dataname=''):
    """ Loader for NCDataset, returns NCDataset. """
    if args.dataset == 'twitch-e':
        # twitch-explicit graph
        if sub_dataname not in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'):
            print('Invalid sub_dataname, deferring to DE graph')
            sub_dataname = 'DE'
        dataset = load_twitch_dataset(sub_dataname)
    elif args.dataset == 'fb100':
        if sub_dataname not in ('Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 'Reed98'):
            print('Invalid sub_dataname, deferring to Penn94 graph')
            sub_dataname = 'Penn94'
        dataset = load_fb100_dataset(sub_dataname)
    elif args.dataset == 'ogbn-proteins':
        dataset = load_proteins_dataset()
    elif args.dataset == 'deezer-europe':
        dataset = load_deezer_dataset()
    elif args.dataset == 'arxiv-year':
        dataset_ind, dataset_ood_te = load_arxiv_year_dataset()
    elif args.dataset == 'pokec':
        dataset = load_pokec_mat()
    elif args.dataset == 'snap-patents':
        dataset_ind, dataset_ood_te = load_snap_patents_mat()
    elif args.dataset == 'yelp-chi':
        dataset = load_yelpchi_dataset()
    elif args.dataset in ('ogbn-arxiv', 'ogbn-products'):
        dataset_ind, dataset_ood_te = load_ogb_dataset(args.dataset)
    elif args.dataset in ('Cora', 'CiteSeer', 'PubMed'):
        dataset = load_planetoid_dataset(args.dataset)
    elif args.dataset in ('chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin'):
        dataset_ind, dataset_ood_te = load_geom_gcn_dataset(args.dataset)
    elif args.dataset == "genius":
        dataset = load_genius()
    elif args.dataset == "twitch-gamer":
        dataset = load_twitch_gamer_dataset() 
    elif args.dataset == "wiki":
        dataset_ind, dataset_ood_te = load_wiki()
    elif args.dataset in ('cora', 'amazon-photo', 'coauthor-cs', 'reddit2'):
        dataset_ind, dataset_ood_te = load_graph_dataset(args)
    else:
        raise ValueError('Invalid dataname')
    return dataset_ind, dataset_ood_te

def create_feat_noise_dataset(data):

    x = data.x
    n = data.num_nodes
    torch.manual_seed(111)
    idx = torch.randint(0, n, (n, 2))
    torch.manual_seed(222)
    weight = torch.rand(n).unsqueeze(1)
    x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)

    dataset = Data(x=x_new, edge_index=data.edge_index, y=data.y)
    dataset.node_idx = torch.arange(n)

    # if hasattr(data, 'train_mask'):
    #     tensor_split_idx = {}
    #     idx = torch.arange(data.num_nodes)
    #     tensor_split_idx['train'] = idx[data.train_mask]
    #     tensor_split_idx['valid'] = idx[data.val_mask]
    #     tensor_split_idx['test'] = idx[data.test_mask]
    #
    #     dataset.splits = tensor_split_idx

    return dataset

def create_sbm_dataset(data, p_ii=1.5, p_ij=0.5):
    from torch_geometric.utils import stochastic_blockmodel_graph
    n = data.num_nodes
    d = data.edge_index.size(1) / data.num_nodes / (data.num_nodes - 1)
    num_blocks = int(data.y.max()) + 1
    p_ii, p_ij = p_ii * d, p_ij * d
    block_size = n // num_blocks
    block_sizes = [block_size for _ in range(num_blocks-1)] + [block_size + n % block_size]
    edge_probs = torch.ones((num_blocks, num_blocks)) * p_ij
    edge_probs[torch.arange(num_blocks), torch.arange(num_blocks)] = p_ii
    edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs)

    dataset = Data(x=data.x, edge_index=edge_index, y=data.y)
    dataset.node_idx = torch.arange(dataset.num_nodes)
    return dataset

def structure_shift_dataset(data, run, args, ood_budget_per_graph=0.1):
    """perturb graph's egdges using the DICE attack

    Args:
        data (torch_geometric.data.Data): data object representing the graph
        ood_budget_per_graph (float, optional): fraction of perturbed edges in the graph. Defaults to 0.1.

    Returns:
        Tuple[torch_geometric.data.Data, None]: tuple of perturbed graph data and None (for pipeline compatibility)
    """
    #performs a random attack on the graph structure
    #budget corresponds to perturbed edges
    #for each targeted community:
    #    - remove delta = budget[%] x intra-community edges
    #    - add delta cross-community edges vu, s.t. c_v != c_u
    #'''
    seed = run + args.seed
    idx = torch.arange(data.num_nodes)

    # extract directed edges of undirected edge_index
    start, end = data.edge_index
    mask = start < end
    start, end = start[mask], end[mask]

    # extract nodes belonging to community
    num_classes = data.y.max() + 1
    nodes_per_class = [None] * num_classes
    for c in range(num_classes):
        nodes_per_class[c] = torch.where(data.y == c)[0].tolist() #得到每个class对应的node

    ood_idx = []
    ood_edge_index = []

    # for each community
    for c in range(num_classes):
        deleted_edges = []
        intra_community_edges = torch.where((data.y[start] == c) & (data.y[end] == c))[0].tolist()#每个社区的intra-edges

        # find all nodes from other communities
        cross_community_nodes = []
        for c_j in range(num_classes):
            if c_j == c:
                continue
            cross_community_nodes.extend(nodes_per_class[c_j])

        # calulcate budget
        n_intra_edges = len(intra_community_edges)
        budget = int(ood_budget_per_graph * n_intra_edges)

        # sample deleted
        set_seed(seed)
        deleted_edges.extend(np.random.choice(intra_community_edges, budget, replace=False))#待去除的intra-edges

        # first: sample community nodes, then sample corresponding cross-community nodes
        n_c = nodes_per_class[c]
        # set_seed(seed)
        # community_nodes = np.random.choice(n_c, budget, replace=True) #本社区选中的节点
        community_nodes = start[deleted_edges] #本社区选中的节点
        ood_idx.extend(community_nodes.tolist())
        set_seed(seed)
        cross_community_nodes = np.random.choice(cross_community_nodes, budget, replace=True) #外社区选中的节点
        ood_edge_index.append(torch.vstack([community_nodes, cross_community_nodes]))
        # end[deleted_edges] = torch.as_tensor(cross_community_nodes)
        ood_idx.extend(cross_community_nodes.tolist())

    ood_edge_index = torch.concat(ood_edge_index, dim=1)
    ood_edge_index = to_undirected(ood_edge_index)
    ood_idx = np.unique(ood_idx)
    ood_idx = torch.as_tensor(ood_idx)
    ood_edge_index, _ = map_index(ood_edge_index.view(-1), ood_idx, inclusive=True)
    ood_edge_index = ood_edge_index.view(2, -1)
    # ood_edge_index = subgraph(ood_idx, edge_index, relabel_nodes=True)[0]
    dataset_ood = Data(x=data.x[ood_idx], edge_index=ood_edge_index, y=data.y[ood_idx])

    edge_index = data.edge_index
    id_mask = torch.ones(data.num_nodes, dtype=bool)
    id_mask[ood_idx] = 0
    id_train_idx = mask_to_index(id_mask)
    set_seed(seed)
    split_idx = rand_splits(id_train_idx, args.train_prop, args.valid_prop)
    test_id_idx = split_idx['test']
    test_id_edge_index = subgraph(test_id_idx, edge_index, relabel_nodes=True)[0]
    dataset_test_id = Data(x=data.x[test_id_idx], edge_index=test_id_edge_index, y=data.y[test_id_idx])

    train_idx = torch.concat([split_idx['train'], split_idx['valid']])
    train_id_edge_index = subgraph(train_idx, edge_index, relabel_nodes=True)[0]
    dataset_train_id = Data(x=data.x[train_idx], edge_index=train_id_edge_index, y=data.y[train_idx])
    new_split_idx = {'train': map_index(split_idx['train'], train_idx, inclusive=True)[0], 'valid': map_index(split_idx['valid'], train_idx, inclusive=True)[0]}
    dataset_train_id.split_idx = new_split_idx

    return dataset_train_id, dataset_test_id, dataset_ood

def feature_shift_dataset(data, run, args, ood_budget_per_graph=0.1):
    """perturb graph's egdges using the DICE attack

    Args:
        data (torch_geometric.data.Data): data object representing the graph
        ood_budget_per_graph (float, optional): fraction of perturbed edges in the graph. Defaults to 0.1.

    Returns:
        Tuple[torch_geometric.data.Data, None]: tuple of perturbed graph data and None (for pipeline compatibility)
    """
    #performs a random attack on the graph structure
    #budget corresponds to perturbed edges
    #for each targeted community:
    #    - remove delta = budget[%] x intra-community edges
    #    - add delta cross-community edges vu, s.t. c_v != c_u
    #'''
    seed = run + args.seed
    data.node_idx = torch.arange(data.num_nodes)
    

def load_graph_dataset(args):
    dataname = args.dataset
    ood_type = args.ood_type
    transform = T.NormalizeFeatures()
    if dataname in ('cora', 'citeseer', 'pubmed'):
        torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid', split='public',
                              name=dataname, transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'amazon-photo':
        torch_dataset = Amazon(root=f'{DATAPATH}Amazon',
                               name='Photo', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'amazon-computer':
        torch_dataset = Amazon(root=f'{DATAPATH}Amazon',
                               name='Computers', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'coauthor-cs':
        torch_dataset = Coauthor(root=f'{DATAPATH}Coauthor',
                                 name='CS', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'reddit2':
        torch_dataset = Reddit2(root=f'{DATAPATH}Reddit2',
                                transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'coauthor-physics':
        torch_dataset = Coauthor(root=f'{DATAPATH}Coauthor',
                                 name='Physics', transform=transform)
        dataset = torch_dataset[0]
    else:
        raise NotImplementedError

    if ood_type == 'structure':
        dataset_ind, dataset_test_id, dataset_ood = structure_shift_dataset(dataset, args)
        dataset_ood_te = (dataset_test_id, dataset_ood)
    elif ood_type == 'feature':
        dataset_ind = dataset
        dataset_ood_te = create_feat_noise_dataset(dataset)
    elif ood_type == 'label':
        label = dataset.y
        if dataname == 'cora':
            class_t = 4
        elif dataname == 'amazon-photo':
            class_t = 5
        elif dataname == 'coauthor-cs':
            class_t = 5
        elif dataname == 'reddit2':
            class_t = 11
        
        y = label - class_t
        y = torch.where(y<0, -1, y)
        dataset_ind = Data(x=dataset.x, edge_index=dataset.edge_index, y=y)
        idx = torch.arange(label.size(0))
        dataset_ind.node_idx = idx[y>=0]
        dataset_ood_te = Data(x=dataset.x, edge_index=dataset.edge_index, y=y)
        dataset_ood_te.node_idx = idx[y<0]
    else:
        raise NotImplementedError
    return dataset_ind, dataset_ood_te


def load_twitch_dataset(lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    A, label, features = load_twitch(lang)
    dataset = NCDataset(lang)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_fb100_dataset(filename):
    A, metadata = load_fb100(filename)
    dataset = NCDataset(filename)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_deezer_dataset():
    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{DATAPATH}deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_arxiv_year_dataset(nclass=5):
    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root="data")
    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    label = even_quantile_labels(
        ogb_dataset.graph['node_year'].flatten(), nclass, verbose=False)
    label = torch.as_tensor(label)
    y = label - 2
    y = torch.where(y<0, -1, y)
    dataset_ind = Data(x=node_feat, edge_index=edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[y>=0]
    dataset_ood_te = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_te.node_idx = idx[y<0]
    return dataset_ind, dataset_ood_te

    """ dataset_ind = Data(x=node_feat, edge_index=edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[label>=2]
    
    dataset_ood_tr = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_tr.node_idx = idx[label==1]

    dataset_ood_te = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_te.node_idx = idx[label==0]

    return dataset_ind, dataset_ood_tr, dataset_ood_te """


def load_proteins_dataset():
    ogb_dataset = NodePropPredDataset(name='ogbn-proteins')
    dataset = NCDataset('ogbn-proteins')

    def protein_orig_split(**kwargs):
        split_idx = ogb_dataset.get_idx_split()
        return {'train': torch.as_tensor(split_idx['train']),
                'valid': torch.as_tensor(split_idx['valid']),
                'test': torch.as_tensor(split_idx['test'])}

    dataset.get_idx_split = protein_orig_split
    dataset.graph, dataset.label = ogb_dataset.graph, ogb_dataset.labels

    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['edge_feat'] = torch.as_tensor(dataset.graph['edge_feat'])
    dataset.label = torch.as_tensor(dataset.label)
    return dataset


def load_ogb_dataset(dname):
    ogb_dataset = NodePropPredDataset(name=dname, root=f'{DATAPATH}ogb')
    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    label = torch.as_tensor(ogb_dataset.labels).squeeze()

    dataset = Data(x=node_feat, edge_index=edge_index, y=label)
    dataset.node_idx = torch.arange(dataset.num_nodes)

    if dname == 'ogbn-products':
        class_t = 12
    
    y = label - class_t
    y = torch.where(y<0, -1, y)
    dataset_ind = Data(x=dataset.x, edge_index=dataset.edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[y>=0]
    dataset_ood_te = Data(x=dataset.x, edge_index=dataset.edge_index, y=y)
    dataset_ood_te.node_idx = idx[y<0]
    return dataset_ind, dataset_ood_te


def load_pokec_mat():
    """ requires pokec.mat
    """
    if not path.exists(f'{DATAPATH}pokec.mat'):
        gdown.download(id=dataset_drive_url['pokec'], \
            output=f'{DATAPATH}pokec.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset


def load_snap_patents_mat(nclass=5):
    if not path.exists(f'{DATAPATH}snap_patents.mat'):
        p = dataset_drive_url['snap-patents']
        print(f"Snap patents url: {p}")
        gdown.download(id=dataset_drive_url['snap-patents'], \
            output=f'{DATAPATH}snap_patents.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}snap_patents.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(
        fulldata['node_feat'].todense(), dtype=torch.float)
    
    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    label = torch.tensor(label, dtype=torch.long)
    y = label - 2
    y = torch.where(y<0, -1, y)
    dataset_ind = Data(x=node_feat, edge_index=edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[y>=0]
    dataset_ood_te = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_te.node_idx = idx[y<0]
    return dataset_ind, dataset_ood_te
    """ dataset_ind = Data(x=node_feat, edge_index=edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[label>=2]
    
    dataset_ood_tr = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_tr.node_idx = idx[label==1]

    dataset_ood_te = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_te.node_idx = idx[label==0]

    return dataset_ind, dataset_ood_tr, dataset_ood_te """


def load_yelpchi_dataset():
    if not path.exists(f'{DATAPATH}YelpChi.mat'):
        gdown.download(id=dataset_drive_url['yelp-chi'], \
            output=f'{DATAPATH}YelpChi.mat', quiet=False)
    fulldata = scipy.io.loadmat(f'{DATAPATH}YelpChi.mat')
    A = fulldata['homo']
    edge_index = np.array(A.nonzero())
    node_feat = fulldata['features']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset('YelpChi')
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat.todense(), dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset


def load_planetoid_dataset(name):
    torch_dataset = Planetoid(root=f'{DATAPATH}/Planetoid',
                              name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}

    def planetoid_orig_split(**kwargs):
        return {'train': torch.as_tensor(dataset.train_idx),
                'valid': torch.as_tensor(dataset.valid_idx),
                'test': torch.as_tensor(dataset.test_idx)}

    dataset.get_idx_split = planetoid_orig_split
    dataset.label = label

    return dataset


def load_geom_gcn_dataset(name):
    fulldata = scipy.io.loadmat(f'{DATAPATH}/{name}.mat')
    edge_index = torch.as_tensor(fulldata['edge_index'], dtype=torch.long)
    edge_index = remove_self_loops(edge_index)[0]
    node_feat = torch.as_tensor(fulldata['node_feat'])
    label = np.array(fulldata['label'], dtype=int).flatten()
    label = torch.as_tensor(label)
    y = label - 2
    y = torch.where(y<0, -1, y)
    dataset_ind = Data(x=node_feat, edge_index=edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[y>=0]
    dataset_ood_te = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_te.node_idx = idx[y<0]
    return dataset_ind, dataset_ood_te

    """ torch_dataset = torch.load(f'{DATAPATH}/{name}.pt')
    dataset = torch_dataset[0]
    dataset.node_idx = torch.arange(dataset.num_nodes)
    label = dataset.y
    y = label - 2
    y = torch.where(y<0, -1, y)
    dataset_ind = Data(x=dataset.x, edge_index=dataset.edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[y>=0]
    dataset_ood_te = Data(x=dataset.x, edge_index=dataset.edge_index, y=y)
    dataset_ood_te.node_idx = idx[y<0]
    return dataset_ind, dataset_ood_te """

    """ y = label - 2
    y = torch.where(y<0, -1, y)
    dataset_ind = Data(x=node_feat, edge_index=edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[label>=2]
    
    dataset_ood_tr = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_tr.node_idx = idx[label==1]

    dataset_ood_te = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_te.node_idx = idx[label==0]

    return dataset_ind, dataset_ood_tr, dataset_ood_te """


def load_genius():
    filename = 'genius'
    dataset = NCDataset(filename)
    fulldata = scipy.io.loadmat(f'data/genius.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
    label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_twitch_gamer_dataset(task="mature", normalize=True):
    if not path.exists(f'{DATAPATH}twitch-gamer_feat.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_feat'],
            output=f'{DATAPATH}twitch-gamer_feat.csv', quiet=False)
    if not path.exists(f'{DATAPATH}twitch-gamer_edges.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_edges'],
            output=f'{DATAPATH}twitch-gamer_edges.csv', quiet=False)
    
    edges = pd.read_csv(f'{DATAPATH}twitch-gamer_edges.csv')
    nodes = pd.read_csv(f'{DATAPATH}twitch-gamer_feat.csv')
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)
    label, features = load_twitch_gamer(nodes, task)
    node_feat = torch.tensor(features, dtype=torch.float)
    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
    dataset = NCDataset("twitch-gamer")
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_wiki():

    if not path.exists(f'{DATAPATH}wiki_features2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_features'], \
            output=f'{DATAPATH}wiki_features2M.pt', quiet=False)
    
    if not path.exists(f'{DATAPATH}wiki_edges2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_edges'], \
            output=f'{DATAPATH}wiki_edges2M.pt', quiet=False)

    if not path.exists(f'{DATAPATH}wiki_views2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_views'], \
            output=f'{DATAPATH}wiki_views2M.pt', quiet=False)


    node_feat = torch.load(f'{DATAPATH}wiki_features2M.pt')
    edge_index = torch.load(f'{DATAPATH}wiki_edges2M.pt').T
    #print(f"edges shape: {edge_index.shape}")
    label = torch.load(f'{DATAPATH}wiki_views2M.pt') 
    #print(f"features shape: {node_feat.shape[1]}")
    #print(f"Label shape: {label.shape[0]}")

    y = label - 2
    y = torch.where(y<0, -1, y)
    dataset_ind = Data(x=node_feat, edge_index=edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[y>=0]
    dataset_ood_te = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_te.node_idx = idx[y<0]
    return dataset_ind, dataset_ood_te


    """ dataset_ind = Data(x=node_feat, edge_index=edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[label>=2]
    
    dataset_ood_tr = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_tr.node_idx = idx[label==1]

    dataset_ood_te = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_te.node_idx = idx[label==0]

    return dataset_ind, dataset_ood_tr, dataset_ood_te """

