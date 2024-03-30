from models import LINK, GCN, MLP, SGC, GAT, SGCMem, MultiLP, MixHop, GCNJK, GATJK, H2GCN, APPNP_Net, LINK_Concat, LINKX, GPRGNN, GCNII, GIN, GraphSAGE
from data_utils import normalize


def parse_method(args, dataset, n, c, d, device='cpu'):
    if args.method == 'link':
        model = LINK(n, c).to(device)
    elif args.method == 'gcn':
        if args.dataset == 'ogbn-proteins':
            # Pre-compute GCN normalization.
            dataset.graph['edge_index'] = normalize(dataset.graph['edge_index'])
            model = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        dropout=args.dropout,
                        save_mem=True,
                        use_bn=not args.no_bn).to(device)
        else:
            model = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=not args.no_bn).to(device)

    elif args.method == 'mlp' or args.method == 'cs':
        model = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                    out_channels=c, num_layers=args.num_layers,
                    dropout=args.dropout).to(device)
    elif args.method == 'sgc':
        if args.cached:
            model = SGC(in_channels=d, out_channels=c, hops=args.hops).to(device)
        else:
            model = SGCMem(in_channels=d, out_channels=c,
                           hops=args.hops).to(device)
    elif args.method == 'gprgnn':
        model = GPRGNN(d, args.hidden_channels, c, alpha=args.gpr_alpha, num_layers=args.num_layers, dropout=args.dropout).to(device)
    elif args.method == 'appnp':
        model = APPNP_Net(d, args.hidden_channels, c, alpha=args.gpr_alpha, dropout=args.dropout, num_layers=args.num_layers).to(device)
    elif args.method == 'gat':
        model = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                    dropout=args.dropout, heads=args.gat_heads).to(device)
    elif args.method == 'gsage':
        model = GraphSAGE(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, use_bn=not args.no_bn).to(device)
    elif args.method == 'gin':
        model = GIN(d, args.hidden_channels, c, num_layers=args.num_layers,
                    dropout=args.dropout, use_bn=not args.no_bn).to(device)
    elif args.method == 'lp':
        mult_bin = args.dataset=='ogbn-proteins'
        model = MultiLP(c, args.lp_alpha, args.hops, mult_bin=mult_bin)
    elif args.method == 'mixhop':
        model = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers,
                       dropout=args.dropout, hops=args.hops).to(device)
    elif args.method == 'gcnjk':
        model = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, jk_type=args.jk_type).to(device)
    elif args.method == 'gatjk':
        model = GATJK(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, heads=args.gat_heads,
                        jk_type=args.jk_type).to(device)
    elif args.method == 'h2gcn':
        model = H2GCN(d, args.hidden_channels, c, dataset.edge_index,
                        dataset.num_nodes,
                        num_layers=args.num_layers, dropout=args.dropout,
                        num_mlp_layers=args.num_mlp_layers).to(device)
    elif args.method == 'link_concat':
        model = LINK_Concat(d, args.hidden_channels, c, args.num_layers, dataset.num_nodes, dropout=args.dropout).to(device)
    elif args.method == 'linkx':
        model = LINKX(d, args.hidden_channels, c, args.num_layers, dataset.num_nodes,
        inner_activation=args.inner_activation, inner_dropout=args.inner_dropout, dropout=args.dropout, init_layers_A=args.link_init_layers_A, init_layers_X=args.link_init_layers_X).to(device)
    elif args.method == 'gcn2':
        model = GCNII(d, args.hidden_channels, c, args.num_layers, args.gcn2_alpha, args.theta, dropout=args.dropout).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--ood_type', type=str, default='label', choices=['structure', 'label', 'feature'])
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--method', '-m', type=str, default='gcn')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--display_step', type=int,
                        default=50, help='how often to print')
    parser.add_argument('--hops', type=int, default=2,
                        help='power of adjacency matrix for certain methods')
    parser.add_argument('--cs_fixed', action='store_true', help='use FDiff-scale')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='attention heads for gat')
    parser.add_argument('--lp_alpha', type=float, default=.1,
                        help='alpha for label prop')
    parser.add_argument('--gpr_alpha', type=float, default=.1,
                        help='alpha for gprgnn')
    parser.add_argument('--gcn2_alpha', type=float, default=.1,
                        help='alpha for gcn2')
    parser.add_argument('--theta', type=float, default=.5,
                        help='theta for gcn2')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--jk_type', type=str, default='max', choices=['max', 'lstm', 'cat'],
                        help='jumping knowledge type')
    parser.add_argument('--rocauc', action='store_true',
                        help='set the eval function to rocauc')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='number of mlp layers in h2gcn')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--adam', action='store_true', help='use adam instead of adamW')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--no_bn', action='store_true', help='do not use batchnorm')
    parser.add_argument('--sampling', action='store_true', help='use neighbor sampling')
    parser.add_argument('--inner_activation', action='store_true', help='Whether linkV3 uses inner activation')
    parser.add_argument('--inner_dropout', action='store_true', help='Whether linkV3 uses inner dropout')
    parser.add_argument("--SGD", action='store_true', help='Use SGD as optimizer')
    parser.add_argument('--link_init_layers_A', type=int, default=1)
    parser.add_argument('--link_init_layers_X', type=int, default=1)
    parser.add_argument('--ood', type=str, default='MSP')
    parser.add_argument('--neighbors', type=int, default=10, help='neighbors for KNN')
    parser.add_argument('--K', type=int, default=8, help='number of layers for belief propagation')
    parser.add_argument('--alpha', type=float, default=0., help='weight for residual connection in propagation')
    parser.add_argument('--T', type=float, default=1.0, help='temperature for Softmax')
    parser.add_argument('--noise', type=float, default=0., help='param for baseline ODIN and Mahalanobis')
    parser.add_argument('--tau1', type=float, default=5, help='threshold to determine s_id and s_ood')
    parser.add_argument('--tau2', type=float, default=50, help='threshold to select train nodes as G')
    parser.add_argument('--delta', type=float, default=1.02, help='weight for G')
    parser.add_argument('--prop', action='store_true', help='whether to use belief propagation')
    parser.add_argument('--prop_type', type=str, default='naive', choices=['naive', 'highorder', 'appnp', 'gdc', 'graphheat', 'gprgnn', 'mixhop'], help='what prop type to use')
    parser.add_argument('--rw', type=str, default='rw', help='use random walk or symmetric to prop')
    parser.add_argument('--grasp', action='store_true', help='whether to use GRASP')
    parser.add_argument('--test', action='store_true', help='whether to augmentate on test')
    parser.add_argument('--gaug', action='store_true', help='whether to use GAug')
    parser.add_argument('--add_pct', type=float, default=5, help='ratio of edges to add')
    parser.add_argument('--remove_pct', type=float, default=50, help='ratio of edges to remove')
    parser.add_argument('--st', type=str, default='top', choices=['top', 'low', 'random'], help='what metric to use')
    # GPN
    #parser.add_argument('--mode', type=str, default='detect', choices=['classify', 'detect'])
    parser.add_argument('--GPN_detect_type', type=str, default='Epist', choices=['Alea', 'Epist', 'Epist_wo_Net'])
    parser.add_argument('--GPN_warmup', type=int, default=5)
    # GKDE hyperparameter
    parser.add_argument('--gkde_seed', default=42, type=int)
    parser.add_argument('--gkde_dim_hidden', default=16, type=int)
    parser.add_argument('--gkde_dropout_prob', default=0.5, type=float)
    parser.add_argument('--gkde_use_kernel', default=1, type=int)
    parser.add_argument('--gkde_lambda_1', default=0.001, type=float)
    parser.add_argument('--gkde_teacher_training', default=1, type=int)
    parser.add_argument('--gkde_use_bayesian_dropout', default=0, type=int)
    parser.add_argument('--gkde_sample_method', default='log_evidence', type=str)
    parser.add_argument('--gkde_num_samples_dropout', default=10, type=int)
    parser.add_argument('--gkde_loss_reduction', default=None, type=str)
    parser.add_argument('--gkde_use_pretrained', action='store_false', help='use pretrained backborn to accelate training')
    # OODGAT
    parser.add_argument('--continuous', default=False, type=bool)
    parser.add_argument('--hidden_dim', default=32, type=int)
    parser.add_argument('--drop_prob', default=0.5, type=float)
    parser.add_argument('--drop_input', default=0.5, type=float)
    parser.add_argument('--drop_edge', default=0.6, type=float)
    parser.add_argument('--heads', default=4, type=int)
    parser.add_argument('--w_consistent', default=2.0, type=float)
    parser.add_argument('--w_ent', default=0.05, type=float)
    parser.add_argument('--w_discrepancy', default=5e-3, type=float)
    parser.add_argument('--margin', default=0.6, type=float)
    parser.add_argument('--OODGAT_detect_type', type=str, default='ATT', choices=['ATT', 'ENT'])
    parser.add_argument('--random_seed_data', default=123, type=int)
    parser.add_argument('--random_seed_model', default=456, type=int)