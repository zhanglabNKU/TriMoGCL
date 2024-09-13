import torch
import numpy as np
from torch_geometric.data import Data
from create_data import random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def floor(x):
    return torch.div(x, 1, rounding_mode='trunc')


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def collect_train_graph(level: str, cycles_list, train: str, tp):
    train_pos = [[], []]
    train_attr = []
    if level == 'cycles':
        train_pos[0].append(cycles_list[train][:, 0])
        train_pos[1].append(cycles_list[train][:, 1])
        train_pos[1].append(cycles_list[train][:, 2])
        train_pos[0].append(cycles_list[train][:, 0])
        train_pos[0].append(cycles_list[train][:, 1])
        train_pos[1].append(cycles_list[train][:, 2])
        train_attr.append(torch.zeros(len(cycles_list[train][:, 0]), dtype=torch.long))
        train_attr.append(torch.zeros(len(cycles_list[train][:, 0]), dtype=torch.long) + 1)
        train_attr.append(torch.zeros(len(cycles_list[train][:, 0]), dtype=torch.long) + 2)
    if level == 'tuples':
        train_pos[0].append(cycles_list[train][:, tp[1][0]])
        train_pos[1].append(cycles_list[train][:, tp[1][1]])
        train_pos[0].append(cycles_list[train][:, tp[1][2]])
        train_pos[1].append(cycles_list[train][:, tp[1][3]])
        train_attr.append(torch.zeros(len(cycles_list[train][:, 0]), dtype=torch.long) + tp[2][0])
        train_attr.append(torch.zeros(len(cycles_list[train][:, 0]), dtype=torch.long) + tp[2][1])
    if level == 'single':
        train_pos[0].append(cycles_list[train][:, tp[1][0]])
        train_pos[1].append(cycles_list[train][:, tp[1][1]])
        train_attr.append(torch.zeros(len(cycles_list[train][:, 0]), dtype=torch.long) + tp[2])
    train_pos[0] = np.concatenate(train_pos[0], axis=0)
    train_pos[1] = np.concatenate(train_pos[1], axis=0)
    train_pos = np.stack((train_pos[0], train_pos[1]), axis=0)
    train_attr = torch.cat(train_attr, dim=0)
    return train_pos, train_attr


def negative_sampling(pos_list, pos_idx, drug_dise_adj, gene_dise_adj, gene_drug_adj):
    neg_n = 0
    neg_cand = []
    if pos_idx == 0:
        for i in range(1, len(pos_list)):
            neg_n += len(pos_list[i])
            neg_cand.append(pos_list[i])
        # neg_idx = torch.randint(0, neg_n, (len(pos_list[0]), ))

    elif pos_idx == (len(pos_list) - 1):
        for i in range(0, len(pos_list) - 1):
            neg_n += len(pos_list[i])
            neg_cand.append(pos_list[i])
        # neg_idx = torch.randint(0, neg_n, (len(pos_list[pos_idx]), ))

    else:
        for i in range(0, pos_idx):
            neg_n += len(pos_list[i])
            neg_cand.append(pos_list[i])
        for i in range(pos_idx + 1, len(pos_list)):
            neg_n += len(pos_list[i])
            neg_cand.append(pos_list[i])
        # neg_idx = torch.randint(0, neg_n, (len(pos_list[pos_idx]), ))
    if len(pos_list[pos_idx]) <= neg_n:
        neg_idx = np.random.choice(neg_n, len(pos_list[pos_idx]), replace=False)
        # neg_idx = torch.from_numpy(neg_idx)
        # neg_idx = torch.unique(neg_idx)
        # neg_sam = torch.from_numpy(np.concatenate(neg_cand, axis=0))[neg_idx].long()
        neg_sam = np.concatenate(neg_cand, axis=0)[neg_idx]
    else:
        neg_sam = np.concatenate(neg_cand, axis=0)
        supp = []
        dise_num = len(drug_dise_adj[0])
        drug_num = len(drug_dise_adj)
        gene_num = len(gene_drug_adj)
        while len(supp) < (len(pos_list[pos_idx]) - neg_n):
            x = np.random.choice(dise_num)
            y = np.random.choice(drug_num)
            z = np.random.choice(gene_num)
            if pos_idx == 0:
                if drug_dise_adj[y, x] * gene_dise_adj[z, x] * gene_drug_adj[z, y] == 0:
                    supp.append((x, y, z))
                else:
                    continue
            if pos_idx == 1:
                if drug_dise_adj[y, x] * gene_dise_adj[z, x] == 0:
                    supp.append((x, y, z))
                elif drug_dise_adj[y, x] * gene_dise_adj[z, x] * gene_drug_adj[z, y] == 1:
                    supp.append((x, y, z))
                else:
                    continue
            if pos_idx == 2:
                if drug_dise_adj[y, x] * gene_drug_adj[z, y] == 0:
                    supp.append((x, y, z))
                elif drug_dise_adj[y, x] * gene_dise_adj[z, x] * gene_drug_adj[z, y] == 1:
                    supp.append((x, y, z))
                else:
                    continue
            if pos_idx == 3:
                if gene_dise_adj[z, x] * gene_drug_adj[z, y] == 0:
                    supp.append((x, y, z))
                elif drug_dise_adj[y, x] * gene_dise_adj[z, x] * gene_drug_adj[z, y] == 1:
                    supp.append((x, y, z))
                else:
                    continue
            if pos_idx == 4:
                if drug_dise_adj[y, x] == 0:
                    supp.append((x, y, z))
                elif drug_dise_adj[y, x] + gene_dise_adj[z, x] + gene_drug_adj[z, y] >= 2:
                    supp.append((x, y, z))
                else:
                    continue
            if pos_idx == 5:
                if gene_dise_adj[z, x] == 0:
                    supp.append((x, y, z))
                elif drug_dise_adj[y, x] + gene_dise_adj[z, x] + gene_drug_adj[z, y] >= 2:
                    supp.append((x, y, z))
                else:
                    continue
            if pos_idx == 6:
                if gene_drug_adj[z, y] == 0:
                    supp.append((x, y, z))
                elif drug_dise_adj[y, x] + gene_dise_adj[z, x] + gene_drug_adj[z, y] >= 2:
                    supp.append((x, y, z))
                else:
                    continue
        supp = np.array(supp)
        supp[:, 1] += dise_num
        supp[:, 2] += dise_num + drug_num
        neg_sam = np.concatenate((neg_sam, supp), axis=0)
    assert len(neg_sam) == len(pos_list[pos_idx])
    neg_sam = torch.from_numpy(neg_sam)
    return neg_sam


def get_binary_dataset(args, cycles, tuples, single):
    drug_num = args.drug_num
    dise_num = args.dise_num
    gene_num = args.gene_num

    train_pos = {}
    valid_pos = {}
    test_pos = {}
    train_neg = {}
    valid_neg = {}
    test_neg = {}
    neg_list = {}
    pos_list = [cycles, tuples[0], tuples[1], tuples[2], single[0], single[1], single[2]]
    name_list = ['clique', '2-star 0', '2-star 1', '2-star 2', 'single 0', 'single 1', 'single 2']
    drug_dise = np.load(args.input_dir + 'Compound-Disease-feat-hierarchy.npy')
    gene_dise = np.load(args.input_dir + 'Gene-Disease-feat-hierarchy.npy')
    gene_drug = np.load(args.input_dir + 'Gene-Compound-feat-hierarchy.npy')
    drug_dise_adj = np.zeros((drug_num, dise_num))
    gene_dise_adj = np.zeros((gene_num, dise_num))
    gene_drug_adj = np.zeros((gene_num, drug_num))

    drug_dise_adj[drug_dise[0], drug_dise[1]] = 1.0
    gene_dise_adj[gene_dise[0], gene_dise[1]] = 1.0
    gene_drug_adj[gene_drug[0], gene_drug[1]] = 1.0

    for i, name in zip(range(7), name_list):
        neg_list[name] = negative_sampling(pos_list, i, drug_dise_adj, gene_dise_adj, gene_drug_adj)

    cycles_list = split_trvate(args, cycles, cycles.shape[0])
    train_graph = []
    train_attr = []
    train_pos['clique'] = torch.from_numpy(cycles_list['train']).to(device).long()
    valid_pos['clique'] = torch.from_numpy(cycles_list['valid']).to(device).long()
    test_pos['clique'] = torch.from_numpy(cycles_list['test']).to(device).long()

    tuples_list = {}
    for i in range(3):
        tuples_list[i] = split_trvate(args, tuples[i], tuples[i].shape[0])
        train_pos[f'2-star {i}'] = torch.from_numpy(tuples_list[i]['train']).to(device).long()
        valid_pos[f'2-star {i}'] = torch.from_numpy(tuples_list[i]['valid']).to(device).long()
        test_pos[f'2-star {i}'] = torch.from_numpy(tuples_list[i]['test']).to(device).long()

    single_list = {}
    for i in range(3):
        single_list[i] = split_trvate(args, single[i], single[i].shape[0])
        train_pos[f'single {i}'] = torch.from_numpy(single_list[i]['train']).to(device).long()
        valid_pos[f'single {i}'] = torch.from_numpy(single_list[i]['valid']).to(device).long()
        test_pos[f'single {i}'] = torch.from_numpy(single_list[i]['test']).to(device).long()

    for name in name_list:
        tmp_list = split_trvate(args, neg_list[name], neg_list[name].shape[0])
        train_neg[name] = tmp_list['train'].to(device)
        valid_neg[name] = tmp_list['valid'].to(device)
        test_neg[name] = tmp_list['test'].to(device)

    egdes, edge_attr = collect_train_graph('cycles', cycles_list, 'train', ())
    train_graph.append(egdes)
    train_attr.append(edge_attr)

    for i in [(0, (0, 1, 0, 2), (0, 1)), (1, (0, 1, 1, 2), (0, 2)), (2, (0, 2, 1, 2), (1, 2))]:
        egdes, edge_attr = collect_train_graph('tuples', tuples_list[i[0]], 'train', i)
        train_graph.append(egdes)
        train_attr.append(edge_attr)

    for i in [(0, (0, 1), 0), (1, (0, 2), 1), (2, (1, 2), 2)]:
        # train_graph.append(single_list[i[0]]['train'])
        # train_attr.append(torch.zeros(len(train_graph[-1][0]), dtype=torch.long) + i)
        egdes, edge_attr = collect_train_graph('single', single_list[i[0]], 'train', i)
        train_graph.append(egdes)
        train_attr.append(edge_attr)

    train_graph = torch.from_numpy(np.concatenate(train_graph, axis=1)).long()
    train_attr = torch.cat(train_attr, dim=0).long()
    train_graph = torch.cat((train_graph, train_graph[[1, 0]]), dim=1)
    train_attr = torch.cat((train_attr, train_attr), dim=0)

    data = Data(edge_index=train_graph, edge_attr=train_attr)
    data.num_nodes = args.dise_num + args.drug_num + args.gene_num
    dise_feat = torch.load(args.input_dir + 'dise_feat.pth')
    drug_feat = torch.load(args.input_dir + 'drug_feat.pth')
    gene_feat = torch.load(args.input_dir + 'gene_feat.pth')
    dise_feat = dise_feat.float()
    # dise_feat = F.normalize(dise_feat, dim=1)

    dise_feat_dim = dise_feat.shape[1]
    drug_feat_dim = drug_feat.shape[1]
    gene_feat_dim = gene_feat.shape[1]
    if args.data_name == 'drkg':
        drug_feat = torch.cat((drug_feat, torch.zeros(args.drug_num, dise_feat_dim - drug_feat_dim)), dim=1)
        gene_feat = torch.cat((gene_feat, torch.zeros(args.gene_num, dise_feat_dim - gene_feat_dim)), dim=1)
    if args.data_name == 'ms':
        dise_feat = torch.cat((dise_feat, torch.zeros(args.dise_num, drug_feat_dim - dise_feat_dim)), dim=1)
        gene_feat = torch.cat((gene_feat, torch.zeros(args.gene_num, drug_feat_dim - gene_feat_dim)), dim=1)
    data.x = torch.cat((dise_feat, drug_feat, gene_feat), dim=0)
    data.train_graph = train_graph
    data.train_attr = train_attr

    return data, (train_pos, valid_pos, test_pos), (train_neg, valid_neg, test_neg)


def split_trvate(args, row, n):
    perm = torch.randperm(n)
    n_v = floor(args.val_ratio * n).int()  # number of validation positive edges
    n_t = floor(args.test_ratio * n).int()  # number of test positive edges
    row = row[perm]
    row_list = {}
    row_list['valid'] = row[:n_v]
    row_list['test'] = row[n_v:n_v+n_t]
    row_list['train'] = row[n_v+n_t:]
    return row_list


def split_motif_mc(args, cycles, tuples, single):
    '''
        cycles              8:1:1
        tuples              8:1:1
            cenof dise      8:1:1
            cenof drug      8:1:1
            cenof gene      8:1:1
        single              8:1:1
            drug dise       8:1:1
            gene dise       8:1:1
            gene drug       8:1:1
    '''
    num_nodes = args.dise_num + args.drug_num + args.gene_num
    train_pos = []
    valid_pos = []
    test_pos = []
    train_lab = []
    valid_lab = []
    test_lab = []
    train_graph = []
    train_attr = []
    # valid_attr = []
    # test_attr = []
    cycles_list = split_trvate(args, cycles, cycles.shape[0])
    train_pos.append(cycles_list['train'])
    train_lab += len(cycles_list['train']) * [0]
    valid_pos.append(cycles_list['valid'])
    valid_lab += len(cycles_list['valid']) * [0]
    test_pos.append(cycles_list['test'])
    test_lab += len(cycles_list['test']) * [0]

    tuples_list = {}
    for i in range(3):
        tuples_list[i] = split_trvate(args, tuples[i], tuples[i].shape[0])
        train_pos.append(tuples_list[i]['train'])
        train_lab += len(tuples_list[i]['train']) * [i + 1]
        valid_pos.append(tuples_list[i]['valid'])
        valid_lab += len(tuples_list[i]['valid']) * [i + 1]
        test_pos.append(tuples_list[i]['test'])
        test_lab += len(tuples_list[i]['test']) * [i + 1]

    single_list = {}
    for i in range(3):
        single_list[i] = split_trvate(args, single[i], single[i].shape[0])
        train_pos.append(single_list[i]['train'])
        train_lab += len(single_list[i]['train']) * [i + 4]
        valid_pos.append(single_list[i]['valid'])
        valid_lab += len(single_list[i]['valid']) * [i + 4]
        test_pos.append(single_list[i]['test'])
        test_lab += len(single_list[i]['test']) * [i + 4]

    egdes, edge_attr = collect_train_graph('cycles', cycles_list, 'train', ())
    train_graph.append(egdes)
    train_attr.append(edge_attr)

    for i in [(0, (0, 1, 2, 0), (0, 1)), (1, (0, 1, 1, 2), (0, 2)), (2, (2, 0, 1, 2), (1, 2))]:
        egdes, edge_attr = collect_train_graph('tuples', tuples_list[i[0]], 'train', i)
        train_graph.append(egdes)
        train_attr.append(edge_attr)

    for i in [(0, (0, 1), 0), (1, (2, 0), 1), (2, (1, 2), 2)]:
        egdes, edge_attr = collect_train_graph('single', single_list[i[0]], 'train', i)
        train_graph.append(egdes)
        train_attr.append(edge_attr)

    train_graph = torch.from_numpy(np.concatenate(train_graph, axis=1)).long()
    # train_graph = train_graph[[1,0]]
    train_attr = torch.cat(train_attr, dim=0).long()
    di_train_graph = train_graph
    train_graph = torch.cat((train_graph, train_graph[[1, 0]]), dim=1)
    train_attr = torch.cat((train_attr, train_attr), dim=0)

    data = Data(edge_index=train_graph, edge_attr=train_attr)
    data.num_nodes = args.dise_num + args.drug_num + args.gene_num
    dise_feat = torch.load(args.input_dir + 'dise_feat.pth')
    drug_feat = torch.load(args.input_dir + 'drug_feat.pth')
    gene_feat = torch.load(args.input_dir + 'gene_feat.pth')
    dise_feat = dise_feat.float()

    dise_feat_dim = dise_feat.shape[1]
    drug_feat_dim = drug_feat.shape[1]
    gene_feat_dim = gene_feat.shape[1]
    if args.data_name == 'drkg':
        drug_feat = torch.cat((drug_feat, torch.zeros(args.drug_num, dise_feat_dim - drug_feat_dim)), dim=1)
        gene_feat = torch.cat((gene_feat, torch.zeros(args.gene_num, dise_feat_dim - gene_feat_dim)), dim=1)
    if args.data_name == 'ms':
        dise_feat = torch.cat((dise_feat, torch.zeros(args.dise_num, drug_feat_dim - dise_feat_dim)), dim=1)
        gene_feat = torch.cat((gene_feat, torch.zeros(args.gene_num, drug_feat_dim - gene_feat_dim)), dim=1)
    data.x = torch.cat((dise_feat, drug_feat, gene_feat), dim=0)
    # data.x = torch.rand(data.num_nodes, 512)
    data.train_graph = train_graph
    data.edge_index = train_graph
    data.di_train_graph = di_train_graph
    data.train_attr = train_attr

    data.train_pos = torch.from_numpy(np.concatenate(train_pos, axis=0)).long()
    data.val_pos = torch.from_numpy(np.concatenate(valid_pos, axis=0)).long()
    data.test_pos = torch.from_numpy(np.concatenate(test_pos, axis=0)).long()
    data.train_lab = torch.tensor(train_lab).long()
    data.valid_lab = torch.tensor(valid_lab).long()
    data.test_lab = torch.tensor(test_lab).long()
    print(di_train_graph.shape)
    print(train_graph.shape)
    return data


def prepare_data(args):
    '''
    1. global graph
    2. train, valid, test

    '''
    # load data from .mat or download from Planetoid dataset.
    # train_graphs = torch.load('../data/drkg/train_tri_hie.pth')
    # val_graphs = torch.load('../data/drkg/val_tri_hie.pth')
    # test_graphs = torch.load('../data/drkg/test_tri_hie.pth')
    set_random_seed(args.seed)
    cliques, drug_tuples, dise_tuples, gene_tuples, single_drdi, single_gedi, single_gedr = random_split(args)
    tuples = [dise_tuples, drug_tuples, gene_tuples]  # (3, N, 3)
    single = [single_drdi, single_gedi, single_gedr]  # (3, N, 3)

    if args.task == 'binary':
        data, poslist, neglist = get_binary_dataset(args, cliques, tuples, single)
        return data, poslist, neglist

    if args.task == 'multi-class':
        return split_motif_mc(args, cliques, tuples, single)
