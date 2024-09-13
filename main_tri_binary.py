import torch
import numpy as np
import argparse
import os.path
from utils import prepare_data
from base_gcn import GCN_binary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,\
precision_recall_curve, auc
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2none(v):
    if v.lower() == 'none':
        return None
    else:
        return str(v)


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser(description='Link Prediction with Walk-Pooling')
# Dataset
parser.add_argument('--data-name', default='ms', help='graph name')
parser.add_argument('--task', default='binary', help='graph name')

# training/validation/test divison and ratio
parser.add_argument('--input_dir', type=str, default='../data/')
parser.add_argument('--res_dir', type=str, default='24-7-4-binary all')

parser.add_argument('--dise_feat_dir', type=str, default='../data/drkg/dise_feats.pth')
parser.add_argument('--drug_feat_dir', type=str, default='../data/drkg/drug_feats.pth')
parser.add_argument('--gene_feat_dir', type=str, default='../data/drkg/gene_feats.pth')

parser.add_argument('--observe-val-and-injection', type=str2bool, default=False,
                    help='whether to contain the validation set in the observed graph and apply injection trick')

parser.add_argument('--test-ratio', type=float, default=0.1,
                    help='0.1 ratio of test links')
parser.add_argument('--val-ratio', type=float, default=0.1,
                    help='ratio of validation links. If using the splitted data from SEAL,\
                     it is the ratio on the observed links, othewise, it is the ratio on the whole links.')

# Model and Training
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='0.001:200, 00005:32, learning rate')
parser.add_argument('--weight-decay', type=float, default=0)
parser.add_argument('--hidden-channels', type=int, default=256)
parser.add_argument('--batch-size', type=int, default=5000)
parser.add_argument('--batch_num', type=int, default=10)
parser.add_argument('--epoch-num', type=int, default=150)
parser.add_argument('--tau', type=float, default=1000)
parser.add_argument('--lam', type=float, default=0.1)

parser.add_argument('--abla_edge', action="store_true", help="whether to remove edge pooling")
parser.add_argument('--abla_basic', action="store_true", help="whether to remove basic pooling")

args = parser.parse_args()
args.input_dir = args.input_dir + args.data_name + '/'
args.res_dir = '../results/' + args.data_name + ' ' + args.res_dir + '/'

if args.data_name == 'drkg':
    args.drug_num = 2908
    args.dise_num = 2157
    args.gene_num = 9809
if args.data_name == 'ms':
    args.drug_num = 1272
    args.dise_num = 694
    args.gene_num = 4519

print('<<Begin generating training data>>')
data, poslist, neglist = prepare_data(args)
data = data.to(device)

print('<<Complete generating training data>>')

lr = args.lr
weight_decay = args.weight_decay

torch.cuda.empty_cache()

torch.cuda.empty_cache()
set_random_seed(args.seed)

num_features = data.x.shape[1]
hidden_channels = args.hidden_channels

total_nodes = args.dise_num + args.drug_num + args.gene_num
model = GCN_binary(in_dim=num_features, h_dim=hidden_channels, out_dim=hidden_channels, number_nodes=total_nodes, args=args)


model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

criterion = torch.nn.BCEWithLogitsLoss()
crsoftmax = torch.nn.CrossEntropyLoss()

batch_size = args.batch_size
batch_num = args.batch_num
total_nodes = args.dise_num + args.drug_num + args.gene_num

def contrastive_loss(z1, z2, tau):
    score = torch.mm(z1, z2.t())
    z1_norm = torch.norm(z1, dim=1, keepdim=True)  # (B, 1)
    z2_norm = torch.norm(z2, dim=1, keepdim=True)  # (B, 1)
    score = score/(z1_norm @ z2_norm.t())/tau  # (B, 1) @ (1, B)
    score = torch.softmax(score, dim=1)
    return -torch.mean(torch.log(torch.diag(score)))


def train(infeat, edge_index, pos_train_edge, neg_train_edge, edge_attr=None):
    '''
    pos_train_edge: N * 3
    '''
    model.train()
    total_loss = 0
    total_examples = 0
    adjmask = torch.ones_like(edge_index[0], dtype=torch.bool).to(device)
    for _ in range(batch_num):
        optimizer.zero_grad()
        loss = 0
        adj_perm = torch.randperm(len(edge_index[0]))
        adjmask[adj_perm[:batch_size]] = 0
        h = model(infeat, edge_index)
        h_ = model(infeat, edge_index[:, adjmask])
        contras_loss = contrastive_loss(h, h_, args.tau)
        loss = args.lam * contras_loss

        start = len(pos_train_edge)//batch_num * _
        end = len(pos_train_edge)//batch_num * (_ + 1)
        edge = pos_train_edge[start:end]

        feat = model.pred(h, edge)
        feat_ = model.pred(h_, edge)
        if not args.abla_edge:
            input_feat = model.pooling2(h, edge)
            input_feat_ = model.pooling2(h_, edge)
            feat_2channle = torch.cat((feat, input_feat), dim=1)
            feat_2channle_ = torch.cat((feat_, input_feat_), dim=1)
        outs = []
        outs_ = []
        if not args.abla_edge:
            outs.append(model.classifier(feat_2channle))
            outs_.append(model.classifier(feat_2channle_))
        else:
            outs.append(model.classifier(feat))
            outs_.append(model.classifier(feat_))

        lab = []
        lab.append(torch.ones(outs[-1].shape[0]).long().to(device))

        edge = neg_train_edge[start:end]

        feat = model.pred(h, edge)
        feat_ = model.pred(h_, edge)
        if not args.abla_edge:
            input_feat = model.pooling2(h, edge)
            input_feat_ = model.pooling2(h_, edge)
            feat_2channle = torch.cat((feat, input_feat), dim=1)
            feat_2channle_ = torch.cat((feat_, input_feat_), dim=1)

            outs.append(model.classifier(feat_2channle))
            outs_.append(model.classifier(feat_2channle_))
        else:
            outs.append(model.classifier(feat))
            outs_.append(model.classifier(feat_))

        lab.append(torch.zeros(outs[-1].shape[0]).long().to(device))
        outs = torch.cat(outs, dim=0)
        outs_ = torch.cat(outs_, dim=0)
        lab = torch.cat(lab, dim=0)
        loss += crsoftmax(outs, lab)
        if args.lam > 0:
            loss += crsoftmax(outs_, lab)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() / 7
    return total_loss / batch_num



def ttest(infeat, edge_index, pos_valid_edge, neg_valid_edge, edge_attr=None, data_type='test'):
    model.eval()
    with torch.no_grad():
        h = model(infeat, edge_index)

        edge = pos_valid_edge
        feat = model.pred(h, edge)  # pooling 1
        if not args.abla_edge:
            input_feat = model.pooling2(h, edge)  # pooling 2
            feat = torch.cat((feat, input_feat), dim=1)

        out = model.classifier(feat)
        out = torch.softmax(out, dim=1)
        prb = [out[:, 1].cpu().detach()]
        prd = [out.argmax(dim=1).cpu().detach()]

        lab = [np.ones(out.shape[0])]

        edge = neg_valid_edge
        feat = model.pred(h, edge)  # pooling 1
        if not args.abla_edge:
            input_feat = model.pooling2(h, edge)  # pooling 2
            feat = torch.cat((feat, input_feat), dim=1)

        out = model.classifier(feat)
        out = torch.softmax(out, dim=1)
        prb.append(out[:, 1].cpu().detach())
        lab.append(np.zeros(out.shape[0]))
        prd.append(out.argmax(dim=1).cpu().detach())

        prb = torch.cat(prb, dim=0).numpy()
        prd = torch.cat(prd, dim=0).numpy()
        lab = np.concatenate(lab, axis=0)
        pre_data = precision_score(lab, prd)
        rec_data = recall_score(lab, prd)
        acc_data = accuracy_score(lab, prd)
        auc_data = roc_auc_score(lab, prb)
        precision, recall, _ = precision_recall_curve(lab, prb)
        apr_data = auc(recall, precision)

    res = [[pre_data, rec_data, acc_data, auc_data, apr_data]]
    return res


def write_results(args, res):
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
        write_form = 'w'
    else:
        write_form = 'a'
    with open(args.res_dir + 'AUC-data.txt', write_form) as f:
        for i in res['AUC'][:-1]:
            f.write(f'{i:.2f},')
        f.write(f'{res["AUC"][-1]:.2f}\n')
        f.close()
    with open(args.res_dir + 'AUPR-data.txt', write_form) as f:
        for i in res['AUPR'][:-1]:
            f.write(f'{i:.2f},')
        f.write(f'{res["AUPR"][-1]:.2f}\n')
        f.close()


def reset_adj(A, row, col):
    A[row, col] = 0.0
    A[col, row] = 0.0
    return A


te_res_list = []

for i, key in enumerate(poslist[0]):
    Best_Val_from_maf1 = 0
    Best_metrics = 0
    Final_Test_AUC_from_maf1 = 0
    Final_Test_AP_from_maf1 = 0
    Final_Test_epoch_from_maf1 = 0
    model = GCN_binary(in_dim=num_features, h_dim=hidden_channels, out_dim=hidden_channels, number_nodes=total_nodes,
                       args=args)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    edges_in_graph = data.train_graph
    adj = torch.zeros(total_nodes, total_nodes).to(device)
    adj[edges_in_graph[0], edges_in_graph[1]] = 1.0
    adj = reset_adj(adj, neglist[1][key][:, 0], neglist[1][key][:, 1])
    adj = reset_adj(adj, neglist[1][key][:, 0], neglist[1][key][:, 2])
    adj = reset_adj(adj, neglist[1][key][:, 1], neglist[1][key][:, 2])
    adj = reset_adj(adj, neglist[2][key][:, 0], neglist[2][key][:, 1])
    adj = reset_adj(adj, neglist[2][key][:, 0], neglist[2][key][:, 2])
    adj = reset_adj(adj, neglist[2][key][:, 1], neglist[2][key][:, 2])
    edges_in_graph = adj.nonzero().t()
    for epoch in range(0, args.epoch_num):
        loss_epoch = train(data.x, edges_in_graph, poslist[0][key], neglist[0][key], data.train_attr)
        va_res = ttest(data.x, edges_in_graph, poslist[1][key], neglist[1][key], data.train_attr, data_type='val')

        va_res_df = pd.DataFrame(va_res, columns=['precision', 'recall', 'ACC', 'AUC', 'AUPR'])
        # print('Valid:------')
        # print(va_res_df)

        if va_res_df['AUC'].item() > Best_Val_from_maf1:
            Best_Val_from_maf1 = va_res_df['AUC'].item()
            te_res = ttest(data.x, edges_in_graph, poslist[2][key], neglist[2][key], data.train_attr, data_type='test')
            te_res_df = pd.DataFrame(te_res, columns=['precision', 'recall', 'ACC', 'AUC', 'AUPR'])
            Best_metrics = te_res[0]
            Final_Test_epoch_from_maf1 = epoch
            # print('Test:------')
            # print(te_res_df)

    print(f'{key} Best Test Result: ', Best_metrics)
    te_res_list.append(Best_metrics)

Best_metrics = pd.DataFrame(te_res_list, columns=['precision', 'recall', 'ACC', 'AUC', 'AUPR'], index=list(poslist[0].keys()))
write_results(args, Best_metrics * 100)

print('ok')

