import torch
import numpy as np
import argparse
import os.path
from utils import prepare_data
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score,\
    accuracy_score, auc
from base_gcn import GCN
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize


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
parser.add_argument('--data-name', default='drkg', help='graph name')
parser.add_argument('--task', default='multi-class', help='graph name')
parser.add_argument('--graph_formula', default='graph', help='graph name')


# training/validation/test divison and ratio
parser.add_argument('--input_dir', type=str, default='../data/')
parser.add_argument('--res_dir', type=str, default='analysis')
# parser.add_argument('--drug_dise_dir', type=str, default='../data/drkg/Compound-Disease-insert-feat.npy')
# parser.add_argument('--gene_dise_dir', type=str, default='../data/drkg/Gene-Disease-insert-feat.npy')
# parser.add_argument('--gene_drug_dir', type=str, default='../data/drkg/Gene-Compound-insert-feat.npy')
#
# parser.add_argument('--dise_feat_dir', type=str, default='../data/drkg/dise_feats.pth')
# parser.add_argument('--drug_feat_dir', type=str, default='../data/drkg/drug_feats.pth')
# parser.add_argument('--gene_feat_dir', type=str, default='../data/drkg/gene_feats.pth')
#
# parser.add_argument('--cycles', type=str, default='../data/drkg/cycles.npy')
# parser.add_argument('--tuples_cenofdise', type=str, default='../data/drkg/tuples_cenofdise.npy')
# parser.add_argument('--tuples_cenofdrug', type=str, default='../data/drkg/tuples_cenofdrug.npy')
# parser.add_argument('--tuples_cenofgene', type=str, default='../data/drkg/tuples_cenofgene.npy')
# parser.add_argument('--single_drdi', type=str, default='../data/drkg/single_drug_dise.npy')
# parser.add_argument('--single_gedi', type=str, default='../data/drkg/single_gene_dise.npy')
# parser.add_argument('--single_gedr', type=str, default='../data/drkg/single_gene_drug.npy')

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
parser.add_argument('--lr', type=float, default=0.0005,
                    help='0.001:200, 00005:32, learning rate')
parser.add_argument('--weight-decay', type=float, default=0)
# parser.add_argument('--walk-len', type=int, default=7, help='cutoff in the length of walks')
# parser.add_argument('--heads', type=int, default=2,
#                     help='using multi-heads in the attention link weight encoder ')
parser.add_argument('--hidden-channels', type=int, default=256)
parser.add_argument('--batch-size', type=int, default=5000)
parser.add_argument('--batch_num', type=int, default=10)
parser.add_argument('--epoch-num', type=int, default=150)
parser.add_argument('--tau', type=float, default=1000)
parser.add_argument('--lam1', type=float, default=0.1)
parser.add_argument('--lam2', type=float, default=0.1)
# parser.add_argument('--hitk', type=int, default=50)
parser.add_argument('--log', type=str, default=None,
                    help='log by tensorboard, default is None')
parser.add_argument('--gnn_conv', type=str, default='sign',
                    help='The convolution method in GNN')
parser.add_argument('--pooling', type=str, default='concat',
                    help='The pooling method to integrate three node features.')


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
# data = prepare_data(args)
data = prepare_data(args)
data = data.to(device)

print('<<Complete generating training data>>')

lr = args.lr
weight_decay = args.weight_decay

torch.cuda.empty_cache()
# print("Dimention of features after concatenation:", num_features)
set_random_seed(args.seed)

num_features = data.x.shape[1]
hidden_channels = args.hidden_channels
total_nodes = args.dise_num + args.drug_num + args.gene_num
model = GCN(in_dim=num_features, h_dim=hidden_channels, out_dim=hidden_channels, number_nodes=total_nodes)
# model = HygLinkPred(in_dim=num_features, h_dim=hidden_channels, out_dim=hidden_channels, number_nodes=total_nodes,
#                     dise_node=args.dise_num, drug_node=args.drug_num,
#                     gene_node=args.gene_num, conv_name='sign', hygs=, adj_2hop=None, adj_3hop=None)
# model = MLP(in_dim=num_features, h_dim=hidden_channels, out_dim=hidden_channels)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

crsoftmax = torch.nn.CrossEntropyLoss()

batch_size = args.batch_size


def contrastive_loss(z1, z2, tau):
    score = torch.mm(z1, z2.t())
    z1_norm = torch.norm(z1, dim=1, keepdim=True)  # (B, 1)
    z2_norm = torch.norm(z2, dim=1, keepdim=True)  # (B, 1)
    score = score/(z1_norm @ z2_norm.t())/tau  # (B, 1) @ (1, B)
    score = torch.softmax(score, dim=1)
    return -torch.mean(torch.log(torch.diag(score)))


def lap_loss(edge, label, c, node_num, feat):
    if (label == c).sum() > 0:
        triangle = edge[label == c]
        triangle_hyg = torch.zeros(triangle.shape[0], node_num).to(device)
        triangle_hyg[:, triangle[:, 0]] = 1.0
        triangle_hyg[:, triangle[:, 1]] = 1.0
        triangle_hyg[:, triangle[:, 2]] = 1.0
        tri2tri = triangle_hyg @ triangle_hyg.t()
        tri2tri = tri2tri.bool().float()
        deg = tri2tri.sum(dim=-1)
        deg = torch.pow(deg, -0.5)
        deg[torch.isinf(deg)] = 0.0
        tri2tri = torch.eye(tri2tri.shape[0]).to(device) - deg.unsqueeze(1) * tri2tri * deg.unsqueeze(0)
        tmp = feat[label == c].t() @ tri2tri
        la_loss = tmp @ feat[label == c]
        la_loss = torch.diag(la_loss).sum()
        return la_loss
    else:
        return 0


def protype_loss(feat, feat_, label, tau):
    prot = []
    prot_ = []
    for i in range(7):
        prot.append(feat[label == i].mean(dim=0))
        prot_.append(feat_[label == i].mean(dim=0))

    prot = torch.stack(prot, dim=0)
    prot_ = torch.stack(prot_, dim=0)
    norm = prot.norm(2, dim=1, keepdim=True) * prot_.norm(2, dim=1, keepdim=True).t()
    score = prot @ prot_.t() / norm/tau  # (B, 1) @ (1, B)
    score = torch.softmax(score, dim=1)
    return -torch.mean(torch.log(torch.diag(score)))


def train(infeat, edge_index, train_label, pos_train_edge, edge_attr=None):
    '''
    pos_train_edge: N * 3
    '''
    model.train()
    total_loss = 0
    total_examples = 0

    adjmask = torch.ones_like(edge_index[0], dtype=torch.bool).to(device)

    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        adj_perm = torch.randperm(len(edge_index[0]))
        adjmask[adj_perm[:batch_size]] = 0
        h = model(infeat, edge_index)
        h_ = model(infeat, edge_index[:, adjmask])
        edge = pos_train_edge[perm]
        label = train_label[perm]

        feat = model.pred(h, edge)  # pooling 1
        feat_ = model.pred(h_, edge)
        input_feat = model.pooling2(h, edge)  # pooling 2
        input_feat_ = model.pooling2(h_, edge)
        feat_2channle = torch.cat((feat, input_feat), dim=1)
        feat_2channle_ = torch.cat((feat_, input_feat_), dim=1)

        out = model.classifier(feat_2channle)  # feat: wo edge, feat_2channle: all
        out_ = model.classifier(feat_2channle_)

        loss = crsoftmax(out, label)
        loss += crsoftmax(out_, label)
        contras_loss = contrastive_loss(h, h_, args.tau)
        loss += args.lam1 * contras_loss

        loss += protype_loss(feat_2channle, feat_2channle_, label, 0.01) * args.lam2
        loss.backward()
        optimizer.step()

        num_examples = feat.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    return total_loss / total_examples


def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        # return auc(recall, precision, reorder=True)
        return auc(recall, precision)

    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)


def test(infeat, edge_index, valid_edge, valid_label):
    model.eval()
    pred_list = []
    label_list = []
    score_list = []
    with torch.no_grad():
        h = model(infeat, edge_index)
        for perm in DataLoader(range(valid_edge.size(0)), batch_size):
            edge = valid_edge[perm]
            label = valid_label[perm]

            feat = model.pred(h, edge)  # pooling 1
            input_feat = model.pooling2(h, edge)  # pooling 2
            feat = torch.cat((feat, input_feat), dim=1)
            out = model.classifier(feat)
            pred = out.argmax(dim=1).reshape(-1)
            pred_list.append(pred.cpu().clone().detach())
            label_list.append(label)
            out = torch.softmax(out, dim=-1)
            score_list.append(out.cpu().clone().detach())
        pred_list = torch.cat(pred_list, dim=0).cpu().numpy()
        label_list = torch.cat(label_list, dim=0).cpu().numpy()
        score_list = torch.cat(score_list, dim=0).cpu().numpy()
    y_one_hot = label_binarize(label_list, classes=np.arange(7))
    results = []
    # results.append(precision_score(label_list, pred_list, labels=list(range(7)), average='micro'))
    # results.append(recall_score(label_list, pred_list, labels=list(range(7)), average='micro'))
    results.append(f1_score(label_list, pred_list, labels=list(range(7)), average='micro') * 100)
    results.append(roc_auc_score(y_one_hot, score_list, average='micro') * 100)
    results.append(roc_aupr_score(y_one_hot, score_list, average='micro') * 100)

    results.append(roc_auc_score(y_one_hot, score_list, average='macro') * 100)
    results.append(roc_aupr_score(y_one_hot, score_list, average='macro') * 100)
    results.append(precision_score(label_list, pred_list, labels=list(range(7)), average='macro') * 100)
    results.append(recall_score(label_list, pred_list, labels=list(range(7)), average='macro') * 100)
    results.append(f1_score(label_list, pred_list, labels=list(range(7)), average='macro') * 100)
    results.append(accuracy_score(label_list, pred_list) * 100)

    results0 = []
    label_list[(0 < label_list) & (label_list < 4)] = 1
    pred_list[(0 < pred_list) & (pred_list < 4)] = 1
    label_list[3 < label_list] = 2
    pred_list[3 < pred_list] = 2
    # results0.append(precision_score(label_list, pred_list, labels=list(range(3)), average='micro'))
    # results0.append(recall_score(label_list, pred_list, labels=list(range(3)), average='micro'))
    results0.append(f1_score(label_list, pred_list, labels=list(range(3)), average='micro') * 100)

    results0.append(precision_score(label_list, pred_list, labels=list(range(3)), average='macro') * 100)
    results0.append(recall_score(label_list, pred_list, labels=list(range(3)), average='macro') * 100)
    results0.append(f1_score(label_list, pred_list, labels=list(range(3)), average='macro') * 100)
    results0.append(accuracy_score(label_list, pred_list) * 100)
    return results, results0


def print_results(ep, met):
    if len(met) > 6:
        print(f'Epoch: {ep:03d},\
              Micro F1: {met[0]:.2f}, AUC: {met[1]:.2f}, AUPR: {met[2]:.2f}\
              Macro AUC: {met[3]:.2f}, AUPR: {met[4]:.2f}, \
              Precision: {met[5]:.4f}, Recall: {met[6]:.4f}, F1: {met[7]:.4f}, ACC: {met[8]:.2f}')
    else:
        print(f'Epoch: {ep:03d},\
              Micro F1: {met[0]:.2f}, \
              Macro \
              Precision: {met[1]:.4f}, Recall: {met[2]:.4f}, F1: {met[3]:.4f}, ACC: {met[4]:.2f}')


def write_results(args, ep, met1, met2):
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
        write_form = 'w'
    else:
        write_form = 'a'
    with open(args.res_dir + '7-class-results.txt', write_form) as f:
        f.write(f'Epoch: {ep:03d},\
              Micro F1: {met1[0]:.2f}, AUC: {met1[1]:.2f}, AUPR: {met1[2]:.2f}\
              Macro AUC: {met1[3]:.2f}, AUPR: {met1[4]:.2f}, \
              Precision: {met1[5]:.4f}, Recall: {met1[6]:.4f}, F1: {met1[7]:.4f}, ACC: {met1[8]:.2f}\n')
        f.close()

    with open(args.res_dir + '7-class-results-data.txt', write_form) as f:
        f.write(f'{met1[0]:.2f},{met1[1]:.2f},{met1[2]:.2f},{met1[3]:.2f},{met1[4]:.2f},{met1[5]:.2f}\
        ,{met1[6]:.2f},{met1[7]:.2f},{met1[8]:.2f}\n')
        f.close()

    with open(args.res_dir + '3-class-results.txt', write_form) as f:
        f.write(f'Epoch: {ep:03d},\
              Micro F1: {met2[0]:.2f}, \
              Macro \
              Precision: {met2[1]:.4f}, Recall: {met2[2]:.4f}, F1: {met2[3]:.4f}, ACC: {met2[4]:.2f}\n')
        f.close()

    with open(args.res_dir + '3-class-results-data.txt', write_form) as f:
        f.write(f'{met2[0]:.2f},{met2[1]:.2f},{met2[2]:.2f},{met2[3]:.2f},{met2[4]:.2f}\n')
        f.close()


# Best_Val_frommif1 = 1e10
# Final_Test_AUC_fromloss = 0
# Final_Test_AP_fromloss = 0

Best_Val_from_maf1 = 0
Best_metrics = 0
Final_Test_AUC_from_maf1 = 0
Final_Test_AP_from_maf1 = 0
Final_Test_epoch_from_maf1 = 0
basic_0s, edge_0s, basic_3s, edge_3s, basic_5s, edge_5s = [], [], [], [], [], []
train_loss, test_loss = [],[]
test_f1 =[]
try:
    for epoch in range(0, args.epoch_num):

        loss_epoch = train(data.x, data.train_graph, data.train_lab, data.train_pos, data.train_attr)

        va_metrics, va_metrics0 = test(data.x, data.train_graph, data.val_pos, data.valid_lab)

        print('Valid:------')
        print_results(epoch, va_metrics)
        print_results(epoch, va_metrics0)

        if (va_metrics[3] + va_metrics0[4])/2 > Best_Val_from_maf1:
            Best_Val_from_maf1 = (va_metrics[3] + va_metrics0[4])/2
            te_metrics, te_metrics0 = test(data.x, data.train_graph,
                                                           data.test_pos, data.test_lab)
            Best_metrics = (te_metrics, te_metrics0)
            Final_Test_epoch_from_maf1 = epoch
            print('Test:------')
            print_results(epoch, Best_metrics[0])
            print_results(epoch, Best_metrics[1])
            optm_model = model.state_dict()
except:
    print('NAN by prot')

print_results(Final_Test_epoch_from_maf1, Best_metrics[0])
print_results(Final_Test_epoch_from_maf1, Best_metrics[1])

write_results(args, Final_Test_epoch_from_maf1,
              Best_metrics[0], Best_metrics[1])

print('ok')
