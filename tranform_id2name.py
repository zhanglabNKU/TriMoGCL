import numpy as np
import argparse

parser = argparse.ArgumentParser(description='TriMoGCL for transforming node id to name')
parser.add_argument('--data-name', default='drkg', help='The name of dataset.')
parser.add_argument('--task', default='binary', help='The task for triplet classification.')
parser.add_argument('--input_dir', type=str, default='../data/')

args = parser.parse_args()
args.input_dir = args.input_dir + args.data_name + '/'

if args.data_name == 'drkg':
    args.drug_num = 2908
    args.dise_num = 2157
    args.gene_num = 9809
if args.data_name == 'ms':
    args.drug_num = 1272
    args.dise_num = 694
    args.gene_num = 4519

if args.data_name == 'drkg':
    id2dise = np.load(args.input_dir + 'id2dise-feat-hierarchy.npy', allow_pickle=True).item()
    id2drug = np.load(args.input_dir + 'id2drug-feat-hierarchy.npy', allow_pickle=True).item()
    id2gene = np.load(args.input_dir + 'id2gene-feat-hierarchy.npy', allow_pickle=True).item()
if args.data_name == 'ms':
    id2dise = np.load(args.input_dir + 'id2dise.npy', allow_pickle=True).item()
    id2drug = np.load(args.input_dir + 'id2drug.npy', allow_pickle=True).item()
    id2gene = np.load(args.input_dir + 'id2gene.npy', allow_pickle=True).item()

entity_dict = np.load(args.input_dir + 'entity_dictionary.npy', allow_pickle=True).item()
entityid2name = {'Disease': {}, 'Gene': {}, 'Compound': {}}
entityid2name['Disease'] = dict(zip(entity_dict['Disease'].values(), entity_dict['Disease'].keys()))
entityid2name['Gene'] = dict(zip(entity_dict['Gene'].values(), entity_dict['Gene'].keys()))
entityid2name['Compound'] = dict(zip(entity_dict['Compound'].values(), entity_dict['Compound'].keys()))


def write_tri(tri):
    '''
    :param tri: a tensor of triples with shape (N, 3)
    '''
    with open(args.input_dir + f'{args.data_name}_id2name.txt', 'w') as f:
        for i in range(len(tri)):
            x, y, z = tri[i]
            x = x.item()
            y = y.item()
            z = z.item()
            if args.data_name == 'drkg':
                dise_name = entityid2name['Disease'][id2dise[x]].split('::')[1]
                drug_name = entityid2name['Compound'][id2drug[y - args.dise_num]].split('::')[1]
                gene_name = entityid2name['Gene'][id2gene[z - args.dise_num - args.drug_num]].split('::')[1]
            if args.data_name == 'ms':
                dise_name = entityid2name['Disease'][id2dise[x]]
                drug_name = entityid2name['Compound'][id2drug[y - args.dise_num]]
                gene_name = entityid2name['Gene'][id2gene[z - args.dise_num - args.drug_num]]
            f.write(f'{dise_name}, {drug_name}, {gene_name}\n')
        f.close()