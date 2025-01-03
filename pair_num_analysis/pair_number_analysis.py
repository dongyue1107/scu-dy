import pandas as pd
import os.path as osp
import torch
# from ogb.utils.mol import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import download_url, extract_zip
from rdkit import RDLogger
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder
import shutil
from pathlib import Path
RDLogger.DisableLog('rdApp.*')
from ogb.utils.features import (bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict)
from rdkit import Chem
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from ogb.utils.features import get_bond_feature_dims
from torch import nn, optim
import os
import argparse
from typing import Optional, Union, Dict


allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'misc'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1
# # miscellaneous case
# i = safe_index(allowable_features['possible_atomic_num_list'], 'asdf')
# assert allowable_features['possible_atomic_num_list'][i] == 'misc'
# # normal case
# i = safe_index(allowable_features['possible_atomic_num_list'], 2)
# assert allowable_features['possible_atomic_num_list'][i] == 2

def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
    return atom_feature

def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
        ]))

def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string, sanitize=False)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph

def parse_args():

    parser = argparse.ArgumentParser(description='Graph data miming with GNN')
    parser.add_argument('--task_name', type=str, default='GINGraphPooling',
                        help='task name')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any(default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--emb_dim', type=int, default=256,
                        help='dimensionality of hidden units in GNNs (default: 256)')
    parser.add_argument('--drop_ratio', type=float, default=0.,
                        help='dropout ratio (default: 0.)')
    parser.add_argument('--save_test', action='store_true')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='early stop (default: 10)')
    parser.add_argument('--num_workers', type=int, default=0,  # 4,
                        help='number of workers (default: 4)')
    # parser.add_argument('--dataset_root', type=str, default="dataset",
    #                     help='dataset root')
    parser.add_argument('--filepath', type=str, default='/home/dy1/uspto_structure/csd_number.csv', help='path to the input file')

    args = parser.parse_args()

    return  args

import sys
sys.argv=['']
del sys

class MyDataset(Dataset):

    def __init__(self, filepath):
        # super(MyDataset, self).__init__()
        self.filepath = Path(filepath)
        data_df = pd.read_csv(self.filepath)
        self.smiles_list = data_df['SMILES']
        self.pair_number = data_df['N_Number']

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles, pair_number = self.smiles_list[idx], self.pair_number[idx]
        graph = smiles2graph(smiles)

        if graph is None or graph['node_feat'] is None or graph['edge_index'] is None or graph['edge_feat'] is None:
            print(f"Skipping entry at index {idx} due to None graph.")
            return None

        assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert(len(graph['node_feat']) == graph['num_nodes'])

        x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        y = torch.Tensor([pair_number])
        num_nodes = int(graph['num_nodes'])
        data = Data(x, edge_index, edge_attr, y, num_nodes=num_nodes)

        return data

    def get_idx_split(self):
        try:
            split_dict = replace_numpy_with_torchtensor(torch.load(self.filepath, map_location=torch.device('cpu')))
            return split_dict
        except Exception as e:
            print(f"Error loading split dictionary from {self.filepath}: {e}")
            return None


full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding

class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding

class GINConv(MessagePassing):
    def __init__(self, emb_dim):

        super(GINConv, self).__init__(aggr = 'add')
        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class GINNodeEmbedding(torch.nn.Module):
    """
    Output:
        node representations
        """
    def __init__(self, num_layers, emb_dim, drop_ratio=0.5, JK='last', residual=False):
        """GIN node embedding module"""

        super(GINNodeEmbedding, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        # add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder =AtomEncoder(emb_dim)

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layers):
            self.convs.append(GINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr
        # computing input node embedding
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):
            h = self.conv[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        if self.JK == 'last':
            node_representation = h_list[-1]
        elif self.JK == 'sum':
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation += h_list[layer]

        return node_representation

class GINGraphPooling(nn.Module):

    def __init__(self, num_tasks=1, num_layers=5, emb_dim=300, residual=False, drop_ratio=0, JK='last', graph_pooling='sum'):
        """Gin Graph Pooling Module

        Out:
            graph representation
        """
        super(GINGraphPooling, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layers < 2:
            raise ValueError('Number of GNN layers must be greater than 1')

        self.gnn_node = GINNodeEmbedding(num_layers, emb_dim, JK=JK, dro_ratio=drop_ratio, residual=residual)

        if graph_pooling == 'sum':
            self.pool = global_add_pool
        elif graph_pooling == 'mean':
            self.pool = global_mean_pool
        elif graph_pooling == 'max':
            self.pool = global_max_pool
        elif graph_pooling == 'attention':
            self.pool = GlobalAttention(gate_nn=nn.Sequential(
                    nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1)))

        elif graph_pooling == 'set2set':
            self.pool = Set2Set(emb_dim, processing_steps=2)

        else:
            raise ValueError('Invalid graph pooling type.')

        if graph_pooling == 'set2set':
            self.graph_pred_liner = nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_liner = nn.Linear(self.emb_dim, self.num_tasks)


    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        output = self.graph_pred_liner(h_graph)

        if self.training:
            return output
        else:
            return torch.clamp(output, min=1, max=20)

class MyEvaluator:
    def __init__(self):
        '''
            Evaluator for the Mydataset
            Metric is Mean Absolute Error
        '''
        pass

    def eval(self, input_dict):
        '''
            y_true: numpy.ndarray or torch.Tensor of shape (num_graphs,)
            y_pred: numpy.ndarray or torch.Tensor of shape (num_graphs,)
            y_true and y_pred need to be of the same type (either numpy.ndarray or torch.Tensor)
        '''
        assert('y_pred' in input_dict)
        assert('y_true' in input_dict)

        y_pred, y_true = input_dict['y_pred'], input_dict['y_true']

        assert((isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray))
                or
                (isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor)))
        assert(y_true.shape == y_pred.shape)
        assert(len(y_true.shape) == 1)

        if isinstance(y_true, torch.Tensor):
            return {'mae': torch.mean(torch.abs(y_pred - y_true)).cpu().item()}
        else:
            return {'mae': float(np.mean(np.absolute(y_pred - y_true)))}

    def save_test_submission(self, input_dict: Dict, dir_path: str):
        '''
            save test submission file at dir_path
        '''
        assert('y_pred' in input_dict)

        y_pred = input_dict['y_pred']

        filename = osp.join(dir_path, 'y_pred_csd')

        assert(isinstance(filename, str))
        assert(isinstance(y_pred, np.ndarray) or isinstance(y_pred, torch.Tensor))

        if not osp.exists(dir_path):
            os.makedirs(dir_path)

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        y_pred = y_pred.astype(np.float32)
        np.savez_compressed(filename, y_pred = y_pred)



def train(model, device, loader, optimizer, criterion_fn):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        pred = model(batch).view(-1,)
        optimizer.zero_grad()
        loss = criterion_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)
            pred = model(batch).view(-1,)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    return evaluator.eval(input_dict)['mae']

def test(model, device, loader):
    model.eval()
    y_pred = []

    with torch.no_grad():
        for _, batch in enumerate(loader):
            batch = batch.to(device)
            pred = model(batch).view(-1,)
            y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)
    return y_pred


def prepartion(args):
    save_dir = os.path.join('saves', args.task_name)
    if os.path.exists(save_dir):
        for idx in range(1000):
            if not os.path.exists(save_dir + '=' + str(idx)):
                save_dir = save_dir + '=' + str(idx)
                break

    args.save_dir = save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    args_device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    args.output_file = open(os.path.join(args.save_dir, 'output'), 'a')
    print(args, file=args.output_file, flush=True)




def main(args):
    prepartion(args)
    nn_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }
    dataset = MyDataset(filepath=args.filepath)
    split_idx = dataset.get_idx_split()
    train_data = dataset[split_idx['train']]
    valid_data = dataset[split_idx['valid']]
    test_data = dataset[split_idx['test']]
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_works=args.num_works)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_works=args.num_works)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_works=args.num_works)
    evaluator = MyEvaluator()
    criterion_fn = torch.nn.MSELoss()
    device = args.device
    model = GINGraphPooling(**nn_params).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}', file=args.output_file, flush=True)
    print(model, file=args.output_file, flush=True)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

    writer = SummaryWriter(log_dir=args.save_dir)
    not_improved = 0
    best_valid_mae = 9999

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch), file=args.output_file, flush=True)
        print('Training...', file=args.output_file, flush=True)
        train_mae = train(model, device, train_loader, optimizer, criterion_fn)

        print('Evaluating...', file=args.output_file, flush=True)
        valid_mae = eval(model, device, valid_loader, evaluator)

        print({'Train': train_mae, 'Validation': valid_mae}, file=args.output_file, flush=True)

        writer.add_scalar('valid/mae', valid_mae, epoch)
        writer.add_scalar('train/mae', train_mae, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if args.save_test:
                print('Saving checkpoint...', file=args.output_file, flush=True)
                checkpoint = {
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae,
                    'num_params': num_params
                }
                torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pt'))
                print('Predicting on test data...', file=args.output_file, flush=True)
                y_pred = test(model, device, test_loader)
                print('Saving test submission file...', file=args.output_file, flush=True)
                evaluator.save_test_submission({'y_pred': y_pred}, args.save_dir)

            not_improved = 0
        else:
            not_improved += 1
            if not_improved == args.early_stop:
                print(f"Have not improved for {not_improved} epoches.", file=args.output_file, flush=True)
                break

        scheduler.step()
        print(f'Best validation MAE so far: {best_valid_mae}', file=args.output_file, flush=True)

    writer.close()
    args.output_file.close()
if __name__ == "__main__":
    args = parse_args()
    main(args)


if __name__ == "__main__":
    dataset = Dataset('/home/dy1/uspto_structure/csd_number.csv')

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)





