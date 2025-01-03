import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data
from pathlib import Path
from feature import smiles2graph
from model import GINGraphPooling, MyEvaluator
import torch
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from torch_geometric.loader import DataLoader
import os
import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='Graph data miming with GNN')
    parser.add_argument('--task_name', type=str, default='GINGraphPooling',
                        help='task name')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='dimensionality of hidden units in GNNs (default: 256)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.)')
    parser.add_argument('--save_test', action='store_true')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 100)')
    # parser.add_argument('--num_classes', type=int, default=10,
    #                     help='number of dataset unique labels')
    parser.add_argument('--weight_decay', type=float, default=0.0004,
                        help='weight decay')
    parser.add_argument('--early_stop', type=int, default=30,
                        help='early stop (default: 10)')
    parser.add_argument('--num_workers', type=int, default=0,  # 4,
                        help='number of workers (default: 4)')
    parser.add_argument('--filepath', type=str,
                        default='/home/dy/dicizi/CN/Sc/gaussian_exist_encode.xlsx',
                        help='path to the input file')
    args = parser.parse_args()

    return args

import sys
sys.argv=['']
del sys

class MyDataset(Dataset):

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.data_df = pd.read_csv(self.filepath)
        self.smiles_list = self.data_df['original_smiles']
        self.index_list = self.data_df['processed_smiles']

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        index = self.index_list[idx]

        graph = smiles2graph(smiles)

        if graph is None or graph['node_feat'] is None or graph['edge_index'] is None or graph['edge_feat'] is None:
            print(f"Skipping entry at index {idx} due to None graph.")
            return None

        assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert(len(graph['node_feat']) == graph['num_nodes'])

        x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        num_nodes = int(graph['num_nodes'])
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, smiles=smiles, index=index)


        return data


def prepartion(args):

    save_dir = os.path.join('/home/dy/1/GIN/predictions', args.task_name)

    if os.path.exists(save_dir):
        for idx in range(1000):
            if not os.path.exists(save_dir + '=' + str(idx)):
                save_dir = save_dir + '=' + str(idx)
                break

    args.save_dir = save_dir
    os.makedirs(args.save_dir, exist_ok=True)

    args.output_file = open(os.path.join(args.save_dir, 'output'), 'a')
    print(args, file=args.output_file, flush=True)


def main(args):
    prepartion(args)

    nn_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling,
    }

    dataset = MyDataset(
        filepath='/home/dy/1/uspto/ml/ml-l/sorted_chunk_1.csv')
    test_data = [dataset[idx] for idx in range(len(dataset))]  # Fix this line

    test_loader = DataLoader(test_data, batch_size=256, shuffle=True, num_workers=args.num_workers)

    # Choose the device based on availability (CPU/GPU)
    device = torch.device('cpu')

    model = GINGraphPooling(**nn_params).to(device)

    checkpoint = torch.load('/home/dy/uspto_structure/csd_number_model/0.848/checkpoint.pt',
                          map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}', file=args.output_file, flush=True)
    print(model, file=args.output_file, flush=True)

    smiles_list = []
    index_list = []
    bl_list = []
    predictions = []

    # Testing loop
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            smiles_list.extend(data.smiles)
            index_list.extend(data.index)
            # bl_list.extend([item.item() for item in data.bl])
            predictions.extend([item.item() for sublist in output.float().cpu().numpy() for item in sublist])


    result_df = pd.DataFrame({'SMILES': smiles_list, 'pair_number': predictions, 'index': index_list})
    result_df.to_csv('/home/dy/1/GIN/predictions/1.csv', index=False)

    print(f"Predictions saved to {os.path.join(args.save_dir, 'predictions_60000.csv')}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
