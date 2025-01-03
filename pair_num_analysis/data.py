import pandas as pd
import os.path as osp
import torch
from torch_geometric.loader import DataLoader
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from rdkit import RDLogger
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from pathlib import Path
from feature import smiles2graph

RDLogger.DisableLog('rdApp.*')



class MyDataset(Dataset):

    def __init__(self, filepath):
        # super(MyDataset, self).__init__()
        self.filepath = Path(filepath)
        self.data_df = pd.read_csv(self.filepath)
        self.smiles_list = self.data_df['standard_smiles']
        self.pair_number = self.data_df['pair_number']

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        # smiles_, pair_number_ = self.smiles_list[idx], self.pair_number[idx]
        # smiles = smiles_.to_list()
        # pair_number = pair_number_.to_list()
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

    # def get_idx_split(self):
    #     split_dict = replace_numpy_with_torchtensor(torch.load(self.filepath))
    #     return split_dict
    def get_idx_split(self):
        split_dict = {"train": [], "valid": [], "test": []}

        # 使用train_test_split函数将数据集分成训练、验证和测试集
        train_data, test_data = train_test_split(self.data_df, test_size=0.2, random_state=42)
        valid_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

        split_dict["train"] = train_data.index.tolist()
        split_dict["valid"] = valid_data.index.tolist()
        split_dict["test"] = test_data.index.tolist()
        return split_dict

if __name__ == "__main__":
    dataset = MyDataset('/home/dy1/uspto_structure/remove_bond_pair_number.csv')

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

