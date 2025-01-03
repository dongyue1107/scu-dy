from tqdm import tqdm
from rdkit.Chem import AllChem
from hotpot import Molecule as Mol
import os
import pandas as pd
from rdkit import Chem
import re
from typing import List
import csv

base_dir = '/path/to/your/base/directory'
smiles_data_dir = os.path.join(base_dir, 'smidata')
reaction_file = os.path.join(base_dir, 'reactionsmiles')
reaction_chunked_file = os.path.join(base_dir, 'React.csv')
product_chunked_file = os.path.join(base_dir, 'Prod.csv')
mol_dir = os.path.join(base_dir, 'pair')
output_mol_dir = os.path.join(base_dir, 'product')
output_csv_file = os.path.join(base_dir, 'Alkali_smi.csv')
metal_smile_file = os.path.join(base_dir, 'organic_metal_pair', 'metal(Sc)_smile.csv')


with open(os.path.join(smiles_data_dir, '1976_Sep2016_USPTOgrants_smiles.rsmi'), 'r') as f:
    lines = f.readlines()
data = []
for line in lines:
    data.append(line.split()[0])

with open(reaction_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['reactionsmile'])
    for item in data:
        writer.writerow([item])

def chunked(lst: List, size: int):
    return(lst[i:i+size] for i in range(0, len(lst), size))

chunk_size = 10000
df = pd.read_csv(reaction_file, chunksize=chunk_size)
results = []
for chunk in df:
    for i, r_smiles in tqdm(enumerate(chunk.iloc[:, 0])):
        try:
            rxn = AllChem.ReactionFromSmarts(r_smiles)
            products = [Chem.MolToSmiles(prod) for prod in rxn.GetProducts()]
            reactants = [Chem.MolToSmiles(react) for react in rxn.GetReactants()]
            result = {'reactants': reactants,
                      'products': products
            }
            results.append(result)
        except Exception as e:
            print(f'Error occurred at index {i}: {e}')
            continue

for i, chunk_results in enumerate(chunked(results, chunk_size)):

    reactants_df = pd.DataFrame.from_dict(
        {'Reactants {}'.format(j + 1): result['reactants'] for j, result in enumerate(chunk_results)}, orient='index')
    products_df = pd.DataFrame.from_dict(
        {'Products {}'.format(j + 1): result['products'] for j, result in enumerate(chunk_results)}, orient='index')
    reactants_df.to_csv(reaction_chunked_file, index=False, header=(i == 0), mode='a')
    products_df.to_csv(product_chunked_file, index=False, header=(i == 0), mode='a')
    if i > 10:
        break

df = pd.read_csv(reaction_chunked_file)
count = 3174337
for i, smile in enumerate(df.iloc[:, 0]):
    if pd.notnull(smile):
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            s = Chem.MolToMolFile(mol, f'{output_mol_dir}/{count}.mol')
            count += 1
        continue

# 将mol转为正则化smiles
unique_smiles = set()
canonical_smiles = []
mol_dir = '/home/dy1/sw/pair'
mol = [f for f in os.listdir(mol_dir)]
for n, i in enumerate(mol):
    mol_path = os.path.join(mol_dir, i)
    try:
        mol = Mol.read_from(mol_path)
        smi = Mol.dump(mol, fmt='smiles')
        mol = Chem.MolFromSmiles(smi)
        smiles = Chem.MolToSmiles(mol, canonical=True)
        if smiles not in unique_smiles:
            canonical_smiles.append(smiles)
            unique_smiles.add(smiles)
    except Exception as e:
        print(f"Error occurred in file {mol_path}: {e}")
        continue



smiles_list = list(unique_smiles)
with open(output_csv_file, 'w', newline='') as file:
    writer = csv.writer(file)

    for smiles in smiles_list:
        writer.writerow([smiles])

def save_mol_to_mol2(mol, output_file):
    writer = Chem.SDWriter(output_file)
    writer.write(mol)
    writer.close()


# smiles encode
charset = ['0', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
           'K', 'Ca', 'Sc', 'As', 'Ti', 'V', 'Cr', 'Mn', 'Ta', 'Pt', 'Fe', 'Te', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
           'Hg', 'Br', 'Sr', 'Mo', 'Y', 'Zr', 'Bi', 'I', 'In', 'W', 'Cs', 'Se', 'Ba',  'Hf', 'Ir', 'Ru', 'U', 'Er',
           'Rh', 'Dy', 'Au', 'La', 'Pd', 'Er', 'Gd', 'Nd', 'Tm', 'Cd', 'Re', 'Ag', 'Tb', 'Eu', 'Sm', 'Lu', 'Pr', 'Pu',
           'Ce', 'Th', 'Rb', 'Ce', 'Tc', 'Am', 'Tl', 'Cm', 'Cf', '1', '2', '3', '4', '5',
           '6', '7', '8', '9', '=', '#', ':', '+', '-', '[', ']', '(', ')', '/', '@', '.', '%', '\\', 'c', 'n', 's',
           'o', 'se', 'te', 'p', 'si', 'b', '*']

def encode_smiles(smiles):
    smi_list = list(filter(None, re.split(r'([A-Z][a-z]?|[s|t][e|i]|:|[|]|(|)|=|#|\+|-|/|@|.|%|\|)', smiles)))
    encoding = []
    for c in smi_list:
        try:
            if len(c) == 2 and c not in charset:
                encoding.extend([charset.index(c[0])]), charset.index(c[1])
            else:
                encoding.append(charset.index(c))
        except ValueError as e:
            print(f"Skipping character '{c}' due to ValueError: {e}")
            continue
    return encoding

df = pd.read_csv(metal_smile_file)
df['encoded_smiles'] = df['smiles'].apply(encode_smiles)
