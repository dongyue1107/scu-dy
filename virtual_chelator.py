import pandas as pd
import itertools
from ccdc import molecule
from tqdm import tqdm
from rdkit import Chem
from concurrent.futures import ProcessPoolExecutor, as_completed


input_file_path = 'data/input/6ml.csv'
output_file_base_path = 'data/output/predictions/cn/cn_'

df = pd.read_csv(input_file_path)

num_chunks = 80
chunks = [df[i::num_chunks] for i in range(num_chunks)]

new_data = []

def save_data(new_data, file_index):
    output_file = f'{output_file_base_path}{file_index}.csv'  # 使用基础路径与文件索引拼接
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(output_file, index=False)

def process_chunk(chunk, file_index):
    local_data = []

    for index, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"Processing Rows (Part {file_index})"):
        smi = row['SMILES']
        mol_name = row['index']
        pair_number = int(row['pair_number'])
        inde = row['in']
        mol = molecule.Molecule.from_string(smi)
        mol.remove_hydrogens()
        atom_number = len(mol.atoms)

        if atom_number > 100:
            continue
        try:
            metal = mol.atom('Ga1')
            connected_atoms = metal.neighbours[0].label

            atoms = [atom.label for atom in mol.atoms if atom.atomic_symbol in ['O', 'N']]
            atoms = list(filter(lambda atom: atom not in connected_atoms, atoms))
            if len(atoms) > 13:
                continue
            possible_combinations = [combo for combo in itertools.product(['Ga1'], atoms)]
            all_combinations = list(itertools.combinations(possible_combinations, pair_number - 1))
            unique_combinations = set(all_combinations)
            unique_combinations_list = list(unique_combinations)

            for i, combination in enumerate(unique_combinations_list):
                try:
                    mol = molecule.Molecule.from_string(smi)
                    for bond in combination:
                        atom1_label = bond[0]
                        atom2_label = bond[1]
                        atom1 = next(atom for atom in mol.atoms if atom.label == atom1_label)
                        atom2 = next(atom for atom in mol.atoms if atom.label == atom2_label)
                        mol.add_bond('Single', atom1, atom2)

                    cn_smiles = mol.smiles
                    local_data.append({
                        'smiles': mol_name,
                        'ml_SMILES': smi,
                        'cn_SMILES': cn_smiles,
                        'index': inde,
                        'pair_number': pair_number
                    })

                except RuntimeError as e:
                    continue

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue

    save_data(local_data, file_index)

def parallel_processing():
    with ProcessPoolExecutor() as executor:
        futures = []
        for i, chunk in enumerate(chunks, 1):
             futures.append(executor.submit(process_chunk, chunk, i))
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    parallel_processing()

def count_N_O_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol:
        N_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
        O_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
        return N_count + O_count
    else:
        return 0

df['N_O_sum'] = df['smiles'].apply(count_N_O_atoms)
df['atom_sum_range'] = pd.cut(df['N_O_sum'], bins=range(0, df['N_O_sum'].max() + 3, 2), right=False)
atom_sum_counts = df['atom_sum_range'].value_counts().sort_index()
result_df = atom_sum_counts.reset_index()
result_df.columns = ['N-O-atom', 'counts']

df['atom_range'] = pd.cut(df['atom_count'], bins=range(0, df['atom_count'].max() + 6, 5), right=False)
atom_range_counts = df['atom_range'].value_counts().sort_index()
result_df = atom_range_counts.reset_index()
result_df.columns = ['atom', 'counts']