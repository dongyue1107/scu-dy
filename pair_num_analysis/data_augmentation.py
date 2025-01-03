import pandas as pd
from rdkit import Chem
import random
from tqdm import tqdm
from timeout_decorator import timeout, TimeoutError
from itertools import product

# 示例数据框

df = pd.read_csv('/home/dy/sw/big/Ga_nosame.csv')

# 定义数据增广函数
def randomize_smiles(smiles, random_type="shuffle"):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    if random_type == "shuffle":
        return Chem.MolToSmiles(mol, doRandom=True)
    elif random_type == "rotated":
        return rotate_smiles(mol)
    elif random_type == "canonical":
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        raise ValueError(f"Unknown randomization type: {random_type}")

def rotate_smiles(mol):
    # 将分子旋转并生成新的SMILES
    smiles = Chem.MolToSmiles(mol)
    n = len(smiles)
    i = random.randint(1, n-1)
    return smiles[i:] + smiles[:i]

@timeout(10)
def augment_smiles(smiles, random_type="shuffle", max_iterations=10):
    augmented_smiles = set()
    while len(augmented_smiles) < max_iterations:
        try:
            new_smiles = randomize_smiles(smiles, random_type)
            augmented_smiles.add(new_smiles)
        except ValueError as e:
            print(f"Error: {e}")
            # 跳过无效的SMILES字符串
            break
    return list(augmented_smiles)

# 数据增广并生成新的数据框
augmented_data = []
n_augments = 5

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting SMILES"):
    if row['bond_number'] < 7:
        smiles = row['rm_smiles']
        all_smiles = row['smiles']
        try:
            augmented_smiles_list = augment_smiles(smiles, max_iterations=10)
            augmented_all_smiles_list = augment_smiles(all_smiles, max_iterations=10)
            if not augmented_smiles_list or not augmented_all_smiles_list:
                # 保存原始数据
                augmented_data.append(row)
                continue
            for aug_smiles, aug_all_smiles in zip(augmented_smiles_list, augmented_all_smiles_list):
                new_row = row.copy()
                new_row['rm_smiles'] = aug_smiles
                new_row['smiles'] = aug_all_smiles
                augmented_data.append(new_row)


        except TimeoutError:
            print(f"Timeout error occurred for SMILES: {smiles}. Skipping this SMILES.")
            augmented_data.append(row)
        except ValueError:
            augmented_data.append(row)

augmented_df = pd.DataFrame(augmented_data)

# 输出新的数据框
print(augmented_df)
# 去重两列，并给键长计算平均值
agg_functions = {
    'bond_l': 'mean',              # 计算 bond_l 列的平均值
    'identifier': 'first',        # 保留每组中第一个 identifier 值
    'normalize_bl': 'mean',
    'ideal_bl': 'mean',
    'neigh': 'first',
    'bond_number': 'mean'# 保留每组中第一个 some_other_col 值
}
bond_l_avg = df.groupby(['smiles', 'rm_smiles']).agg(agg_functions).reset_index()

df_filtered = bond_l_avg[bond_l_avg['rm_smiles'].apply(lambda x: len(x) <= 995)]