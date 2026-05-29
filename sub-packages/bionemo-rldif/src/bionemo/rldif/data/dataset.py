from torch.utils.data import Dataset
import json 
import numpy as np 
import pandas as pd
import torch 
from tqdm import tqdm
import ast
from Bio.PDB import PDBParser
from torch.utils.data import DataLoader

AMINO_ACIDS = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


class RLDIFDataset(Dataset):
    def __init__(
        self,
        pdb_paths
    ):
        super().__init__()
        self.RS = []

        iterator = tqdm(pdb_paths)

        for block in iterator:
            self.RS.append(self.preprocess_pdb(block))
                
        print(f"Processed {len(self.RS)} PDBs to generate sequences for")
    
    def preprocess_pdb(self, pdb_path):
        parser = PDBParser()
        structure = parser.get_structure("name", pdb_path)
        model = structure[0]
        chain = model["A"]
        seq = ""
        coords = {'CA': [], 'C': [], 'N': [], 'O': []}
        for residue in chain:
            if residue.get_resname() not in AMINO_ACIDS.keys():
                continue
            seq += AMINO_ACIDS[residue.get_resname()]
            coords['CA'].append(residue["CA"].get_coord())
            coords['C'].append(residue["C"].get_coord())
            coords['N'].append(residue["N"].get_coord())
            coords['O'].append(residue["O"].get_coord())
    
        return {
            "name": pdb_path.split("/")[-1].split('.')[0],
            "seq": seq,
            "coords": coords,
        }

    def preprocess_src(
        self,
        src: pd.DataFrame,
        index_for_redis: int = 0,
    ) -> list:

        src = src.reset_index()
        sample_dict_list = {
            int(index_for_redis + idx): self.get_sample_dict(src.loc[idx])
            for idx in src.index.values
        }
        return sample_dict_list

    def preprocess_src_samples_dict(
        self, src: pd.DataFrame, index_for_redis: int = 0
    ) -> dict:
        # reset the index as src dataframe might be filtered, these needs to be linear for redis indexing
        src = src.reset_index()
        sample_dict_dict = {
            int(index_for_redis + idx): self.get_sample_dict(src.loc[idx])
            for idx in src.index.values
        }
        return sample_dict_dict
    
    def get_sample_dict(self, row) -> dict:
        return {
            "title": row["name"],
            "seq": row["seq"],
            "CA": np.stack(row["CA"]),
            "N": np.stack(row["N"]),
            "C": np.stack(row["C"]),
            "O": np.stack(row["O"]),
            "score": 100.0,
        }

    def __len__(self) -> int:
        return len(self.RS)

    def __getitem__(self, idx):
        return self.RS[idx]

    def return_dataloader(self, model):
        return DataLoader(
                self,
                batch_size=4,
                shuffle=False,
                collate_fn=model.collate_fn,
            )
