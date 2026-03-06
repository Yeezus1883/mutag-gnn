from rdkit import Chem
import torch
from torch_geometric.data import Data

ATOM_MAP = {
    6: 0,   # C
    7: 1,   # N
    8: 2,   # O
    9: 3,   # F
    53: 4,  # I
    17: 5,  # Cl
    35: 6   # Br
}


def smiles_to_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    atoms = []

    for atom in mol.GetAtoms():

        atomic_num = atom.GetAtomicNum()

        feat = [0]*7

        if atomic_num in ATOM_MAP:
            feat[ATOM_MAP[atomic_num]] = 1

        atoms.append(feat)

    edges = []

    for bond in mol.GetBonds():

        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        edges.append([a,b])
        edges.append([b,a])

    x = torch.tensor(atoms, dtype=torch.float)
    edge_index = torch.tensor(edges).t().contiguous()

    return Data(x=x, edge_index=edge_index)