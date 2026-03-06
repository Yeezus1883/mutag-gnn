from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol


def render_molecule(smiles):

    mol = Chem.MolFromSmiles(smiles)

    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)

    mol_block = Chem.MolToMolBlock(mol)

    viewer = py3Dmol.view(width=700, height=500)

    viewer.addModel(mol_block, "mol")

    viewer.setStyle({"stick": {}})

    viewer.setBackgroundColor("black")

    viewer.zoomTo()

    return viewer._make_html()