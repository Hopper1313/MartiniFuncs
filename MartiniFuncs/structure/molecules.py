"""Small molecule helpers: element maps, ID subsets, and 3D templates from SMILES.

Component and site centers of mass are structure.graph.component_center_of_mass;
point wrapping is structure.periodic.wrap_points.
"""
from __future__ import annotations
from collections.abc import Iterable
from ase import Atoms
from MartiniFuncs.structure import graph as mfg


def get_element_map(atoms: Atoms, index_array: str = "global_index") -> dict[int, str]:
    """Return ``persistent_atom_id -> element symbol`` (creates the ID array if absent)."""
    ids, _ = mfg.get_index_maps(atoms, index_array=index_array)
    return {
        int(atom_id): symbol
        for atom_id, symbol in zip(ids, atoms.get_chemical_symbols())
    }


def _subset(atoms: Atoms, atom_ids: Iterable[int], index_array="global_index") -> Atoms:
    """Slice ``atoms`` using persistent IDs rather than local ASE indices."""
    local = mfg.local_indices(atoms, atom_ids, index_array=index_array)
    return atoms[local]


def embed_smiles(smiles, seed=61453):
    """RDKit molecule with hydrogens, embedded (ETKDG) and relaxed with MMFF94; deterministic for a seed."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    molecule = Chem.AddHs(molecule)
    if AllChem.EmbedMolecule(molecule, randomSeed=seed) != 0:
        raise RuntimeError(f"RDKit could not embed {smiles}")
    AllChem.MMFFOptimizeMolecule(molecule, maxIters=2000)
    return molecule


def atoms_from_smiles(smiles, seed=61453):
    """A whole, nonperiodic ASE molecule (Angstrom) from SMILES via embed_smiles.

    A geometry template (e.g. for structure.packing), not an optimized
    structure for the production model: relax it with that model if needed.
    """
    molecule = embed_smiles(smiles, seed)
    return Atoms([a.GetSymbol() for a in molecule.GetAtoms()], positions=molecule.GetConformer().GetPositions())
