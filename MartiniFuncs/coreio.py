from Psience.Molecools import Molecule
from McUtils.Data import UnitsData
from ase import Atoms
import McUtils.Coordinerds as coordops
import numpy as np
from ase.data import chemical_symbols, atomic_masses
from ase.io import read

ANG_TO_BOHR = UnitsData.convert("Angstroms", "BohrRadius")
BOHR_TO_ANG = UnitsData.convert("BohrRadius", "Angstroms")

def ase_atoms_to_molecule(ase_atoms, charge=0,energy_evaluator=['aimnet2',None]):
    """Convert ASE coordinates in Angstrom to Psience coordinates in Bohr.

    This legacy helper also attaches an evaluator. For geometry-only display,
    use visualization.to_display_molecule, which applies the same conversion.
    """
    coords_bohr = ase_atoms.get_positions() * ANG_TO_BOHR
    symbols = ase_atoms.get_chemical_symbols()
    # energy_evaluator={'method':'aimnet2', 'model':path/to/jpt}
    if energy_evaluator[1] is not None and energy_evaluator[0] == 'aimnet2':
        mol=Molecule(atoms=symbols, coords=coords_bohr, energy_evaluator={'method':'aimnet2', 'model':energy_evaluator[1]},charge=charge)
    else:
        mol = Molecule(atoms=symbols, coords=coords_bohr, energy_evaluator='aimnet2', charge=charge)
    return mol

def molecule_to_ase_atoms(molecule):
    """Convert Psience coordinates in Bohr to ASE coordinates in Angstrom."""
    coords_ang = molecule.coords * BOHR_TO_ANG
    symbols = molecule.atoms
    ase_atoms = Atoms(symbols=symbols, positions=coords_ang)
    return ase_atoms

def check_if_internal_part_of_multiple_fragments(internal, molecule, only_bonds=True):
    internal = internal[:2] if only_bonds else internal
    # Count how many fragments the internal is part of
    fragment_count = 0
    for fragment in molecule.fragment_indices:
        if any(set(internal) & set(fragment)):
            fragment_count += 1
            if fragment_count > 1:
                break
    return fragment_count > 1

def initialize_constraints_boolean(molecule):
    # Nx3 boolean array. One row for each internal (bond, angle, dihedral)
    # Angles and dihedrals always False. Bonds False if part of multiple fragments, True otherwise
    constraints = np.zeros((len(molecule.get_canonical_zmatrix()), 3), dtype=bool)
    for i, internal in enumerate(molecule.get_canonical_zmatrix()):
        constraints[i, 0] = not check_if_internal_part_of_multiple_fragments(internal, molecule)
    return constraints

def get_symbol_from_mass(target_mass):
    # Find the atomic number (index) where the absolute difference in mass is minimal
    atomic_number = min(range(len(atomic_masses)), key=lambda i: abs(atomic_masses[i] - target_mass))
    return chemical_symbols[atomic_number]

def get_geoms_from_multixyz(xyzfile):
    return read(xyzfile, index=':')

def coords_str_to_ase(coords_str):
    # Converts:
    # [
    #   "H 0.0 0.0 0.0",
    #   "O 0.0 0.0 1.0"
    # ]
    # to an ASE Atoms object
    symbols = []
    positions = []
    for line in coords_str:
        parts = line.split()
        symbols.append(parts[0])
        positions.append([float(x) for x in parts[1:4]])
    return Atoms(symbols=symbols, positions=positions)
