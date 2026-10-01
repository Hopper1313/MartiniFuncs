"""ASE <-> Psience conversion and small structure helpers.

Psience and McUtils are imported only by the functions that need them, so the
mass/XYZ helpers here work without those packages installed.
"""
from ase import Atoms
import numpy as np
from ase.data import chemical_symbols, atomic_masses
from ase.io import read


def __getattr__(name):
    """ANG_TO_BOHR / BOHR_TO_ANG from McUtils, computed on first access."""
    if name in ("ANG_TO_BOHR", "BOHR_TO_ANG"):
        from McUtils.Data import UnitsData
        source, target = ("Angstroms", "BohrRadius") if name == "ANG_TO_BOHR" else ("BohrRadius", "Angstroms")
        value = UnitsData.convert(source, target)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def ase_atoms_to_molecule(ase_atoms, charge=0, energy_evaluator=['aimnet2', None]):
    """Convert ASE coordinates in Angstrom to Psience coordinates in Bohr.

    energy_evaluator=None attaches no evaluator (geometry only). The default,
    ('aimnet2', None), keeps the historical behaviour of attaching Psience's
    'aimnet2' evaluator; ('aimnet2', model_path) selects a model file. For
    geometry-only display, visualization.to_display_molecule does the same
    conversion.

    r_bohr=r_ASE*ANG_TO_BOHR; ANG_TO_BOHR is McUtils' Angstrom-to-Bohr
    conversion. This legacy evaluator path is not the UMA workflow.
    """
    from Psience.Molecools import Molecule
    from . import coreio
    coords_bohr = ase_atoms.get_positions() * coreio.ANG_TO_BOHR
    symbols = ase_atoms.get_chemical_symbols()
    if energy_evaluator is None:
        return Molecule(atoms=symbols, coords=coords_bohr, charge=charge)
    name, model = energy_evaluator
    if model is not None and name == 'aimnet2':
        return Molecule(atoms=symbols, coords=coords_bohr,
                        energy_evaluator={'method': 'aimnet2', 'model': model}, charge=charge)
    return Molecule(atoms=symbols, coords=coords_bohr, energy_evaluator=name, charge=charge)

def molecule_to_ase_atoms(molecule):
    """Convert Psience coordinates in Bohr to ASE coordinates in Angstrom.

    r_angstrom=molecule.coords*BOHR_TO_ANG. No cell, constraints,
    forces, or velocities are transferred.
    """
    from . import coreio
    coords_ang = molecule.coords * coreio.BOHR_TO_ANG
    symbols = molecule.atoms
    ase_atoms = Atoms(symbols=symbols, positions=coords_ang)
    return ase_atoms

def check_if_internal_part_of_multiple_fragments(internal, molecule, only_bonds=True):
    """Return whether the selected internal atom-index set intersects more
    than one fragment. only_bonds restricts it to its first two indices;
    index 0 is a valid atom, not an empty intersection.
    """
    internal = internal[:2] if only_bonds else internal
    # Count how many fragments the internal is part of
    fragment_count = 0
    for fragment in molecule.fragment_indices:
        if set(internal) & set(fragment):
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
    """Choose the real element Z minimizing |atomic_masses[Z]-target_mass|
    with masses in amu. This heuristic cannot identify isotopes or coarse grains.
    """
    if not np.isfinite(target_mass) or target_mass <= 0:
        raise ValueError("Atomic mass must be positive and finite")
    atomic_number = min(range(1, len(atomic_masses)), key=lambda i: abs(atomic_masses[i] - target_mass))
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
