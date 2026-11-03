"""ASE <-> Psience conversion and small structure helpers.

Psience and McUtils are imported only by the functions that need them, so the
mass/XYZ helpers here work without those packages installed.
"""
from ase import Atoms
from ase.io import read

from MartiniFuncs.io.lammps import mass_to_symbol

# Historical name; one nearest-mass rule for the whole package.
get_symbol_from_mass = mass_to_symbol


def _angstrom_to_bohr():
    """McUtils' Angstrom-to-Bohr factor (imports McUtils on first use)."""
    from McUtils.Data import UnitsData
    return UnitsData.convert("Angstroms", "BohrRadius")


def __getattr__(name):
    """ANG_TO_BOHR / BOHR_TO_ANG from McUtils, computed on first access."""
    if name in ("ANG_TO_BOHR", "BOHR_TO_ANG"):
        factor = _angstrom_to_bohr()
        value = factor if name == "ANG_TO_BOHR" else 1 / factor
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def ase_atoms_to_molecule(ase_atoms, charge=0, energy_evaluator=("aimnet2", None)):
    """Convert ASE coordinates in Angstrom to Psience coordinates in Bohr.

    energy_evaluator=None attaches no evaluator (geometry only), as the
    chemistry visualization helpers use it. The default,
    ('aimnet2', None), keeps the historical behaviour of attaching Psience's
    'aimnet2' evaluator; ('aimnet2', model_path) selects a model file.

    r_bohr=r_ASE*ANG_TO_BOHR; ANG_TO_BOHR is McUtils' Angstrom-to-Bohr
    conversion. This legacy evaluator path is not the UMA workflow.
    """
    from Psience.Molecools import Molecule
    coords_bohr = ase_atoms.get_positions() * _angstrom_to_bohr()
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

    r_angstrom=molecule.coords/ANG_TO_BOHR. No cell, constraints,
    forces, or velocities are transferred.
    """
    return Atoms(symbols=molecule.atoms, positions=molecule.coords / _angstrom_to_bohr())


def get_geoms_from_multixyz(xyzfile):
    """All structures of a multi-frame XYZ file as ASE Atoms."""
    return read(xyzfile, index=':')
