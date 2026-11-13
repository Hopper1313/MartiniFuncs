"""Organic molecular crystals: CIF -> whole-molecule crystal -> slabs -> interfaces, with MLIP properties.

Modules
-------
cif           read_cif (symmetry operators from the CIF, disorder, whole molecules), MolecularCrystal
slabs         Miller-plane bases, molecule-preserving terminations, build_slab, in-plane supercells,
              flip_slab
interfaces    match_lattices (pymatgen ZSL + strain), build_interface (rigid-molecule strain),
              set_collision_velocities, separation_curve
calculators   crystal_calculator: UMA (task omc), MACE-OFF, MACE-MP, any ASE calculator
bulk          relax_bulk, gas references, lattice_energy, equation_of_state, elastic_constants
surfaces      surface_energy, face_energies, thickness_convergence, molecule_removal_energy,
              surface_sites, site_removal_energies
morphology    distinct_miller_indices, systematic absences, bfdh, wulff
workflow      run_crystal_workflow and its command line (python -m MartiniFuncs.crystals.workflow)

pymatgen (and spglib) are needed for interfaces, morphology and symmetry-constrained relaxations:
pip install "MartiniFuncs[crystals]".
"""
from importlib import import_module

_API = {
    "MolecularCrystal": "cif", "read_cif": "cif", "crystal_from_atoms": "cif", "write_cif": "cif",
    "build_slab": "slabs", "terminations": "slabs", "flip_slab": "slabs", "slab_supercell": "slabs",
    "match_lattices": "interfaces", "build_interface": "interfaces", "set_collision_velocities": "interfaces",
    "crystal_calculator": "calculators",
    "relax_bulk": "bulk", "lattice_energy": "bulk", "gas_references": "bulk",
    "equation_of_state": "bulk", "elastic_constants": "bulk",
    "surface_energy": "surfaces", "face_energies": "surfaces", "thickness_convergence": "surfaces",
    "molecule_removal_energy": "surfaces", "surface_sites": "surfaces", "site_removal_energies": "surfaces",
    "bfdh": "morphology", "wulff": "morphology", "distinct_miller_indices": "morphology",
    "run_crystal_workflow": "workflow",
}

__all__ = sorted(_API)


def __getattr__(name):
    if name in _API:
        value = getattr(import_module(f"{__name__}.{_API[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_API))
