"""Bulk properties of a molecular crystal with an ASE calculator (UMA, MACE, ...).

relax_bulk           cell + positions (FrechetCellFilter), optionally keeping the
                     space group (ASE FixSymmetry, spglib), at a target pressure
relax_molecule       isolated molecule (gas-phase reference)
lattice_energy       E_latt = (E_crystal - sum_s n_s E_gas,s) / n_molecules  (eV and kJ/mol per
                     molecule; negative = bound) and the rigid-molecule harmonic estimate
                     dH_sub(T) = -E_latt - 2RT
equation_of_state    E(V) on isotropic scalings with rigid molecules (only COMs scale), with
                     the positions relaxed at each volume (relax=True), fitted by ASE's
                     Birch-Murnaghan EOS: V0, E0, B0 (GPa)
elastic_constants    C_ij (GPa) from central differences of the stress under +-delta strains,
                     positions relaxed at each strain; Voigt/Reuss/Hill moduli and the
                     Born stability check (all eigenvalues of C positive)

Energies are eV, lengths Angstrom, stresses as ASE (eV/Angstrom^3) converted to GPa
in results. Relaxation defaults (fmax 0.01 eV/Angstrom) are tighter than usual
because lattice energies of organics are differences of large numbers.
"""
from __future__ import annotations

import warnings

import numpy as np
from ase import units
from ase.filters import FrechetCellFilter

from .calculators import resolve
from .cif import MolecularCrystal

EV_TO_KJ_MOL = units.mol / units.kJ
OPTIMIZERS = ("BFGS", "LBFGS", "FIRE")


def _optimizer(name):
    from ase import optimize
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
    return getattr(optimize, name)


def _run(obj, optimizer, fmax, steps, trajectory=None, logfile=None):
    opt = _optimizer(optimizer)(obj, trajectory=trajectory, logfile=logfile)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    return converged, int(opt.nsteps)


def relax_positions(atoms, calculator, *, fmax=0.01, steps=500, optimizer="BFGS", fixed=None, logfile=None):
    """Relax positions at fixed cell (in place); fixed: atom indices held still. Returns (energy, converged, steps)."""
    from ase.constraints import FixAtoms
    atoms.calc = resolve(calculator)
    if fixed is not None and len(fixed):
        atoms.set_constraint(FixAtoms(indices=np.asarray(fixed)))
    converged, n = _run(atoms, optimizer, fmax, steps, logfile=logfile)
    energy = float(atoms.get_potential_energy())
    atoms.set_constraint()
    return energy, converged, n


def relax_bulk(crystal: MolecularCrystal, calculator, *, fmax=0.01, steps=1000, relax_cell=True, symmetrize=True,
               symprec=0.01, pressure_GPa=0.0, optimizer="BFGS", trajectory=None, logfile=None):
    """Relaxed copy of the crystal and a result dict.

    symmetrize keeps the space group found by spglib at symprec (ASE
    FixSymmetry); without spglib it is skipped with a warning. The result has
    energy (eV, whole cell), energy_per_molecule, converged, steps, the
    relaxed cell, and volume/density/cell changes against the input.
    """
    atoms = crystal.atoms.copy()
    atoms.calc = resolve(calculator)
    if symmetrize:
        try:
            from ase.constraints import FixSymmetry
            atoms.set_constraint(FixSymmetry(atoms, symprec=symprec))
        except ImportError:
            warnings.warn("spglib not available: relaxing without FixSymmetry")
            symmetrize = False
    target = FrechetCellFilter(atoms, scalar_pressure=pressure_GPa * units.GPa) if relax_cell else atoms
    converged, n = _run(target, optimizer, fmax, steps, trajectory, logfile)
    energy = float(atoms.get_potential_energy())
    atoms.set_constraint()
    relaxed = crystal.with_atoms(atoms)
    before, after = crystal.atoms.cell.cellpar(), relaxed.atoms.cell.cellpar()
    result = {
        "energy": energy, "energy_per_molecule": energy / crystal.n_molecules, "converged": converged, "steps": n,
        "symmetrized": symmetrize, "pressure_GPa": pressure_GPa,
        "cellpar_initial": before.tolist(), "cellpar": after.tolist(),
        "cell_length_change_pct": (100 * (after[:3] / before[:3] - 1)).tolist(),
        "cell_angle_change_deg": (after[3:] - before[3:]).tolist(),
        "volume_initial": float(crystal.atoms.get_volume()), "volume": float(relaxed.atoms.get_volume()),
        "volume_change_pct": float(100 * (relaxed.atoms.get_volume() / crystal.atoms.get_volume() - 1)),
        "density_g_cm3": relaxed.density,
    }
    if "experimental_cellpar" in crystal.source:
        exp = np.asarray(crystal.source["experimental_cellpar"])
        result["cell_length_vs_experiment_pct"] = (100 * (after[:3] / exp[:3] - 1)).tolist()
    if not converged:
        warnings.warn(f"bulk relaxation not converged in {n} steps (fmax={fmax})")
    return relaxed, result


def relax_molecule(molecule, calculator, *, fmax=0.01, steps=1000, optimizer="BFGS", box=None, charge=0, spin=1):
    """Relaxed isolated molecule and its energy (eV).

    box=None keeps it non-periodic; box=L puts it in a periodic L x L x L cell
    (for calculators that need a cell; L must exceed twice the model cutoff
    plus the molecule size).
    """
    mol = molecule.copy()
    mol.arrays.pop("mol_id", None)
    mol.info.update(charge=charge, spin=spin)
    if box:
        mol.set_cell([box] * 3)
        mol.center()
        mol.pbc = True
    else:
        mol.set_cell(None)
        mol.pbc = False
    energy, converged, n = relax_positions(mol, calculator, fmax=fmax, steps=steps, optimizer=optimizer)
    if not converged:
        warnings.warn(f"gas-phase relaxation of {mol.get_chemical_formula()} not converged in {n} steps")
    return mol, energy


def gas_references(crystal: MolecularCrystal, calculator, **options):
    """{formula: (relaxed molecule, energy)} for each molecular species of the crystal."""
    return {formula: relax_molecule(mol, calculator, **options)
            for formula, mol in crystal.representative_molecules().items()}


def lattice_energy(crystal: MolecularCrystal, crystal_energy, gas_energies, *, temperature_K=298.15):
    """Lattice energy per molecule from a crystal energy (whole cell) and {formula: E_gas}."""
    composition = crystal.composition
    missing = set(composition) - set(gas_energies)
    if missing:
        raise ValueError(f"no gas-phase energy for {sorted(missing)}")
    gas = sum(n * gas_energies[f] for f, n in composition.items())
    per_molecule = (crystal_energy - gas) / crystal.n_molecules
    rt = units.kB * temperature_K
    return {
        "lattice_energy_eV": float(per_molecule), "lattice_energy_kJ_mol": float(per_molecule * EV_TO_KJ_MOL),
        "sublimation_enthalpy_kJ_mol": float((-per_molecule - 2 * rt) * EV_TO_KJ_MOL),
        "temperature_K": temperature_K, "composition": composition,
    }


def scale_rigid(crystal: MolecularCrystal, factor):
    """Atoms with the cell scaled by factor (a matrix or a scalar) and molecules moved rigidly with their COMs."""
    F = np.eye(3) * factor if np.isscalar(factor) else np.asarray(factor, dtype=float)
    atoms = crystal.atoms.copy()
    masses = atoms.get_masses()
    for idx in crystal.molecules:
        com = masses[idx] @ atoms.positions[idx] / masses[idx].sum()
        atoms.positions[idx] += com @ F - com
    atoms.set_cell(atoms.cell.array @ F, scale_atoms=False)
    return atoms


def equation_of_state(crystal: MolecularCrystal, calculator, *, volume_range=0.06, n_points=7, relax=True,
                      fmax=0.02, steps=300, eos="birchmurnaghan"):
    """E(V) over volumes (1 +- volume_range) V, rigid-molecule scaling; returns V0, E0, B0_GPa and the data."""
    from ase.eos import EquationOfState
    calculator = resolve(calculator)
    factors = np.linspace(1 - volume_range, 1 + volume_range, int(n_points)) ** (1 / 3)
    volumes, energies = [], []
    for f in factors:
        atoms = scale_rigid(crystal, f)
        if relax:
            energy = relax_positions(atoms, calculator, fmax=fmax, steps=steps)[0]
        else:
            atoms.calc = calculator
            energy = float(atoms.get_potential_energy())
        volumes.append(atoms.get_volume())
        energies.append(energy)
    fit = EquationOfState(volumes, energies, eos=eos)
    v0, e0, b = fit.fit()
    return {"V0": float(v0), "E0": float(e0), "B0_GPa": float(b / units.GPa), "eos": eos,
            "volumes": list(map(float, volumes)), "energies": list(map(float, energies)),
            "V0_per_molecule": float(v0 / crystal.n_molecules)}


VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def elastic_constants(crystal: MolecularCrystal, calculator, *, delta=0.005, relax=True, fmax=0.005, steps=300):
    """6x6 elastic tensor (GPa, Voigt order xx yy zz yz xz xy) by central stress differences.

    Use a well-relaxed (zero-stress) structure. Engineering shear strains:
    Voigt component 4 strains e_yz = e_zy = delta/2. Positions are relaxed at
    each strain when relax=True (12 relaxations).
    """
    calculator = resolve(calculator)
    C = np.zeros((6, 6))
    for j, (a, b) in enumerate(VOIGT):
        stresses = []
        for sign in (1, -1):
            eps = np.zeros((3, 3))
            if a == b:
                eps[a, a] = sign * delta
            else:
                eps[a, b] = eps[b, a] = sign * delta / 2
            atoms = crystal.atoms.copy()
            atoms.set_cell(atoms.cell.array @ (np.eye(3) + eps), scale_atoms=True)
            if relax:
                relax_positions(atoms, calculator, fmax=fmax, steps=steps)
            else:
                atoms.calc = calculator
            stresses.append(atoms.get_stress(voigt=True))
        C[:, j] = (stresses[0] - stresses[1]) / (2 * delta) / units.GPa
    C = (C + C.T) / 2
    S = np.linalg.inv(C)
    kv = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[1, 2] + C[0, 2])) / 9
    gv = (C[0, 0] + C[1, 1] + C[2, 2] - C[0, 1] - C[1, 2] - C[0, 2] + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15
    kr = 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[1, 2] + S[0, 2]))
    gr = 15 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[1, 2] + S[0, 2])
               + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    eig = np.linalg.eigvalsh(C)
    kh, gh = (kv + kr) / 2, (gv + gr) / 2
    return {
        "C_GPa": C.tolist(), "eigenvalues_GPa": eig.tolist(), "mechanically_stable": bool(eig.min() > 0),
        "bulk_modulus_voigt": float(kv), "bulk_modulus_reuss": float(kr), "bulk_modulus_hill": float(kh),
        "shear_modulus_voigt": float(gv), "shear_modulus_reuss": float(gr), "shear_modulus_hill": float(gh),
        "youngs_modulus_hill": float(9 * kh * gh / (3 * kh + gh)),
        "poisson_ratio_hill": float((3 * kh - 2 * gh) / (2 * (3 * kh + gh))), "delta": delta,
    }
