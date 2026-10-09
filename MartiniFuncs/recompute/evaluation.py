"""Isolated single points and geometry optimizations of fragments, with any ASE calculator."""
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS, FIRE, LBFGS

from MartiniFuncs._checks import integer, positive

OPTIMIZERS = {"BFGS": BFGS, "LBFGS": LBFGS, "FIRE": FIRE}

# Overrides of the MD inference settings for fragment work. merge_mole and compile are
# the fixed-composition MD fast path; fragments change composition at every call, so
# both are off (same model numerics: merged and unmerged forces agree to ~1e-6 eV/A).
# This also runs on CPU and on systems without a C++ compiler for torch.compile.
FRAGMENT_INFERENCE_SETTINGS = {"merge_mole": False, "compile": False}


def uma_calculator(checkpoint, *, task="omol", device="cuda", inference_settings=None):
    """A UMA calculator for recomputing fragments: the MD settings with FRAGMENT_INFERENCE_SETTINGS.

    inference_settings may be a dict of further overrides (applied on top) or a
    fairchem preset name (used as given). Loads the checkpoint through
    workflows.force_validation.local_uma_calculator.
    """
    from MartiniFuncs.workflows.force_validation import local_uma_calculator
    if inference_settings is None or isinstance(inference_settings, dict):
        inference_settings = {**FRAGMENT_INFERENCE_SETTINGS, **(inference_settings or {})}
    return local_uma_calculator(checkpoint, task=task, device=device, inference_settings=inference_settings)


class FragmentEvaluator:
    """Energy (eV) and forces (eV/Angstrom) of a fragment of a structure, evaluated in isolation.

    The fragment is the given rows, copied into a new nonperiodic Atoms with only
    symbols and positions, plus info charge and spin (multiplicity), which UMA's
    omol task reads. The calculator must be the model/settings used for the MD
    forces (e.g. workflows.force_validation.local_uma_calculator), or F_env
    mixes in model differences.
    """

    def __init__(self, calculator, *, charge=0, spin=1):
        if not isinstance(charge, (int, np.integer)) or isinstance(charge, bool):
            raise ValueError("charge must be an integer")
        self.calculator = calculator
        self.charge = int(charge)
        self.spin = integer(spin, "spin")

    def isolate(self, structure, rows):
        """The fragment rows as a new finite Atoms (calculator attached)."""
        rows = np.asarray(rows, dtype=int)
        fragment = Atoms(np.asarray(structure.get_chemical_symbols())[rows].tolist(),
                         positions=structure.positions[rows], pbc=False)
        fragment.info.update(charge=self.charge, spin=self.spin)
        fragment.calc = self.calculator
        return fragment

    def evaluate(self, structure, rows):
        """(energy, forces (len(rows), 3)) of the isolated fragment at its current geometry."""
        fragment = self.isolate(structure, rows)
        return float(fragment.get_potential_energy()), np.array(fragment.get_forces(apply_constraint=False))


@dataclass(frozen=True)
class Relaxation:
    """Result of FragmentRelaxer.relax: final geometry, energies (eV), unconstrained final forces."""
    positions: np.ndarray
    initial_energy: float
    final_energy: float
    forces: np.ndarray
    converged: bool
    steps: int


class FragmentRelaxer:
    """Geometry optimization of an isolated fragment, optionally with some of its atoms fixed.

    optimizer is one of BFGS, LBFGS, FIRE (ase.optimize); convergence is the
    maximum force on free atoms <= fmax (eV/Angstrom) within steps iterations.
    """

    def __init__(self, evaluator, *, optimizer="BFGS", fmax=0.05, steps=500):
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {sorted(OPTIMIZERS)}")
        self.evaluator = evaluator
        self.optimizer = optimizer
        self.fmax = positive(fmax, "fmax")
        self.steps = integer(steps, "steps", minimum=0)

    def relax(self, structure, rows, *, fixed=()):
        """Optimize the fragment rows; fixed are positions within rows (0-based) held in place."""
        fragment = self.evaluator.isolate(structure, rows)
        fixed = np.asarray(fixed, dtype=int)
        if len(fixed):
            fragment.set_constraint(FixAtoms(indices=fixed))
        initial = float(fragment.get_potential_energy())
        optimizer = OPTIMIZERS[self.optimizer](fragment, logfile=None)
        converged = bool(optimizer.run(fmax=self.fmax, steps=self.steps))
        return Relaxation(fragment.positions.copy(), initial, float(fragment.get_potential_energy()),
                          np.array(fragment.get_forces(apply_constraint=False)), converged, int(optimizer.nsteps))
