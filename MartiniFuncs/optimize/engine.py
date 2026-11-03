"""Constrained geometry optimization of one finite structure with a selectable model.

Models (each needs its package in the running environment and, for the ML
models, its weights file):

    uma      fairchem UMA, task omol             UMA_CHECKPOINT
    aimnet   AIMNet2 (aimnetcentral "aimnet")    AIMNET_MODEL
    mace     MACE-OMOL (mace-torch)              MACE_MODEL
    gfn2     GFN2-xTB (tblite)
    emt      ASE EMT (crude; for testing without ML models)

Charge and spin multiplicity are passed per run (atoms.info "charge"/"spin"
for UMA and MACE; AIMNet2 and tblite take them as arguments).

Constraints (constraint["mode"]):

    none                no constraint
    fix_atoms           atoms (row indices) held in place
    fix_distances       pairs [[i, j], ...] held at their starting distances
    restrain_distances  harmonic restraints 0.5 k (d - d0)^2 on pairs; d0 from
                        "targets" (Angstrom) or the starting distances, k in eV/Angstrom^2

Optimizers: BFGS, LBFGS, FIRE (ASE), sella_min (Sella, minimum) and sella_ts
(Sella, first-order saddle). ASE optimizers use ASE constraints (FixAtoms,
FixBondLengths); Sella uses its own internal-coordinate constraints. Each step
is reported through a callback; a callback may raise Cancelled to stop.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

MODELS = {
    "uma": {"label": "UMA (uma-s-1p2p1, omol)", "package": "fairchem", "weights": "UMA_CHECKPOINT"},
    "aimnet": {"label": "AIMNet2", "package": "aimnet", "weights": "AIMNET_MODEL"},
    "mace": {"label": "MACE-OMOL", "package": "mace", "weights": "MACE_MODEL"},
    "gfn2": {"label": "GFN2-xTB (tblite)", "package": "tblite", "weights": None},
    "emt": {"label": "EMT (test only)", "package": "ase", "weights": None},
}
OPTIMIZERS = ("BFGS", "LBFGS", "FIRE", "sella_min", "sella_ts")
CONSTRAINT_MODES = ("none", "fix_atoms", "fix_distances", "restrain_distances")


class Cancelled(Exception):
    """Raised by a step callback to stop an optimization."""


def model_status(names=None):
    """{model: {"label", "available", "reason"}} for this environment (imports nothing heavy)."""
    status = {}
    for name in names or MODELS:
        spec = MODELS[name]
        reason = ""
        if importlib.util.find_spec(spec["package"]) is None:
            reason = f"package {spec['package']!r} not installed in this environment"
        elif spec["weights"] and not Path(os.environ.get(spec["weights"], "")).is_file():
            reason = f"weights not found: set {spec['weights']} (now {os.environ.get(spec['weights'], '')!r})"
        status[name] = {"label": spec["label"], "available": not reason, "reason": reason}
    return status


class ModelCache:
    """Loads each model once; calculator(name, charge, spin) returns a ready ASE calculator."""

    def __init__(self):
        self._loaded = {}

    def is_loaded(self, name):
        return name in self._loaded

    def _load(self, name):
        if name == "uma":
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
            return FAIRChemCalculator(pretrained_mlip.load_predict_unit(os.environ["UMA_CHECKPOINT"], device="cpu"),
                                      task_name="omol")
        if name == "aimnet":
            from aimnet.calculators import AIMNet2Calculator
            return AIMNet2Calculator(os.environ["AIMNET_MODEL"], device="cpu")
        if name == "mace":
            from mace.calculators import mace_omol
            return mace_omol(model=os.environ["MACE_MODEL"], device="cpu", default_dtype="float64")
        if name in ("gfn2", "emt"):
            return None   # cheap: built per run
        raise ValueError(f"Unknown model {name!r}; available: {sorted(MODELS)}")

    def calculator(self, name, *, charge=0, spin=1):
        if name not in self._loaded:
            self._loaded[name] = self._load(name)
        base = self._loaded[name]
        if name == "aimnet":
            from aimnet.calculators import AIMNet2ASE
            return AIMNet2ASE(base, charge=charge, mult=spin)
        if name == "gfn2":
            from tblite.ase import TBLite
            return TBLite(method="GFN2-xTB", charge=charge, multiplicity=spin, verbosity=0)
        if name == "emt":
            from ase.calculators.emt import EMT
            return EMT()
        return base


class RestrainedCalculator(Calculator):
    """A calculator plus harmonic distance restraints 0.5 k (d - d0)^2 (eV, Angstrom).

    results["restraint_energy"] holds the restraint part of the energy.
    """
    implemented_properties = ["energy", "forces"]

    def __init__(self, base, pairs, targets, k):
        super().__init__()
        self.base, self.pairs, self.targets, self.k = base, [tuple(map(int, p)) for p in pairs], list(targets), float(k)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        copy = self.atoms.copy()
        copy.calc = self.base
        energy, forces = copy.get_potential_energy(), copy.get_forces().copy()
        restraint = 0.0
        for (i, j), target in zip(self.pairs, self.targets):
            vector = self.atoms.positions[j] - self.atoms.positions[i]
            distance = np.linalg.norm(vector)
            restraint += 0.5 * self.k * (distance - target) ** 2
            pull = self.k * (distance - target) * vector / distance      # -dE/dr_j
            forces[i] += pull
            forces[j] -= pull
        self.results = {"energy": energy + restraint, "forces": forces, "restraint_energy": restraint}


def _check(constraint, n_atoms):
    mode = constraint.get("mode", "none")
    if mode not in CONSTRAINT_MODES:
        raise ValueError(f"constraint mode must be one of {CONSTRAINT_MODES}")
    atoms = [int(i) for i in constraint.get("atoms", [])]
    pairs = [tuple(int(i) for i in p) for p in constraint.get("pairs", [])]
    if any(not 0 <= i < n_atoms for i in atoms + [i for p in pairs for i in p]) or any(len(set(p)) != 2 for p in pairs):
        raise ValueError("constraint atoms/pairs must be distinct row indices of the structure")
    if mode == "fix_atoms" and not atoms:
        raise ValueError("fix_atoms needs at least one atom")
    if mode in ("fix_distances", "restrain_distances") and not pairs:
        raise ValueError(f"{mode} needs at least one pair")
    return mode, atoms, pairs


def optimize(symbols, positions, *, calculator, constraint=None, optimizer="BFGS", fmax=0.05, steps=500,
             charge=0, spin=1, callback=None):
    """Optimize one structure; return (final Atoms, converged, number of steps).

    callback(frame) is called for the start geometry (step 0) and after every
    step with {"step", "energy", "restraint_energy", "fmax", "positions"};
    energies in eV, fmax the largest constraint-projected force (ASE
    optimizers) or the largest raw force (Sella), eV/Angstrom.
    """
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
    if not (np.isfinite(fmax) and fmax > 0) or int(steps) < 1:
        raise ValueError("fmax must be positive and steps at least 1")
    atoms = Atoms(list(symbols), positions=np.asarray(positions, dtype=float))
    atoms.info.update(charge=int(charge), spin=int(spin))
    mode, fixed, pairs = _check(constraint or {}, len(atoms))
    sella = optimizer.startswith("sella")
    if mode == "restrain_distances":
        targets = constraint.get("targets") or [atoms.get_distance(i, j) for i, j in pairs]
        calculator = RestrainedCalculator(calculator, pairs, targets, constraint.get("k", 10.0))
    atoms.calc = calculator
    sella_constraints = None
    if mode == "fix_atoms":
        if sella:
            from sella import Constraints
            sella_constraints = Constraints(atoms)
            for i in fixed:
                sella_constraints.fix_translation(i)
        else:
            from ase.constraints import FixAtoms
            atoms.set_constraint(FixAtoms(indices=fixed))
    elif mode == "fix_distances":
        if sella:
            from sella import Constraints
            sella_constraints = Constraints(atoms)
            for pair in pairs:
                sella_constraints.fix_bond(pair)
        else:
            from ase.constraints import FixBondLengths
            # ASE's default tolerance (1e-13) cannot be met with single-precision ML forces.
            atoms.set_constraint(FixBondLengths(pairs, tolerance=1e-8, iterations=1000))

    step = {"n": 0}

    def report():
        if callback is None:
            return
        forces = atoms.get_forces()
        callback({"step": step["n"], "energy": float(atoms.get_potential_energy()),
                  "restraint_energy": float(atoms.calc.results.get("restraint_energy", 0.0)),
                  "fmax": float(np.sqrt((forces ** 2).sum(axis=1)).max()),
                  "positions": np.round(atoms.positions, 6).tolist()})
        step["n"] += 1

    if sella:
        from sella import Sella
        runner = Sella(atoms, order=1 if optimizer == "sella_ts" else 0, internal=True,
                       constraints=sella_constraints, logfile=None)
    else:
        from ase import optimize as ase_optimize
        runner = getattr(ase_optimize, optimizer)(atoms, logfile=None)
    # Optimizers call their observers for the start geometry and after every step.
    runner.attach(report, interval=1)
    converged = bool(runner.run(fmax=float(fmax), steps=int(steps)))
    return atoms, converged, step["n"] - 1
