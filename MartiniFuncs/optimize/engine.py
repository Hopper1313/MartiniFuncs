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

Single points and vibrations: single_point gives energy and forces; hessian
the Cartesian Hessian (eV/Angstrom^2) by method "analytic" (the model's own
second derivatives: UMA, AIMNet2, MACE-OMOL; ModelCache.analytic_hessian),
"finite_diff" (central differences of the forces, 6N force calls) or "auto"
(analytic when the model has it, else finite differences); vibrations turns a
Hessian into harmonic frequencies (cm^-1, imaginary as negative) and normal
modes, with translations and rotations projected out.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
from ase import Atoms, units
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
HESSIAN_METHODS = ("auto", "analytic", "finite_diff")
ANALYTIC_HESSIAN = ("uma", "aimnet", "mace")      # models with their own second derivatives


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

    def analytic_hessian(self, name, *, charge=0, spin=1):
        """A function atoms -> (3N, 3N) Hessian (eV/Angstrom^2) from the model itself, or None if it has none.

        UMA: a second predict unit with the untrained omol Hessian enabled
        (loaded on first use); AIMNet2: AIMNet2ASE.get_hessian; MACE-OMOL:
        MACECalculator.get_hessian. hessian() checks one column against finite
        differences, so a shape or sign convention change shows up there.
        """
        if name not in ANALYTIC_HESSIAN:
            return None
        if name == "uma":
            if "uma_hessian" not in self._loaded:
                from fairchem.core import FAIRChemCalculator, pretrained_mlip
                from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
                settings = InferenceSettings(predict_untrained_hessian={"omol"}, activation_checkpointing=False)
                unit = pretrained_mlip.load_predict_unit(os.environ["UMA_CHECKPOINT"], device="cpu",
                                                         inference_settings=settings)
                self._loaded["uma_hessian"] = FAIRChemCalculator(unit, task_name="omol")
            calculator = self._loaded["uma_hessian"]

            def uma(atoms):
                copy = atoms.copy()
                copy.info.update(charge=int(charge), spin=int(spin))
                copy.calc = calculator
                return np.asarray(copy.calc.get_property("hessian", copy))
            return uma
        calculator = self.calculator(name, charge=charge, spin=spin)

        def own(atoms):
            copy = atoms.copy()
            copy.info.update(charge=int(charge), spin=int(spin))
            copy.calc = calculator
            return np.asarray(calculator.get_hessian(copy))
        return own

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


def restraint_hessian(positions, pairs, targets, k):
    """Cartesian Hessian (3N x 3N, eV/Angstrom^2) of the harmonic distance restraints 0.5 k (d - d0)^2."""
    xyz = np.asarray(positions, dtype=float)
    matrix = np.zeros((3 * len(xyz), 3 * len(xyz)))
    for (i, j), target in zip(pairs, targets):
        vector = xyz[j] - xyz[i]
        d = np.linalg.norm(vector)
        u = vector / d
        block = k * ((1 - target / d) * np.eye(3) + (target / d) * np.outer(u, u))
        for a, b, sign in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
            matrix[3 * a:3 * a + 3, 3 * b:3 * b + 3] += sign * block
    return matrix


def optimize(symbols, positions, *, calculator, constraint=None, optimizer="BFGS", fmax=0.05, steps=500,
             charge=0, spin=1, callback=None, hessian_function=None, diag_every_n=None):
    """Optimize one structure; return (final Atoms, converged, number of steps).

    Sella only: hessian_function (atoms -> 3N x 3N model Hessian, eV/Angstrom^2)
    replaces Sella's iterative Hessian estimate: it is called at the start and
    whenever Sella would rediagonalize, and with diag_every_n also every that many
    steps. Restraint terms (restrain_distances) are added to it here.

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
                  "positions": np.round(atoms.positions, 6).tolist(),
                  # total forces (model + any restraint), before constraint projection
                  "forces": np.round(atoms.get_forces(apply_constraint=False), 6).tolist()})
        step["n"] += 1

    if sella:
        from sella import Sella
        options = {}
        if hessian_function is not None:
            model_hessian = hessian_function
            if mode == "restrain_distances":
                restraint = (calculator.pairs, calculator.targets, calculator.k)

                def model_hessian(a, base=hessian_function):
                    return base(a) + restraint_hessian(a.positions, *restraint)
            options.update(hessian_function=model_hessian, eig=True,
                           diag_every_n=int(diag_every_n) if diag_every_n else None)
        runner = Sella(atoms, order=1 if optimizer == "sella_ts" else 0, internal=True,
                       constraints=sella_constraints, logfile=None, **options)
    else:
        from ase import optimize as ase_optimize
        runner = getattr(ase_optimize, optimizer)(atoms, logfile=None)
    # Optimizers call their observers for the start geometry and after every step.
    runner.attach(report, interval=1)
    converged = bool(runner.run(fmax=float(fmax), steps=int(steps)))
    return atoms, converged, step["n"] - 1


def _atoms(symbols, positions, charge, spin, calculator):
    atoms = Atoms(list(symbols), positions=np.asarray(positions, dtype=float))
    atoms.info.update(charge=int(charge), spin=int(spin))
    atoms.calc = calculator
    return atoms


def single_point(symbols, positions, *, calculator, charge=0, spin=1):
    """{"energy" (eV), "forces" (eV/Angstrom, N x 3), "fmax"} of one structure."""
    atoms = _atoms(symbols, positions, charge, spin, calculator)
    forces = atoms.get_forces()
    return {"energy": float(atoms.get_potential_energy()), "forces": np.round(forces, 6).tolist(),
            "fmax": float(np.sqrt((forces ** 2).sum(axis=1)).max())}


def _force_column(atoms, k, step):
    """-dF/dx_k by central differences: column k of the Hessian (eV/Angstrom^2)."""
    i, axis = divmod(k, 3)
    columns = []
    for sign in (1., -1.):
        displaced = atoms.copy()
        displaced.calc = atoms.calc
        displaced.positions[i, axis] += sign * step
        columns.append(displaced.get_forces().ravel())
    return -(columns[0] - columns[1]) / (2 * step)


def hessian(symbols, positions, *, calculator, method="auto", analytic=None, step=0.005, charge=0, spin=1,
            callback=None):
    """(H, info): the Cartesian Hessian (3N x 3N, eV/Angstrom^2, symmetrized) of one structure.

    method "analytic" uses analytic (a function atoms -> Hessian, e.g.
    ModelCache.analytic_hessian), "finite_diff" central differences of the
    forces with step (Angstrom; 6N force calls), "auto" analytic when given and
    working, else finite differences. An analytic Hessian is checked against
    one finite-difference column (the coordinate with the largest diagonal
    element); a negated result is corrected, a mismatch is reported in
    info["check"]. callback(message) reports progress and may raise Cancelled.
    info: {"method" (used), "requested", "step", "check", "note", "energy", "forces", "fmax"}.
    """
    if method not in HESSIAN_METHODS:
        raise ValueError(f"method must be one of {HESSIAN_METHODS}")
    if method == "analytic" and analytic is None:
        raise ValueError("this model has no analytic Hessian; use finite_diff or auto")
    report = callback or (lambda message: None)
    atoms = _atoms(symbols, positions, charge, spin, calculator)
    n3 = 3 * len(atoms)
    info = {"requested": method, "step": float(step), "check": None, "note": ""}
    center = single_point(symbols, positions, calculator=calculator, charge=charge, spin=spin)
    info.update(energy=center["energy"], forces=center["forces"], fmax=center["fmax"])
    matrix = None
    if method in ("analytic", "auto") and analytic is not None:
        report("analytic Hessian")
        try:
            matrix = np.asarray(analytic(atoms), dtype=float).reshape(n3, n3)
            info["method"] = "analytic"
        except Exception as exc:          # auto falls back; analytic reports the failure
            if method == "analytic":
                raise
            info["note"] = f"analytic Hessian failed ({type(exc).__name__}: {exc}); used finite differences"
    if matrix is not None:
        k = int(np.argmax(np.abs(np.diag(matrix))))
        reference = _force_column(atoms, k, step)
        scale = np.linalg.norm(reference) or 1.0
        same, flipped = np.linalg.norm(matrix[:, k] - reference) / scale, np.linalg.norm(matrix[:, k] + reference) / scale
        if flipped < same:
            matrix, same = -matrix, flipped
            info["note"] = "analytic Hessian had the opposite sign convention; negated"
        info["check"] = float(same)
        if same > 0.05:
            info["note"] = (info["note"] + "; " if info["note"] else "") + \
                f"analytic column differs from finite differences by {100 * same:.1f}%"
    if matrix is None:
        info["method"] = "finite_diff"
        matrix = np.zeros((n3, n3))
        for k in range(n3):
            if k % 6 == 0:
                report(f"finite differences {k}/{n3} coordinates ({2 * k} force calls)")
            matrix[:, k] = _force_column(atoms, k, step)
    return 0.5 * (matrix + matrix.T), info


# sqrt(eV / (Angstrom^2 amu)) in cm^-1
_WAVENUMBER = float(np.sqrt(units._e / (units._amu * 1e-20)) / (2 * np.pi * units._c * 100))


def vibrations(symbols, positions, matrix, *, masses=None):
    """Harmonic analysis of a Cartesian Hessian (eV/Angstrom^2); translations and rotations projected out.

    Returns {"frequencies" (cm^-1, imaginary as negative, ascending),
    "reduced_masses" (amu), "force_constants" (mdyn/Angstrom; both as Gaussian
    reports them: mu = 1 / sum of the squared Cartesian components of the
    mass-weighted unit mode, k = eigenvalue x mu), "modes"
    (Cartesian displacements, N x 3 each, scaled so the largest atom moves 1),
    "masses"}: 3N - 6 modes (3N - 5 for a linear structure).
    """
    xyz = np.asarray(positions, dtype=float)
    n = len(xyz)
    m = np.asarray(masses if masses is not None else Atoms(list(symbols)).get_masses(), dtype=float)
    root = np.repeat(np.sqrt(m), 3)
    weighted = np.asarray(matrix, dtype=float) / np.outer(root, root)
    # translations and rotations about the centre of mass, mass-weighted
    centred = xyz - (m[:, None] * xyz).sum(axis=0) / m.sum()
    external = []
    for axis in range(3):
        t = np.zeros((n, 3))
        t[:, axis] = 1.
        external.append((t * np.sqrt(m)[:, None]).ravel())
        r = np.cross(np.eye(3)[axis], centred)
        external.append((r * np.sqrt(m)[:, None]).ravel())
    u, singular, _ = np.linalg.svd(np.array(external).T, full_matrices=True)
    rank = int((singular > 1e-6 * singular.max()).sum())
    basis = u[:, rank:]                                   # internal coordinates, orthonormal
    values, vectors = np.linalg.eigh(basis.T @ weighted @ basis)
    weighted_modes = basis @ vectors                       # unit vectors in mass-weighted space
    cartesian = weighted_modes / root[:, None]
    reduced = 1.0 / (cartesian ** 2).sum(axis=0)
    modes = []
    for j in range(cartesian.shape[1]):
        mode = cartesian[:, j].reshape(n, 3)
        mode = mode / np.linalg.norm(mode, axis=1).max()
        modes.append(np.round(mode, 5).tolist())
    return {"frequencies": (np.sign(values) * np.sqrt(np.abs(values)) * _WAVENUMBER).tolist(),
            "reduced_masses": reduced.tolist(),
            "force_constants": (values * reduced * units._e / 1e-20 / 1e2).tolist(),   # J/m^2 -> mdyn/Angstrom
            "modes": modes, "masses": m.tolist()}
