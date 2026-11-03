"""The three recompute tasks; each maps one exported structure to a RecomputeResult.

All energies are eV and forces eV/Angstrom. F_MD is the per-atom force array
written by extract (lammps_forces): the total force on each atom in the
periodic MD frame, at the exported geometry.
"""
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from MartiniFuncs.recompute.fragments import Fragments

MD_FORCE_ARRAY = "lammps_forces"


@dataclass(frozen=True)
class RecomputeResult:
    """structure: the input structure with the task's arrays and info added; extras: further structures."""
    structure: Atoms
    extras: tuple = ()


def _annotated(structure, task, **info):
    """A copy of the structure (constraints and calculator dropped) with the task recorded in info."""
    out = structure.copy()
    out.set_constraint()
    out.calc = None
    out.info.update(recompute_task=task, energy_units="eV", force_units="eV/Angstrom", **info)
    return out


def _md_forces(structure):
    if MD_FORCE_ARRAY not in structure.arrays:
        raise ValueError(f"Structure has no {MD_FORCE_ARRAY} array; F_env needs the exported MD forces")
    if structure.info.get("force_units", "eV/Angstrom") != "eV/Angstrom":
        raise ValueError("MD forces must be in eV/Angstrom (export from a LammpsFrame, not legacy raw tuples)")
    return np.asarray(structure.arrays[MD_FORCE_ARRAY], dtype=float)


class ForceDecomposition:
    """Single points at the MD geometry: F_isolated, F_env = F_MD - F_isolated, and energies.

    Default (molecules=False): the pair is evaluated as one isolated fragment, so
    its F_env is the force from everything beyond the pair. Each shell molecule
    (clusters) is evaluated alone. Adds per-atom arrays isolated_forces,
    environment_forces and isolated_fragment (0 = pair, k = shell molecule k-1),
    and info energy_pair (plus energy_shell, one value per shell molecule).

    molecules=True also evaluates each pair member alone and adds
    molecule_isolated_forces / molecule_environment_forces (shell rows as above),
    info energy_<role label> (one per pair member) and
    interaction_energy = energy_pair - energy_<first> - energy_<second>.
    """

    def __init__(self, evaluator, *, molecules=False, role_labels=None, scale=1.3):
        self.evaluator = evaluator
        self.molecules = bool(molecules)
        self.role_labels = role_labels
        self.scale = scale

    def apply(self, structure):
        fragments = Fragments.from_structure(structure, role_labels=self.role_labels, scale=self.scale)
        md = _md_forces(structure)
        out = _annotated(structure, "single_point", recompute_fragments="pair+molecules" if self.molecules else "pair",
                         recompute_charge=self.evaluator.charge, recompute_spin=self.evaluator.spin)
        isolated = np.full(md.shape, np.nan)
        fragment_index = np.full(len(structure), -1, dtype=int)
        out.info["energy_pair"], isolated[fragments.pair] = self.evaluator.evaluate(structure, fragments.pair)
        fragment_index[fragments.pair] = 0
        shell_energies = []
        for k, (_, rows) in enumerate(fragments.shell_molecules(), start=1):
            energy, isolated[rows] = self.evaluator.evaluate(structure, rows)
            fragment_index[rows] = k
            shell_energies.append(energy)
        if shell_energies:
            out.info["energy_shell"] = np.asarray(shell_energies)
        out.new_array("isolated_forces", isolated)
        out.new_array("environment_forces", md - isolated)
        out.new_array("isolated_fragment", fragment_index)
        if self.molecules:
            per_molecule = isolated.copy()
            energies = {}
            for label, rows in fragments.members():
                energies[label], per_molecule[rows] = self.evaluator.evaluate(structure, rows)
                out.info[f"energy_{label}"] = energies[label]
            out.info["interaction_energy"] = out.info["energy_pair"] - sum(energies.values())
            out.new_array("molecule_isolated_forces", per_molecule)
            out.new_array("molecule_environment_forces", md - per_molecule)
        return RecomputeResult(out)


class FragmentRelaxation:
    """Relax the isolated pair (and with molecules=True each member alone) from the MD geometry.

    Adds info, per fragment label L (pair, and the role labels): energy_L at the
    MD geometry, relaxed_energy_L, deformation_energy_L = energy_L -
    relaxed_energy_L (>= 0 when the relaxation lowers the energy),
    relaxation_converged_L and relaxation_steps_L. Each relaxed fragment is also
    returned as an extra structure (relaxed positions, md_positions,
    global_index/pair_role of its atoms). Shell molecules are not relaxed.
    """

    def __init__(self, relaxer, *, molecules=False, role_labels=None, scale=1.3):
        self.relaxer = relaxer
        self.molecules = bool(molecules)
        self.role_labels = role_labels
        self.scale = scale

    def apply(self, structure):
        fragments = Fragments.from_structure(structure, role_labels=self.role_labels, scale=self.scale)
        out = _annotated(structure, "relax_fragments", recompute_fragments="pair+molecules" if self.molecules else "pair",
                         optimizer=self.relaxer.optimizer, fmax=self.relaxer.fmax)
        targets = [("pair", fragments.pair)] + (fragments.members() if self.molecules else [])
        extras = []
        for label, rows in targets:
            result = self.relaxer.relax(structure, rows)
            out.info.update({f"energy_{label}": result.initial_energy, f"relaxed_energy_{label}": result.final_energy,
                             f"deformation_energy_{label}": result.initial_energy - result.final_energy,
                             f"relaxation_converged_{label}": result.converged,
                             f"relaxation_steps_{label}": result.steps})
            relaxed = Atoms(np.asarray(structure.get_chemical_symbols())[rows].tolist(), positions=result.positions)
            relaxed.new_array("md_positions", structure.positions[rows].copy())
            for name in ("global_index", "pair_role"):
                if name in structure.arrays:
                    relaxed.new_array(name, np.asarray(structure.arrays[name])[rows].copy())
            relaxed.new_array("relaxed_forces", result.forces)
            relaxed.info.update(fragment=label, energy_md_geometry=result.initial_energy,
                                relaxed_energy=result.final_energy, converged=result.converged, steps=result.steps,
                                **{key: structure.info[key] for key in ("structure_index", "frame_index", "time_ps")
                                   if key in structure.info})
            extras.append(relaxed)
        return RecomputeResult(out, tuple(extras))


class FrozenShellOptimization:
    """Optimize the pair with every shell atom fixed (clusters.extxyz only).

    The whole cluster is one isolated fragment. Returns the cluster at the
    optimized geometry with md_positions (the exported geometry), frozen (shell
    mask), optimized_forces (unconstrained calculator forces at the optimized
    geometry) and info energy_md_geometry, energy_optimized,
    optimization_converged, optimization_steps. The exported per-atom MD arrays
    (lammps_forces, velocities) are kept unchanged and refer to md_positions.
    """

    def __init__(self, relaxer, *, role_labels=None, scale=1.3):
        self.relaxer = relaxer
        self.role_labels = role_labels
        self.scale = scale

    def apply(self, structure):
        fragments = Fragments.from_structure(structure, role_labels=self.role_labels, scale=self.scale)
        if not fragments.shell:
            raise ValueError("Frozen-shell optimization needs a cluster with shell atoms (clusters.extxyz)")
        frozen = np.zeros(len(structure), dtype=bool)
        frozen[np.concatenate(fragments.shell)] = True
        result = self.relaxer.relax(structure, np.arange(len(structure)), fixed=np.flatnonzero(frozen))
        out = _annotated(structure, "frozen_shell_optimization", optimizer=self.relaxer.optimizer,
                         fmax=self.relaxer.fmax, md_arrays_geometry="md_positions",
                         energy_md_geometry=result.initial_energy, energy_optimized=result.final_energy,
                         optimization_converged=result.converged, optimization_steps=result.steps)
        out.new_array("md_positions", structure.positions.copy())
        out.positions = result.positions
        out.new_array("frozen", frozen)
        out.new_array("optimized_forces", result.forces)
        return RecomputeResult(out)
