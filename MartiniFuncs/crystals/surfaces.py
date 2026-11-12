"""Surface energies of molecular-crystal slabs with an ASE calculator.

gamma = (E_slab - N_mol * e_bulk) / (2 A)

with e_bulk the energy per molecule of the RELAXED bulk computed with the same
calculator, N_mol the molecules in the slab and A the surface-cell area; the
slab must be cut from that relaxed bulk (build_slab on the crystal returned by
relax_bulk) so that its in-plane lattice is the calculator's. Both faces are
counted: for a non-centrosymmetric crystal (h k l) and (-h -k -l) differ and
gamma is their average. Results are in J/m^2 (1 eV/Angstrom^2 = 16.0218 J/m^2).

surface_energy        unrelaxed (cleavage) and relaxed gamma of one slab; the
                      relaxation fixes the n_fixed_layers central layers (default 1)
face_energies         gamma of several faces and terminations; the lowest
                      termination of each face is its surface energy
thickness_convergence gamma against slab thickness, plus the linear-fit
                      estimate (E_slab(n) = n E_layer + 2 gamma A), which does not
                      depend on e_bulk and so removes bulk-reference error
"""
from __future__ import annotations

import warnings

import numpy as np

from .bulk import relax_positions
from .calculators import resolve
from .cif import MolecularCrystal
from .slabs import Slab, build_slab, terminations

EV_A2_TO_J_M2 = 16.02176634


def _check_stoichiometry(slab: Slab, crystal: MolecularCrystal):
    bulk = crystal.composition
    n = slab.n_molecules / crystal.n_molecules
    if any(abs(slab.composition.get(f, 0) - n * c) > 1e-6 for f, c in bulk.items()):
        raise ValueError(f"slab composition {slab.composition} is not a multiple of the bulk {bulk}")


def surface_energy(slab: Slab, calculator, bulk_energy_per_molecule, *, relax=True, n_fixed_layers=1,
                   fmax=0.02, steps=500, optimizer="BFGS", crystal=None):
    """Unrelaxed and relaxed surface energy of one slab.

    Returns a dict with gamma_unrelaxed_J_m2, gamma_J_m2 (relaxed, or unrelaxed
    when relax=False), energies, and the relaxed Atoms under "atoms".
    """
    if crystal is not None:
        _check_stoichiometry(slab, crystal)
    calculator = resolve(calculator)
    atoms = slab.atoms.copy()
    atoms.calc = calculator
    reference = slab.n_molecules * bulk_energy_per_molecule
    e0 = float(atoms.get_potential_energy())
    area = slab.area
    out = {
        "hkl": tuple(slab.hkl), "termination": slab.termination.index, "layers": slab.layers,
        "n_molecules": slab.n_molecules, "area": area, "energy_unrelaxed": e0,
        "gamma_unrelaxed_J_m2": (e0 - reference) / (2 * area) * EV_A2_TO_J_M2,
        "broken_contacts_per_nm2": slab.termination.contacts_per_nm2,
    }
    if relax:
        fixed = slab.central_atoms(n_fixed_layers) if n_fixed_layers else None
        energy, converged, n = relax_positions(atoms, calculator, fmax=fmax, steps=steps, optimizer=optimizer,
                                               fixed=fixed)
        if not converged:
            warnings.warn(f"slab {slab.hkl} termination {slab.termination.index} not converged in {n} steps")
        out.update(energy=energy, converged=converged, steps=n,
                   gamma_J_m2=(energy - reference) / (2 * area) * EV_A2_TO_J_M2)
    else:
        out.update(energy=e0, converged=None, steps=0, gamma_J_m2=out["gamma_unrelaxed_J_m2"])
    out["gamma_mJ_m2"] = out["gamma_J_m2"] * 1000
    out["atoms"] = atoms
    return out


def face_energies(crystal: MolecularCrystal, calculator, bulk_energy_per_molecule, faces, *, n_terminations=1,
                  layers=None, min_thickness=12.0, vacuum=15.0, min_lengths=None, max_atoms=None, relax=True,
                  n_fixed_layers=1, fmax=0.02, steps=500, on_result=None, slab_options=None):
    """Surface energies of each face in faces for its n_terminations best-ranked terminations.

    Returns (per_face, all_results): per_face maps hkl to the lowest-energy
    result for that face; all_results lists every slab computed. on_result(result)
    is called after each slab (e.g. to save progress).
    """
    calculator = resolve(calculator)
    results, best = [], {}
    for hkl in faces:
        terms = terminations(crystal, hkl)[:n_terminations]
        for term in terms:
            slab = build_slab(crystal, hkl, layers=layers, min_thickness=min_thickness, termination=term,
                              vacuum=vacuum, min_lengths=min_lengths, max_atoms=max_atoms, **(slab_options or {}))
            result = surface_energy(slab, calculator, bulk_energy_per_molecule, relax=relax,
                                    n_fixed_layers=n_fixed_layers, fmax=fmax, steps=steps, crystal=crystal)
            result["slab"] = slab
            results.append(result)
            if on_result:
                on_result(result)
            key = tuple(slab.hkl)
            if key not in best or result["gamma_J_m2"] < best[key]["gamma_J_m2"]:
                best[key] = result
    return best, results


def thickness_convergence(crystal: MolecularCrystal, calculator, hkl, bulk_energy_per_molecule, *,
                          layers=(2, 3, 4, 5), termination=0, relax=False, **options):
    """gamma vs number of layers, and the linear-fit gamma from E_slab(n)."""
    calculator = resolve(calculator)
    rows = []
    for n in layers:
        slab = build_slab(crystal, hkl, layers=n, termination=termination,
                          **{k: v for k, v in options.items() if k in ("vacuum", "min_lengths", "supercell")})
        r = surface_energy(slab, calculator, bulk_energy_per_molecule, relax=relax,
                           **{k: v for k, v in options.items() if k in ("n_fixed_layers", "fmax", "steps")})
        rows.append({"layers": n, "thickness": slab.thickness, "gamma_J_m2": r["gamma_J_m2"],
                     "energy": r["energy"], "area": r["area"]})
    n = np.array([r["layers"] for r in rows], float)
    e = np.array([r["energy"] for r in rows])
    slope, intercept = np.polyfit(n, e, 1)
    area = rows[0]["area"]
    return {"rows": rows, "gamma_linear_fit_J_m2": float(intercept / (2 * area) * EV_A2_TO_J_M2),
            "energy_per_layer_fit": float(slope)}
