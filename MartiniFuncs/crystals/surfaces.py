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
molecule_removal_energy  energy to take one surface molecule out of a slab
                      (a terrace vacancy) into the gas phase
surface_sites         the distinct exposed molecules of a face (different
                      positions or heights in the surface cell)
site_removal_energies removal energies of every one of those sites
attachment_energies   slice and attachment energies per face and termination
                      (Hartman-Perdok): E_latt = E_slice + E_att, from one-layer slabs

Attachment energy: a d_hkl-thick growth slice bounded by two cuts of the same
termination has energy e_slice per molecule (a one-layer slab, unrelaxed). Then

    E_att = e_bulk - e_slice   (per molecule, negative)

is the energy released when that slice attaches to the crystal face, and
E_slice = E_latt - E_att the binding inside the slice. In the attachment-energy
growth model a face grows at a rate proportional to |E_att|, and each face grows
by the termination whose slice is most strongly bound (smallest |E_att|).
(e_slice - e_bulk) N / 2A is the cleavage surface energy the slice model
implies (only the bonds between neighbouring slices broken).

Removal energies (eV and kJ/mol, positive = costs energy):

    vertical   = E(slab - m) + E(m frozen) - E(slab)    nothing relaxes: the binding of
                 that molecule to the cleaved surface (its intermolecular interaction energy)
    vertical_relaxed  the same, taken on the relaxed slab: the binding once the surface
                 has reorganised (larger than `vertical` when relaxation draws the surface
                 molecules into stronger contacts, e.g. reoriented carbonyls)
    adiabatic  = E(slab - m, relaxed) + E_gas - E(slab, relaxed)   the desorption energy
                 to an isolated, relaxed molecule
    deformation = E(m frozen) - E_gas    strain the crystal imposes on the molecule

with the same central layer(s) fixed in both slab relaxations. In a pair-additive
picture |E_latt| is half of one molecule's total binding (each pair is shared),
so removing a bulk molecule costs 2|E_latt|, a kink-site molecule (half its
neighbours) exactly |E_latt|, and a molecule in a flat terrace, which keeps
more than half its neighbours, between |E_latt| and 2|E_latt|. The ratio to
|E_latt| is reported when the lattice energy is given: a terrace value near 1
means the face is weakly bound, near 2 that the surface molecule is held almost
as tightly as in the bulk. The in-plane cell must be large
enough that the vacancy's periodic images do not interact: build the slab with
min_lengths of at least ~15 Angstrom.
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


KJ_MOL_PER_EV = 96.48533212


def _fixed_layer_atoms(atoms, layers, n_fixed):
    """Indices of atoms in the n_fixed central layers (by the slab's "layer" array)."""
    if not n_fixed:
        return None
    start = (layers - n_fixed) // 2
    return np.flatnonzero(np.isin(atoms.arrays["layer"], list(range(start, start + n_fixed))))


def surface_molecule(slab: Slab, side="top"):
    """mol_id of the outermost molecule (highest COM on the top face, lowest on the bottom)."""
    atoms = slab.atoms
    mol, masses = atoms.arrays["mol_id"], atoms.get_masses()
    ids = np.unique(mol)
    z = np.array([masses[mol == m] @ atoms.positions[mol == m, 2] / masses[mol == m].sum() for m in ids])
    return int(ids[np.argmax(z)] if side == "top" else ids[np.argmin(z)])


def molecule_removal_energy(slab: Slab, calculator, gas_energy, *, molecule=None, side="top", relax=True,
                            n_fixed_layers=1, fmax=0.02, steps=600, optimizer="BFGS", lattice_energy=None,
                            min_image_distance=12.0):
    """Energy to remove one surface molecule from slab (see the module docstring).

    gas_energy: energy (eV) of the relaxed isolated molecule from the same
    calculator, or {formula: energy} (results.json "gas") to pick the removed
    molecule's species. molecule: mol_id to remove (default:
    the outermost one on `side`). lattice_energy: E_latt per molecule (eV,
    negative) to report the removal energies as a fraction of |E_latt|.
    Returns a dict of energies plus "slab_atoms" (the intact slab, relaxed when relax=True)
    and "vacancy_atoms" (the slab without the molecule, relaxed when relax=True).
    """
    calculator = resolve(calculator)
    in_plane = np.linalg.norm(slab.atoms.cell.array[:2], axis=1)
    if in_plane.min() < min_image_distance:
        warnings.warn(f"in-plane cell vectors {np.round(in_plane, 1).tolist()} Angstrom: the vacancy is closer than "
                      f"{min_image_distance} Angstrom to its periodic images; build the slab with larger min_lengths")
    m = surface_molecule(slab, side) if molecule is None else int(molecule)
    atoms = slab.atoms.copy()
    gone = atoms.arrays["mol_id"] == m
    if not gone.any():
        raise ValueError(f"no molecule {m} in the slab")
    atoms.calc = calculator
    e_slab = float(atoms.get_potential_energy())

    def isolated(a):
        mol = a[gone].copy()
        for name in ("mol_id", "layer"):
            mol.arrays.pop(name, None)
        mol.set_cell(None)
        mol.pbc = False
        return mol

    frozen = isolated(atoms)
    if isinstance(gas_energy, dict):
        formula = frozen.get_chemical_formula(mode="hill")
        if formula not in gas_energy:
            raise ValueError(f"no gas-phase energy for {formula}; have {sorted(gas_energy)}")
        gas_energy = gas_energy[formula]
    frozen.calc = calculator
    e_frozen = float(frozen.get_potential_energy())
    vacancy = atoms[~gone]
    vacancy.calc = calculator
    e_vacancy = float(vacancy.get_potential_energy())
    vertical = e_vacancy + e_frozen - e_slab
    out = {
        "hkl": tuple(slab.hkl), "termination": slab.termination.index, "side": side, "molecule": m,
        "formula": frozen.get_chemical_formula(mode="hill"), "n_atoms_slab": len(atoms), "layers": slab.layers,
        "in_plane_lengths": in_plane.tolist(), "gas_energy": float(gas_energy),
        "vertical_eV": vertical, "vertical_kJ_mol": vertical * KJ_MOL_PER_EV,
        "deformation_eV": e_frozen - gas_energy, "deformation_kJ_mol": (e_frozen - gas_energy) * KJ_MOL_PER_EV,
    }
    result_atoms = vacancy
    if relax:
        relaxed = atoms.copy()
        e_slab_rel, conv_slab, n_slab = relax_positions(relaxed, calculator, fmax=fmax, steps=steps, optimizer=optimizer,
                                                        fixed=_fixed_layer_atoms(relaxed, slab.layers, n_fixed_layers))
        # vertical removal from the RELAXED surface: the molecule's binding once the surface has reorganised
        frozen_rel = isolated(relaxed)
        frozen_rel.calc = calculator
        vac_rel = relaxed[~gone]
        vac_rel.calc = calculator
        vertical_rel = float(vac_rel.get_potential_energy()) + float(frozen_rel.get_potential_energy()) - e_slab_rel
        e_vac_rel, conv_vac, n_vac = relax_positions(vac_rel, calculator, fmax=fmax, steps=steps, optimizer=optimizer,
                                                     fixed=_fixed_layer_atoms(vac_rel, slab.layers, n_fixed_layers))
        adiabatic = e_vac_rel + gas_energy - e_slab_rel
        out.update(vertical_relaxed_eV=vertical_rel, vertical_relaxed_kJ_mol=vertical_rel * KJ_MOL_PER_EV,
                   adiabatic_eV=adiabatic, adiabatic_kJ_mol=adiabatic * KJ_MOL_PER_EV,
                   converged=bool(conv_slab and conv_vac), steps_slab=n_slab, steps_vacancy=n_vac)
        if not (conv_slab and conv_vac):
            warnings.warn(f"removal-energy relaxations for {slab.hkl} not converged in {steps} steps")
        result_atoms = vac_rel
        relaxed.calc = None
        out["slab_atoms"] = relaxed
    else:
        out["slab_atoms"] = atoms
    if lattice_energy:
        out["vertical_over_lattice"] = vertical / abs(lattice_energy)
        if relax:
            out["vertical_relaxed_over_lattice"] = out["vertical_relaxed_eV"] / abs(lattice_energy)
            out["adiabatic_over_lattice"] = out["adiabatic_eV"] / abs(lattice_energy)
    out["vacancy_atoms"] = result_atoms
    return out


def surface_sites(slab: Slab, side="top", depth=3.0, tol=0.3):
    """Distinct exposed molecules of a slab face.

    A molecule is exposed when one of its atoms lies within `depth` Angstrom of
    the outermost atom on `side`. Exposed molecules that are copies of each
    other under the primitive surface lattice (the slab may be an in-plane
    supercell) form one site; molecules related only by a point operation
    (a 2-fold axis or a glide) stay separate sites, and their removal energies
    then come out equal. Returns a list of dicts, most exposed first: site
    index, representative mol_id (the copy nearest the middle of the cell), all
    member mol_ids, depth below the outermost atom of the molecule's outermost
    atom and of its centre of mass, and its fractional position in the
    primitive surface cell.
    """
    atoms = slab.atoms
    mol, masses, pos = atoms.arrays["mol_id"], atoms.get_masses(), atoms.positions
    sign = 1.0 if side == "top" else -1.0
    outer = (pos[:, 2] * sign).max()
    S = np.asarray(slab.inplane, dtype=float)
    prim = np.linalg.inv(S) @ atoms.cell.array[:2, :2]           # primitive in-plane vectors, slab frame
    inv_prim = np.linalg.inv(prim)
    inv_cell = np.linalg.inv(atoms.cell.array[:2, :2])

    def com_of(m):
        idx = mol == m
        return masses[idx] @ pos[idx] / masses[idx].sum()

    sites = []
    for m in np.unique(mol):
        extreme = (pos[mol == m, 2] * sign).max()
        if outer - extreme > depth:
            continue
        com = com_of(m)
        frac = com[:2] @ inv_prim
        frac -= np.floor(frac + 1e-9)
        com_depth = outer - sign * com[2]
        for s in sites:
            d = frac - s["frac"]
            d -= np.round(d)
            if np.linalg.norm(d @ prim) < tol and abs(com_depth - s["com_depth"]) < tol:
                s["members"].append(int(m))
                break
        else:
            sites.append({"frac": frac, "com_depth": float(com_depth), "atom_depth": float(outer - extreme),
                          "members": [int(m)]})
    for s in sites:
        s["molecule"] = min(s["members"], key=lambda mm: np.linalg.norm(com_of(mm)[:2] @ inv_cell - 0.5))
        s["frac"] = [float(x) for x in s["frac"]]
    sites.sort(key=lambda s: (round(s["atom_depth"], 2), s["com_depth"]))
    for k, s in enumerate(sites):
        s["site"] = k
    return sites


def site_removal_energies(slab: Slab, calculator, gas_energy, *, side="top", depth=3.0, relax=True,
                          n_fixed_layers=1, fmax=0.02, steps=600, optimizer="BFGS", lattice_energy=None,
                          min_image_distance=12.0, sites=None):
    """Removal energies (as in molecule_removal_energy) of every distinct exposed molecule (surface_sites).

    The intact slab is evaluated, and relaxed, once; each site then costs two
    single points and one vacancy relaxation. gas_energy: eV or {formula: eV}.
    Returns {"sites": [rows], "slab_atoms": intact (relaxed) slab, "vacancies":
    {site: vacancy Atoms}, ...}; each row holds the site description and its
    vertical, vertical_relaxed and adiabatic energies (eV, kJ/mol, / |E_latt|).
    """
    calculator = resolve(calculator)
    in_plane = np.linalg.norm(slab.atoms.cell.array[:2], axis=1)
    if in_plane.min() < min_image_distance:
        warnings.warn(f"in-plane cell vectors {np.round(in_plane, 1).tolist()} Angstrom: each vacancy is closer than "
                      f"{min_image_distance} Angstrom to its periodic images; build the slab with larger min_lengths")
    sites = surface_sites(slab, side, depth) if sites is None else sites
    atoms = slab.atoms.copy()
    atoms.calc = calculator
    e_slab = float(atoms.get_potential_energy())
    relaxed, e_slab_rel, conv_slab, n_slab = atoms, e_slab, True, 0
    if relax:
        relaxed = atoms.copy()
        e_slab_rel, conv_slab, n_slab = relax_positions(relaxed, calculator, fmax=fmax, steps=steps,
                                                        optimizer=optimizer,
                                                        fixed=_fixed_layer_atoms(relaxed, slab.layers, n_fixed_layers))

    def isolated(a, gone):
        part = a[gone].copy()
        for name in ("mol_id", "layer"):
            part.arrays.pop(name, None)
        part.set_cell(None)
        part.pbc = False
        part.calc = calculator
        return part

    def single(a):
        a = a.copy()
        a.calc = calculator
        return float(a.get_potential_energy())

    rows, vacancies = [], {}
    for s in sites:
        gone = atoms.arrays["mol_id"] == s["molecule"]
        frozen = isolated(atoms, gone)
        formula = frozen.get_chemical_formula(mode="hill")
        e_gas = gas_energy[formula] if isinstance(gas_energy, dict) else float(gas_energy)
        e_frozen = float(frozen.get_potential_energy())
        row = dict(s, hkl=tuple(slab.hkl), termination=slab.termination.index, side=side, formula=formula,
                   multiplicity=len(s["members"]), n_atoms_slab=len(atoms), layers=slab.layers,
                   in_plane_lengths=in_plane.tolist(), gas_energy=e_gas,
                   vertical_eV=single(atoms[~gone]) + e_frozen - e_slab, deformation_eV=e_frozen - e_gas)
        vacancy = atoms[~gone]
        if relax:
            row["vertical_relaxed_eV"] = (single(relaxed[~gone]) + float(isolated(relaxed, gone).get_potential_energy())
                                          - e_slab_rel)
            vacancy = relaxed[~gone]
            e_vac, conv_vac, n_vac = relax_positions(vacancy, calculator, fmax=fmax, steps=steps, optimizer=optimizer,
                                                     fixed=_fixed_layer_atoms(vacancy, slab.layers, n_fixed_layers))
            vacancy.calc = None
            row.update(adiabatic_eV=e_vac + e_gas - e_slab_rel, converged=bool(conv_slab and conv_vac),
                       steps_slab=n_slab, steps_vacancy=n_vac)
            if not (conv_slab and conv_vac):
                warnings.warn(f"site {s['site']} of {slab.hkl}: relaxations not converged in {steps} steps")
        for key in ("vertical", "vertical_relaxed", "adiabatic", "deformation"):
            if f"{key}_eV" in row:
                row[f"{key}_kJ_mol"] = row[f"{key}_eV"] * KJ_MOL_PER_EV
                if lattice_energy and key != "deformation":
                    row[f"{key}_over_lattice"] = row[f"{key}_eV"] / abs(lattice_energy)
        rows.append(row)
        vacancies[s["site"]] = vacancy
    relaxed = relaxed.copy()
    relaxed.calc = None
    return {"sites": rows, "slab_atoms": relaxed, "vacancies": vacancies,
            "slab_energy": e_slab, "slab_energy_relaxed": e_slab_rel}


def attachment_energies(crystal: MolecularCrystal, calculator, bulk_energy_per_molecule, faces, *,
                        terminations_per_face=None, lattice_energy=None, vacuum=15.0, max_terminations=None):
    """Slice and attachment energies of each face and termination (Hartman-Perdok).

    crystal must be the relaxed bulk and bulk_energy_per_molecule its energy per
    molecule with the same calculator. faces: iterable of (h, k, l).
    terminations_per_face: optional {hkl: [indices]} (default: every termination,
    or the first max_terminations). lattice_energy (eV per molecule, negative)
    adds the slice energy E_slice = E_latt - E_att and the fraction E_att/E_latt.

    Each slice is one d_hkl layer cut at the termination, as a slab with vacuum,
    evaluated without relaxation. Returns a list of dicts with hkl, termination,
    d_hkl, n_molecules, area, e_slice (eV/molecule), attachment_energy_eV and
    _kJ_mol, gamma_slice_J_m2 and, with lattice_energy, slice_energy_eV/_kJ_mol
    and attachment_fraction.
    """
    calculator = resolve(calculator)
    rows = []
    for hkl in faces:
        hkl = tuple(int(x) for x in hkl)
        terms = terminations(crystal, hkl)
        wanted = (terminations_per_face or {}).get(hkl)
        chosen = [terms[i] for i in wanted] if wanted is not None else terms[:max_terminations]
        for term in chosen:
            piece = build_slab(crystal, hkl, layers=1, termination=term, vacuum=vacuum)
            _check_stoichiometry(piece, crystal)
            atoms = piece.atoms.copy()
            atoms.calc = calculator
            e_slice = float(atoms.get_potential_energy()) / piece.n_molecules
            e_att = bulk_energy_per_molecule - e_slice
            row = {"hkl": hkl, "termination": term.index, "d_hkl": piece.d_spacing, "n_molecules": piece.n_molecules,
                   "area": piece.area, "broken_contacts_per_nm2": term.contacts_per_nm2, "e_slice": e_slice,
                   "attachment_energy_eV": e_att, "attachment_energy_kJ_mol": e_att * KJ_MOL_PER_EV,
                   "gamma_slice_J_m2": -e_att * piece.n_molecules / (2 * piece.area) * EV_A2_TO_J_M2}
            if lattice_energy is not None:
                row.update(slice_energy_eV=lattice_energy - e_att,
                           slice_energy_kJ_mol=(lattice_energy - e_att) * KJ_MOL_PER_EV,
                           attachment_fraction=e_att / lattice_energy)
            rows.append(row)
    return rows
