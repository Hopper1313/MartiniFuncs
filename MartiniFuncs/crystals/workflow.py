"""CIF -> relaxed crystal -> bulk properties -> surface energies -> favoured faces.

run_crystal_workflow(cif, calculator, out_dir) does, in order:

1. read the CIF (read_cif) and record space group, Z, Z', density, disorder
2. relax each molecular species in the gas phase (reference for the lattice energy)
3. relax the bulk (cell + positions, space group kept) and compare with experiment
4. lattice energy per molecule and the sublimation-enthalpy estimate
5. optional: equation of state (bulk modulus) and elastic tensor
6. candidate faces: the given list, else the top n_faces BFDH faces up to max_index
7. surface energy of each face (n_terminations best-ranked terminations each),
   slabs cut from the relaxed bulk
8. Wulff shape from the surface energies: area fraction of each face; faces are
   ranked by surface energy and by area fraction

Everything is written to out_dir: results.json (updated after every step, so
a killed run keeps what it finished), the relaxed bulk (bulk_relaxed.cif /
.extxyz), gas molecules and every slab (relaxed) as Extended XYZ.

Command line:
    python -m MartiniFuncs.crystals.workflow ANTCEN.cif --calculator uma --out antcen_uma
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from ase.io import write

from .._checks import plain
from .bulk import elastic_constants, equation_of_state, gas_references, lattice_energy, relax_bulk
from .calculators import CALCULATORS, describe, resolve
from .cif import read_cif, write_cif
from .morphology import bfdh, wulff
from .slabs import primitive_miller
from .surfaces import face_energies


def _save(path, results):
    path.write_text(json.dumps(plain(results, str), indent=2))


def run_crystal_workflow(cif, calculator, out_dir, *, faces=None, max_index=2, n_faces=6, n_terminations=2,
                         min_thickness=12.0, layers=None, vacuum=15.0, min_lengths=10.0, max_atoms=None,
                         bulk_fmax=0.01, slab_fmax=0.02, relax_slabs=True, n_fixed_layers=1, symmetrize=True,
                         compute_eos=True, compute_elastic=False, gas_box=None, read_options=None, log=print):
    """Run the full workflow; returns the results dict (also in out_dir/results.json)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    calc = resolve(calculator)
    results = {"cif": str(cif), "calculator": describe(calc), "settings": {
        "max_index": max_index, "n_faces": n_faces, "n_terminations": n_terminations, "min_thickness": min_thickness,
        "layers": layers, "vacuum": vacuum, "min_lengths": min_lengths, "bulk_fmax": bulk_fmax,
        "slab_fmax": slab_fmax, "relax_slabs": relax_slabs, "n_fixed_layers": n_fixed_layers}}
    path = out / "results.json"
    t0 = time.time()

    crystal = read_cif(cif, **(read_options or {}))
    results["structure"] = crystal.summary()
    results["structure"]["source"] = crystal.source
    _save(path, results)
    log(f"[read] {crystal.name}: {results['structure']['formula']}, Z={crystal.n_molecules}, "
        f"{results['structure']['space_group']}")

    gas = gas_references(crystal, calc, box=gas_box)
    for formula, (mol, energy) in gas.items():
        write(out / f"gas_{formula}.extxyz", mol)
    results["gas"] = {f: e for f, (_, e) in gas.items()}
    _save(path, results)
    log(f"[gas] {results['gas']}")

    relaxed, bulk = relax_bulk(crystal, calc, fmax=bulk_fmax, symmetrize=symmetrize)
    write_cif(relaxed, out / "bulk_relaxed.cif")
    write(out / "bulk_relaxed.extxyz", relaxed.atoms)
    results["bulk"] = bulk
    results["lattice_energy"] = lattice_energy(relaxed, bulk["energy"], results["gas"])
    _save(path, results)
    log(f"[bulk] volume change {bulk['volume_change_pct']:.2f}%, lattice energy "
        f"{results['lattice_energy']['lattice_energy_kJ_mol']:.1f} kJ/mol")

    if compute_eos:
        results["eos"] = equation_of_state(relaxed, calc)
        _save(path, results)
        log(f"[eos] B0 = {results['eos']['B0_GPa']:.2f} GPa")
    if compute_elastic:
        results["elastic"] = elastic_constants(relaxed, calc)
        _save(path, results)
        log(f"[elastic] K_H = {results['elastic']['bulk_modulus_hill']:.2f} GPa, "
            f"G_H = {results['elastic']['shear_modulus_hill']:.2f} GPa")

    ranking = bfdh(relaxed, max_index)
    results["bfdh"] = ranking
    faces = [primitive_miller(h) for h in faces] if faces else [f["hkl"] for f in ranking[:n_faces]]
    results["faces"] = faces
    _save(path, results)
    log(f"[faces] {faces}")

    e_bulk = bulk["energy_per_molecule"]
    surface_rows = []

    def record(result):
        slab = result["slab"]
        name = "slab_{}_t{}".format("".join(str(x) for x in slab.hkl).replace("-", "m"), slab.termination.index)
        write(out / f"{name}.extxyz", result["atoms"])
        surface_rows.append({k: v for k, v in result.items() if k not in ("atoms", "slab")} | {"file": f"{name}.extxyz"})
        results["surfaces"] = surface_rows
        _save(path, results)
        log(f"[surface] {slab.hkl} termination {slab.termination.index}: {result['gamma_mJ_m2']:.1f} mJ/m^2 "
            f"({slab.n_molecules} molecules)")

    best, _ = face_energies(relaxed, calc, e_bulk, faces, n_terminations=n_terminations, layers=layers,
                            min_thickness=min_thickness, vacuum=vacuum, min_lengths=min_lengths,
                            max_atoms=max_atoms, relax=relax_slabs, n_fixed_layers=n_fixed_layers,
                            fmax=slab_fmax, on_result=record)
    energies = {h: r["gamma_J_m2"] for h, r in best.items()}
    try:
        shape = wulff(relaxed, energies)
    except ValueError as error:                        # too few face families to close the shape
        shape = None
        results["wulff"] = {"error": str(error)}
        log(f"[wulff] {error}")
    else:
        results["wulff"] = {k: v for k, v in shape.items() if k != "shape"}
    fractions = shape["area_fractions"] if shape else {}
    results["face_ranking"] = sorted(
        ({"hkl": h, "gamma_J_m2": energies[h], "termination": best[h]["termination"],
          "area_fraction": fractions.get(h)} for h in energies),
        key=lambda r: r["gamma_J_m2"])
    results["elapsed_s"] = time.time() - t0
    _save(path, results)
    if shape:
        try:
            import matplotlib
            matplotlib.use("Agg")
            shape["shape"].get_plot().figure.savefig(out / "wulff.png", dpi=150)
        except Exception as error:                      # plotting is optional
            log(f"[wulff] plot skipped: {error}")
        log("[wulff] " + ", ".join(f"{h}: {a:.2f}" for h, a in fractions.items() if a > 0.005))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cif")
    parser.add_argument("--calculator", default="uma", choices=CALCULATORS)
    parser.add_argument("--model", default=None, help="UMA checkpoint/name or MACE model/size")
    parser.add_argument("--task", default="omc", help="UMA task head (default omc)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    parser.add_argument("--faces", nargs="*", default=None, help="e.g. 001 110 1-10 (digits; '-' before negatives)")
    parser.add_argument("--max-index", type=int, default=2)
    parser.add_argument("--n-faces", type=int, default=6)
    parser.add_argument("--n-terminations", type=int, default=2)
    parser.add_argument("--min-thickness", type=float, default=12.0)
    parser.add_argument("--min-lengths", type=float, default=10.0)
    parser.add_argument("--max-atoms", type=int, default=None)
    parser.add_argument("--vacuum", type=float, default=15.0)
    parser.add_argument("--no-slab-relax", action="store_true")
    parser.add_argument("--no-eos", action="store_true", help="skip the equation of state")
    parser.add_argument("--elastic", action="store_true", help="compute the elastic tensor")
    parser.add_argument("--ncpus", type=int, default=None)
    args = parser.parse_args(argv)
    from .calculators import crystal_calculator
    calc = crystal_calculator(args.calculator, model=args.model, task=args.task, device=args.device,
                              ncpus=args.ncpus)
    faces = [_parse_face(f) for f in args.faces] if args.faces else None
    run_crystal_workflow(args.cif, calc, args.out, faces=faces, max_index=args.max_index, n_faces=args.n_faces,
                         n_terminations=args.n_terminations, min_thickness=args.min_thickness,
                         min_lengths=args.min_lengths, max_atoms=args.max_atoms, vacuum=args.vacuum,
                         relax_slabs=not args.no_slab_relax, compute_eos=not args.no_eos, compute_elastic=args.elastic)


def _parse_face(text):
    """'1-10' -> (1, -1, 0); '1,-1,0' also works."""
    import re
    parts = text.split(",") if "," in text else re.findall(r"-?\d", text)
    if len(parts) != 3:
        raise ValueError(f"cannot read a Miller index from {text!r}")
    return tuple(int(p) for p in parts)


if __name__ == "__main__":
    main()
