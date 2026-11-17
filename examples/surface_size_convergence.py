"""How sensitive surface energies are to the size of the slab cell.

For each face and termination, from a finished crystal workflow run (bulk_relaxed.extxyz and
results.json give the bulk and its energy per molecule):

* thickness: gamma (unrelaxed and relaxed) for each number of layers in --layers, at the primitive
  in-plane cell, plus the linear-fit gamma from E_slab(n), which needs no bulk reference;
* lateral: relaxed gamma for each in-plane supercell in --supercells (e.g. 1x1 2x1 1x2 2x2) at
  the reference thickness. The unrelaxed gamma cannot change with a lateral supercell (energy is
  extensive), so any change is surface relaxation the small cell could not express;
* vacuum: unrelaxed gamma for each vacuum in --vacuums at the reference thickness.

Rows are written to --out after every slab, and a rerun skips rows already there.

    python examples/surface_size_convergence.py mleica_uma --faces 110 101 --terminations 0 \
        --model uma_data/uma-s-1p2p1.pt --ncpus 24
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read, write

from MartiniFuncs.crystals import build_slab, crystal_calculator, crystal_from_atoms, terminations
from MartiniFuncs.crystals.surfaces import EV_A2_TO_J_M2, surface_energy
from MartiniFuncs.crystals.workflow import _parse_face


def _supercell(text):
    m, n = (int(x) for x in text.lower().split("x"))
    return (m, n)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--faces", nargs="+", required=True,
                        help="faces, e.g. 110 or 11-1; '110:0,1' gives that face its own terminations")
    parser.add_argument("--terminations", type=int, nargs="+", default=[0])
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 4, 5, 6, 8])
    parser.add_argument("--thicknesses", type=float, nargs="+", default=None,
                        help="thickness series in Angstrom instead of --layers: for each face, the fewest layers "
                             "reaching each value (at least --min-layers), so faces with different d_hkl compare")
    parser.add_argument("--supercells", nargs="+", default=["1x1", "2x1", "1x2", "2x2"])
    parser.add_argument("--vacuums", type=float, nargs="+", default=[10.0, 15.0, 20.0, 30.0])
    parser.add_argument("--min-thickness", type=float, default=20.0, help="reference thickness for the lateral and vacuum series")
    parser.add_argument("--min-layers", type=int, default=3)
    parser.add_argument("--vacuum", type=float, default=15.0, help="vacuum for the thickness and lateral series")
    parser.add_argument("--max-atoms", type=int, default=2500, help="skip slabs larger than this")
    parser.add_argument("--skip", nargs="*", default=[], choices=["thickness", "lateral", "vacuum"])
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--n-fixed-layers", type=int, default=1)
    parser.add_argument("--rattle", type=float, default=0.0,
                        help="random displacement (Angstrom, std) of the free atoms before relaxing; without it "
                             "a supercell keeps the primitive cell's translational symmetry and cannot reconstruct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-structures", action="store_true", help="do not write each computed slab to <out>_structures/")
    parser.add_argument("--out", default="size_convergence.json")
    parser.add_argument("--calculator", default="uma")
    parser.add_argument("--model", default=None)
    parser.add_argument("--task", default="omc")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ncpus", type=int, default=None)
    args = parser.parse_args(argv)

    run = Path(args.run_dir)
    e_bulk = json.loads((run / "results.json").read_text())["bulk"]["energy_per_molecule"]
    crystal = crystal_from_atoms(read(run / "bulk_relaxed.extxyz"))
    calc = crystal_calculator(args.calculator, model=args.model, task=args.task, device=args.device, ncpus=args.ncpus)
    out_path = run / args.out
    done = json.loads(out_path.read_text()) if out_path.exists() else {"rows": []}
    done.update(bulk_energy_per_molecule=e_bulk, settings=vars(args))
    have = {(tuple(r["hkl"]), r["termination"], r["series"], r["layers"], tuple(r["supercell"]), r["vacuum"], r["relaxed"],
             r.get("rattle", 0.0)) for r in done["rows"]}
    rng = np.random.default_rng(args.seed)

    def run_one(hkl, term, series, layers, supercell, vacuum, relax):
        rattle = args.rattle if relax else 0.0
        key = (tuple(hkl), term.index, series, layers, tuple(supercell), vacuum, relax, rattle)
        if key in have:
            return
        slab = build_slab(crystal, hkl, layers=layers, termination=term, vacuum=vacuum,
                          supercell=None if tuple(supercell) == (1, 1) else tuple(supercell))
        if len(slab.atoms) > args.max_atoms:
            print(f"skip {hkl} T{term.index} {series} {layers}L {supercell}: {len(slab.atoms)} atoms", flush=True)
            return
        if rattle:
            free = np.setdiff1d(np.arange(len(slab.atoms)), slab.central_atoms(args.n_fixed_layers))
            slab.atoms.positions[free] += rng.normal(0.0, rattle, (len(free), 3))
        t0 = time.time()
        r = surface_energy(slab, calc, e_bulk, relax=relax, n_fixed_layers=args.n_fixed_layers, fmax=args.fmax,
                           steps=args.steps, crystal=crystal)
        row = {k: v for k, v in r.items() if k != "atoms"}
        if not args.no_structures:          # the slab as computed (relaxed when relax=True), for inspection
            sdir = out_path.with_name(out_path.stem + "_structures")
            sdir.mkdir(exist_ok=True)
            name = "{}_t{}_{}_{}L_{}x{}_vac{:g}{}.extxyz".format(
                "".join(map(str, hkl)).replace("-", "m"), term.index, series, layers, supercell[0], supercell[1],
                vacuum, f"_rattle{rattle:g}" if rattle else "")
            write(sdir / name, r["atoms"])
            row["file"] = f"{sdir.name}/{name}"
        row.update(hkl=list(hkl), termination=term.index, series=series, layers=layers, supercell=list(supercell),
                   vacuum=vacuum, relaxed=relax, rattle=rattle, thickness=slab.thickness, n_atoms=len(slab.atoms),
                   cell_lengths=[float(x) for x in slab.atoms.cell.lengths()[:2]], seconds=time.time() - t0)
        done["rows"].append(row)
        have.add(key)
        out_path.write_text(json.dumps(done, indent=1, default=str))
        print(f"{hkl} T{term.index} {series:9s} {layers}L {supercell[0]}x{supercell[1]} vac {vacuum:4.0f} rattle {rattle:.2f} "
              f"{len(slab.atoms):5d} atoms: gamma {r['gamma_unrelaxed_J_m2'] * 1000:6.1f} -> "
              f"{r['gamma_J_m2'] * 1000:6.1f} mJ/m^2 ({r['steps']} steps, {time.time() - t0:.0f} s)", flush=True)

    for face in args.faces:
        name, _, term_spec = face.partition(":")
        hkl = _parse_face(name)
        terms = terminations(crystal, hkl)
        chosen = [int(t) for t in term_spec.split(",")] if term_spec else args.terminations
        for ti in chosen:
            if ti >= len(terms):
                continue
            term = terms[ti]
            probe = build_slab(crystal, hkl, min_thickness=args.min_thickness, termination=term)
            ref_layers = max(args.min_layers, probe.layers)
            if args.thicknesses:
                layer_list = sorted({max(args.min_layers, build_slab(crystal, hkl, min_thickness=t, termination=term).layers)
                                     for t in args.thicknesses})
            else:
                layer_list = args.layers
            if "thickness" not in args.skip:
                for n in layer_list:
                    run_one(hkl, term, "thickness", n, (1, 1), args.vacuum, True)
            if "lateral" not in args.skip:
                for sc in args.supercells:
                    run_one(hkl, term, "lateral", ref_layers, _supercell(sc), args.vacuum, True)
            if "vacuum" not in args.skip:
                for v in args.vacuums:
                    run_one(hkl, term, "vacuum", ref_layers, (1, 1), v, False)

            rows = [r for r in done["rows"] if tuple(r["hkl"]) == tuple(hkl) and r["termination"] == term.index
                    and r["series"] == "thickness"]
            if len(rows) >= 3:              # linear fit E(n) = n*e_layer + 2*gamma*A: no bulk reference needed
                n = np.array([r["layers"] for r in rows], float)
                fits = {}
                for which in ("energy_unrelaxed", "energy"):
                    slope, intercept = np.polyfit(n, [r[which] for r in rows], 1)
                    fits[which] = {"gamma_J_m2": float(intercept / (2 * rows[0]["area"]) * EV_A2_TO_J_M2),
                                   "e_per_molecule": float(slope / (rows[0]["n_molecules"] / rows[0]["layers"]))}
                done.setdefault("linear_fits", {})[f"{''.join(map(str, hkl))}_t{term.index}"] = fits
                out_path.write_text(json.dumps(done, indent=1, default=str))


if __name__ == "__main__":
    main()
